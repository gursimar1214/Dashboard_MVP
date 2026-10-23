"""
sql_validator.py -- stage 5.

An independent, deterministic check on whatever stage 4 produced.
sqlglot parses the SQL into a real AST, so tables and columns are
checked structurally -- no regexes on raw SQL text.

Rejects anything that:
  - isn't exactly one statement, or isn't a SELECT
  - contains parameter placeholders (?, :name) -- none are bound
  - references a table outside schema_glossary.ALLOWED_TABLES
  - uses a column that does not exist in the table(s) the query uses
  - touches sqlite_master/sqlite_sequence or a pragma_* virtual table
  - has no LIMIT, or one above MAX_ROWS_RETURNED (capped, not rejected)

Derived tables (FROM (SELECT ...) AS sub), CTEs and window functions are
supported. A derived table is an exp.Subquery in the AST, not an exp.Table,
so its alias and the columns it exposes are tracked separately: every real
table inside it is still checked against the whitelist (the walk is over the
whole tree), but "sub.total" / a bare "total" defined by an inner SELECT is
accepted. The checks are flat (not per-scope); a typo inside a derived table
that happens to match an alias elsewhere in the query is left for SQLite to
report, and the pipeline's one retry sends that error back to the model.
"""
from __future__ import annotations

from typing import Optional

import sqlglot
from sqlglot import exp

from .config import MAX_ROWS_RETURNED
from .schema_glossary import ALLOWED_TABLES

_BANNED_TABLES = {"sqlite_master", "sqlite_sequence"}
_BANNED_FUNCTIONS = {"pragma_table_info", "pragma_index_list", "pragma_index_info"}


class SQLValidationError(Exception):
    pass


def _limit_value(tree: exp.Select) -> Optional[int]:
    limit_node = tree.args.get("limit")
    if limit_node is None:
        return None
    try:
        return int(limit_node.expression.this)
    except Exception:
        return None


def _columns_defined_by(select: exp.Expression) -> set[str]:
    """Names a SELECT list defines (aliases and plain column names), so
    ORDER BY / WHERE / outer references can use them."""
    out: set[str] = set()
    if isinstance(select, exp.Select):
        for e in select.selects:
            alias = e.alias_or_name
            if alias:
                out.add(alias.lower())
    return out


def _all_defined_names(tree: exp.Expression) -> set[str]:
    """Names defined by EVERY select in the query, including inner ones. A
    derived table's output columns (SELECT SUM(x) AS total ... ) are used by
    the outer query as plain 'total'."""
    out: set[str] = set()
    for sel in tree.find_all(exp.Select):
        out |= _columns_defined_by(sel)
    return out


def _derived_table_aliases(tree: exp.Expression) -> set[str]:
    """Aliases given to subqueries used as tables: FROM (SELECT ...) AS sub."""
    out: set[str] = set()
    for sq in tree.find_all(exp.Subquery):
        alias = sq.alias
        if alias:
            out.add(alias.lower())
    return out


_OPAQUE = "<opaque>"      # alias whose columns are validated elsewhere


def _check_select(tree: exp.Select, cte_names: set[str]) -> None:
    """Validate one SELECT body (not the outer WITH wrapper)."""

    # --- tables and aliases ------------------------------------------------
    alias_to_table: dict[str, str] = {}
    real_tables_used: set[str] = set()

    def bind(alias: str, target: str) -> None:
        # The walk is flat, so the same alias can be reused for different
        # tables in different parts of the query. Rather than guess which
        # one a column means, treat a clashing alias as opaque.
        prev = alias_to_table.get(alias)
        alias_to_table[alias] = target if prev in (None, target) else _OPAQUE

    for t in tree.find_all(exp.Table):
        if not t.name:
            continue
        real = t.name.lower()
        if real in cte_names:
            # A CTE reference: treat as an opaque relation, not a real table.
            bind((t.alias_or_name or t.name).lower(), f"<cte:{real}>")
            continue
        bind(real, real)
        bind((t.alias_or_name or t.name).lower(), real)
        real_tables_used.add(real)

    # Derived tables: FROM (SELECT ...) AS sub. Their inner real tables were
    # collected above; the alias itself only needs to be recognised.
    derived_aliases = _derived_table_aliases(tree)
    for alias in derived_aliases:
        bind(alias, _OPAQUE)

    unknown = real_tables_used - set(ALLOWED_TABLES)
    if unknown & _BANNED_TABLES:
        raise SQLValidationError("access to schema metadata tables is not allowed")
    if unknown:
        raise SQLValidationError(f"unknown table(s): {', '.join(sorted(unknown))}")

    # --- columns -----------------------------------------------------------
    # Unqualified columns are valid if they exist in any real table this
    # query uses, or match a name defined by any SELECT list in the query
    # (outer or inner -- derived tables expose their inner aliases), or the
    # query uses a CTE (we can't know the CTE's shape without recursing, so
    # we allow any unqualified column then -- the CTE body itself is
    # validated separately).
    columns_in_scope: set[str] = set()
    for tname in real_tables_used:
        columns_in_scope |= ALLOWED_TABLES[tname]
    select_aliases = _all_defined_names(tree)
    uses_cte = bool(cte_names)

    for col in tree.find_all(exp.Column):
        name = (col.name or "").lower()
        if not name or name == "*":
            continue
        qualifier = (col.table or "").lower()
        if qualifier:
            table = alias_to_table.get(qualifier)
            if table is None:
                raise SQLValidationError(f"unknown table or alias: '{qualifier}'")
            if table.startswith("<"):
                continue  # CTE / derived table / reused alias -- not a plain table
            if name not in ALLOWED_TABLES[table]:
                raise SQLValidationError(
                    f"column '{name}' does not exist in table '{table}'"
                )
        else:
            if uses_cte:
                continue
            if name not in columns_in_scope and name not in select_aliases:
                raise SQLValidationError(
                    f"column '{name}' is not in the table(s) this query uses "
                    f"({', '.join(sorted(real_tables_used)) or 'none'})"
                )


def validate_and_cap(sql: str) -> str:
    """Raises SQLValidationError, or returns a safe-to-run SQL string
    (identical to the input except the LIMIT is guaranteed <= the cap)."""
    sql = sql.strip().rstrip(";")
    if not sql:
        raise SQLValidationError("empty SQL")

    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except Exception as exc:
        raise SQLValidationError(f"not valid SQL ({exc})") from exc

    if len(statements) != 1:
        raise SQLValidationError("only a single SELECT statement is allowed")

    tree = statements[0]

    # Allow the top level to be a SELECT or a WITH-wrapped SELECT.
    if isinstance(tree, exp.With):
        inner = tree.this
        if not isinstance(inner, exp.Select):
            raise SQLValidationError("only SELECT statements are allowed")
        select_for_checks: exp.Select = inner
    elif isinstance(tree, exp.Select):
        select_for_checks = tree
    else:
        raise SQLValidationError("only SELECT statements are allowed")

    if list(tree.find_all(exp.Placeholder)):
        raise SQLValidationError(
            "query contains a parameter placeholder (?, :name, etc) -- "
            "no parameters are bound at execution, every value must be a literal"
        )

    for func in tree.find_all(exp.Anonymous):
        if func.name and func.name.lower() in _BANNED_FUNCTIONS:
            raise SQLValidationError(f"function '{func.name}' is not allowed")

    # Collect CTE names so real-table checks don't reject them.
    cte_names: set[str] = set()
    for cte in tree.find_all(exp.CTE):
        if cte.alias:
            cte_names.add(cte.alias.lower())

    # Validate each CTE body as its own SELECT, then the outer SELECT.
    for cte in tree.find_all(exp.CTE):
        body = cte.this
        if isinstance(body, exp.Select):
            _check_select(body, cte_names - {cte.alias.lower() if cte.alias else ""})

    _check_select(select_for_checks, cte_names)

    existing_limit = _limit_value(select_for_checks)
    needs_cap = existing_limit is None or existing_limit > MAX_ROWS_RETURNED
    if not needs_cap:
        return sql  # unchanged: preserve the model's exact formatting/quoting

    capped = select_for_checks.limit(MAX_ROWS_RETURNED)
    # If the top level was a WITH, re-attach the WITH wrapper.
    if isinstance(tree, exp.With):
        tree.set("this", capped)
        return tree.sql(dialect="sqlite")
    return capped.sql(dialect="sqlite")