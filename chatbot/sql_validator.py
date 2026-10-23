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
table inside it is still checked against the whitelist. Direct FROM/JOIN
sources are also examined per SELECT scope so a bare column shared by joined
tables is rejected before SQLite execution; the pipeline retries with that
specific correction. Other SQL errors still use the normal execution retry.
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


def _output_columns(select: exp.Expression) -> set[str]:
    """Names exposed by a CTE or derived SELECT, when explicitly selected."""
    if isinstance(select, exp.With):
        select = select.this
    return _columns_defined_by(select)


def _direct_sources(
    select: exp.Select,
    cte_columns: dict[str, set[str]],
) -> list[tuple[str, set[str]]]:
    """Return the exposed columns for this SELECT's immediate FROM/JOINs.

    Nested SELECTs are intentionally kept in their own scope. The older flat
    table scan is still used for the whitelist check; this scoped view exists
    to identify unqualified names that SQLite would find ambiguous.
    """
    relations: list[exp.Expression] = []
    from_clause = select.args.get("from_")
    if from_clause is not None:
        first = from_clause.this
        if first is not None:
            relations.append(first)
        relations.extend(from_clause.expressions or [])
    relations.extend(join.this for join in (select.args.get("joins") or []) if join.this is not None)

    sources: list[tuple[str, set[str]]] = []
    for relation in relations:
        while isinstance(relation, exp.Paren):
            relation = relation.this
        if isinstance(relation, exp.Table):
            name = relation.name.lower()
            columns = cte_columns.get(name, ALLOWED_TABLES.get(name))
            alias = (relation.alias_or_name or name).lower()
        elif isinstance(relation, exp.Subquery):
            columns = _output_columns(relation.this)
            alias = (relation.alias_or_name or "").lower()
        else:
            # Unknown relation shapes remain covered by normal validation and
            # SQLite execution. Do not guess their output columns here.
            continue
        if alias and columns is not None:
            sources.append((alias, columns))
    return sources


def _immediate_base_tables(select: exp.Select, cte_names: set[str]) -> set[str]:
    """Base tables joined directly in this SELECT scope (not inside CTEs)."""
    relations: list[exp.Expression] = []
    from_clause = select.args.get("from_")
    if from_clause is not None:
        if from_clause.this is not None:
            relations.append(from_clause.this)
        relations.extend(from_clause.expressions or [])
    relations.extend(
        join.this for join in (select.args.get("joins") or [])
        if join.this is not None
    )
    result: set[str] = set()
    for relation in relations:
        while isinstance(relation, exp.Paren):
            relation = relation.this
        if isinstance(relation, exp.Table):
            name = relation.name.lower()
            if name not in cte_names:
                result.add(name)
    return result


def _owning_select(node: exp.Expression) -> exp.Select | None:
    parent = node.parent
    while parent is not None and not isinstance(parent, exp.Select):
        parent = parent.parent
    return parent if isinstance(parent, exp.Select) else None


def _check_known_join_fanout(tree: exp.Expression, cte_names: set[str]) -> None:
    """Reject the known campaign-budget x raw-session fan-out deterministically.

    This targeted guard only rejects a SUM(campaign_budget) in a SELECT scope
    that directly joins marketing_campaigns to raw sessions. Pre-aggregated
    session CTEs or derived tables remain valid.
    """
    for select in tree.find_all(exp.Select):
        direct_tables = _immediate_base_tables(select, cte_names)
        if not {"marketing_campaigns", "sessions"}.issubset(direct_tables):
            continue

        sums_budget = any(
            _owning_select(agg) is select
            and any(
                (column.name or "").lower() == "campaign_budget"
                and _owning_select(column) is select
                for column in agg.find_all(exp.Column)
            )
            for agg in select.find_all(exp.Sum)
        )
        if sums_budget:
            raise SQLValidationError(
                "campaign budget is multiplied by the raw sessions join; "
                "aggregate sessions to one row per campaign_id in a CTE or "
                "derived table before joining marketing_campaigns"
            )

_OPAQUE = "<opaque>"      # alias whose columns are validated elsewhere


def _check_select(
    tree: exp.Select,
    cte_names: set[str],
    cte_columns: dict[str, set[str]],
) -> None:
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
    scope_sources: dict[int, list[tuple[str, set[str]]]] = {}

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
            # Walk up to the SELECT that owns this column so nested queries do
            # not make one another's table columns look ambiguous.
            owner = col.parent
            while owner is not None and not isinstance(owner, exp.Select):
                owner = owner.parent
            if owner is not None:
                owner_id = id(owner)
                if owner_id not in scope_sources:
                    scope_sources[owner_id] = _direct_sources(owner, cte_columns)
                scoped_sources = scope_sources[owner_id]
            else:
                scoped_sources = []
            if scoped_sources:
                matches = [alias for alias, columns in scoped_sources if name in columns]
                if len(matches) > 1:
                    raise SQLValidationError(
                        f"ambiguous unqualified column '{name}' "
                        f"(matches {', '.join(matches)}); qualify it with its table alias"
                    )
                if len(matches) == 1:
                    continue
            if uses_cte:
                if scoped_sources and name not in select_aliases:
                    raise SQLValidationError(
                        f"unqualified column '{name}' cannot be resolved safely through a CTE "
                        "or derived table; explicitly project it and qualify its source"
                    )
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
    cte_columns: dict[str, set[str]] = {}
    for cte in tree.find_all(exp.CTE):
        if cte.alias:
            name = cte.alias.lower()
            cte_names.add(name)
            cte_columns[name] = _output_columns(cte.this)

    # Validate each CTE body as its own SELECT, then the outer SELECT.
    for cte in tree.find_all(exp.CTE):
        body = cte.this
        if isinstance(body, exp.Select):
            _check_select(
                body,
                cte_names - {cte.alias.lower() if cte.alias else ""},
                cte_columns,
            )

    _check_select(select_for_checks, cte_names, cte_columns)
    _check_known_join_fanout(tree, cte_names)

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
