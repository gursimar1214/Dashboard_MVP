# sql_grammar.gbnf
#
# A restrictive GBNF grammar for llama.cpp-compatible servers, limiting
# generation to a single read-only SELECT statement over this
# dashboard's five tables. Point a raw llama.cpp server at it:
#
#   llama-server -m qwen2.5-coder-7b-instruct-q4_k_m.gguf --grammar-file sql_grammar.gbnf
#
# and then generation literally cannot produce a token that falls
# outside this grammar -- stage 5's sqlglot check becomes a second,
# independent safety net rather than the only one.
#
# Why this isn't wired into the default Ollama pipeline: Ollama's
# public HTTP API only accepts a JSON-schema `format` (which Ollama
# itself compiles into a GBNF grammar internally) -- it does not accept
# an arbitrary hand-written grammar file like this one. See llm.py's
# module docstring. Treat this file as the stronger, optional path for
# anyone running their own llama.cpp server instead of/alongside Ollama.
#
# This is a deliberately small subset of SQL -- enough for the kinds of
# aggregate/lookup questions this dashboard answers, not a general SQL
# grammar. Extend it if you add new query shapes.

root         ::= ws select-stmt ws

select-stmt  ::= "SELECT" ws select-list ws "FROM" ws table (ws join-clause)* (ws where-clause)? (ws group-clause)? (ws order-clause)? (ws limit-clause)?

select-list  ::= select-item (ws "," ws select-item)*
select-item  ::= agg-fn | column | "*"
agg-fn       ::= ("COUNT" | "SUM" | "AVG" | "MIN" | "MAX") ws "(" ws ("DISTINCT" ws)? (column | "*") ws ")" (ws "AS" ws ident)?

table        ::= "customers" | "sessions" | "transactions" | "geo_data" | "marketing_campaigns"

join-clause  ::= "JOIN" ws table ws "ON" ws column ws "=" ws column

where-clause ::= "WHERE" ws condition (ws bool-op ws condition)*
condition    ::= column ws cmp-op ws value
bool-op      ::= "AND" | "OR"
cmp-op       ::= "=" | "!=" | ">=" | "<=" | ">" | "<"
value        ::= string | number

group-clause ::= "GROUP BY" ws column (ws "," ws column)*
order-clause ::= "ORDER BY" ws order-item (ws "," ws order-item)*
order-item   ::= column (ws ("ASC" | "DESC"))?
limit-clause ::= "LIMIT" ws number

column       ::= ident ("." ident)?
ident        ::= [a-zA-Z_][a-zA-Z0-9_]*
string       ::= "'" [^']* "'"
number       ::= [0-9]+ ("." [0-9]+)?
ws           ::= [ \t\n]*
