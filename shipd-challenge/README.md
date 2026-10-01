# Mini Spreadsheet Evaluator

## Summary

Implement `evaluate(cells)` in Python. It computes the final value of every cell in a small spreadsheet with formulas, cell references, ranges, aggregate functions, error propagation and cycle detection.

**Difficulty:** Hard. **Language:** Python 3.10+. Standard library only.

## Problem statement

You are given a dict `cells` that maps cell names (e.g. `"A1"`, `"AB12"`) to raw strings. Return a dict with **the same keys**, mapping each cell to its evaluated value. A value is a `float`, a `str` (text), or one of the error strings listed below.

### Cell names

- A cell name is one or more uppercase letters (the column) followed by a positive integer with no leading zeros (the row). Examples: `A1`, `Z99`, `AA1`.
- Columns are numbered like a spreadsheet: `A`=1 … `Z`=26, `AA`=27, `AB`=28, …
- Keys of `cells` are always valid, uppercase names. **References inside formulas are case-insensitive** (`=a1` refers to `A1`).

### Plain (non-formula) cells

A raw value that does not start with `=` is a plain cell.

- After stripping surrounding whitespace, if it matches `-?[0-9]+(\.[0-9]+)?`, it is a number and evaluates to a `float`.
- Otherwise it evaluates to its **original, unmodified string** (text). For example `""`, `"1e5"`, `"5."`, `"+5"` and `"#DIV/0!"` are all text.

### Formula cells

A raw value starting with `=` is a formula. The rest of the string is an expression. Whitespace between tokens is ignored.

```
expr    := term (('+' | '-') term)*
term    := unary (('*' | '/') unary)*
unary   := '-' unary | primary
primary := NUMBER | CELL | FUNC '(' arg (',' arg)* ')' | '(' expr ')'
arg     := CELL ':' CELL | expr
NUMBER  := [0-9]+ ('.' [0-9]+)?
```

- Supported functions (case-insensitive): `SUM`, `MIN`, `MAX`, `AVG`. They take at least one argument.
- A range `X:Y` is allowed **only as a whole function argument**. `=A1:B2` and `=SUM(A1:B2 + 1)` are parse errors.
- Unary plus is not supported (`=+1` is a parse error).
- Any formula that does not match the grammar evaluates to `#ERROR!`. That includes unknown functions, empty parentheses, a bad cell name like `A0`, and number forms like `.5` or `1e5`.

### Evaluation rules

1. **References.** A reference to a cell that is not in `cells` evaluates to `0.0`. A reference to an existing cell evaluates to that cell's value, which may be text or an error.
2. **Arithmetic** (`+ - * /` and unary `-`). Operands are evaluated left to right. The **first** operand that is an error or text decides the result: an error propagates as-is, and text gives `#VALUE!`. Dividing by `0` gives `#DIV/0!`.
3. **Parentheses** do not change the value. `=(A1)` with `A1` = `"hi"` evaluates to `"hi"`. A formula that is just a reference returns that cell's value, text included.
4. **Ranges** cover every existing cell inside the rectangle spanned by the two corners, whichever corner comes first. Cells are visited in **row-major order**: by row ascending, then by column ascending. Cells not in `cells` are skipped.
5. **Function arguments** are processed left to right:
   - A range: for each visited cell, an error is returned immediately, text is **skipped**, and numbers are collected.
   - Any other argument: it is evaluated. An error is returned, text gives `#VALUE!`, and a number is collected.
   - `SUM` of no numbers is `0.0`. `MIN`/`MAX` of no numbers is `0.0`. `AVG` of no numbers is `#DIV/0!`.
6. **Cycles are detected statically.** Build a dependency graph where each formula cell points to every existing cell it references, directly or through a range, regardless of whether that reference would actually be evaluated. Every cell that lies on a cycle (including a cell that references itself) evaluates to `#CYCLE!`. Other cells that depend on such cells get `#CYCLE!` through normal error propagation.
7. Errors only come from formulas. A plain cell whose text happens to be `"#DIV/0!"` is just text.

### Error values

| Value     | Meaning                                       |
|-----------|-----------------------------------------------|
| `#ERROR!` | the formula cannot be parsed                  |
| `#DIV/0!` | division by zero, or `AVG` with no numbers    |
| `#VALUE!` | text used where a number is required          |
| `#CYCLE!` | the cell lies on, or depends on, a cycle      |

### Performance

- Up to 20,000 cells, with dependency chains up to 20,000 deep. **The solution must not hit Python's recursion limit.**
- Ranges can be huge (e.g. `A1:A999999999`). Do not enumerate every coordinate in the rectangle.
- The whole test suite should run in a few seconds.

## Examples

```python
evaluate({"A1": "10", "B1": "=A1 * 2", "C1": "=SUM(A1:B1, 5)"})
# {"A1": 10.0, "B1": 20.0, "C1": 35.0}

evaluate({"A1": "word", "B1": "=SUM(A1)", "B2": "=SUM(A1:A1)"})
# {"A1": "word", "B1": "#VALUE!", "B2": 0.0}

evaluate({"A1": "=1/0 + B1", "B1": "=A1"})
# {"A1": "#CYCLE!", "B1": "#CYCLE!"}

evaluate({"A1": "txt", "B1": "=A1 + 1/0", "B2": "=1/0 + A1"})
# {"A1": "txt", "B1": "#VALUE!", "B2": "#DIV/0!"}
```

## Files

| File | Purpose |
|------|---------|
| `starter.py` | Function signature given to solvers |
| `solution.py` | Reference solution |
| `tests/test_public.py` | Public tests (9) shown to solvers |
| `tests/test_hidden.py` | Hidden tests (21) for edge cases, ordering, cycles and scale |

Run the tests:

```bash
python3 -m unittest tests/test_public.py tests/test_hidden.py
```
