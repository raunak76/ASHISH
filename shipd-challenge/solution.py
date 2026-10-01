"""Reference solution: Mini Spreadsheet Evaluator."""

import re

_CELL_RE = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")
_NUMBER_RE = re.compile(r"^-?[0-9]+(\.[0-9]+)?$")
_TOKEN_RE = re.compile(r"\s*(?:([0-9]+(?:\.[0-9]+)?)|([A-Za-z]+[0-9]*)|([-+*/(),:]))")
_FUNCTIONS = {"SUM", "MIN", "MAX", "AVG"}


class _Err:
    def __init__(self, code):
        self.code = code


DIV0 = _Err("#DIV/0!")
VALUE = _Err("#VALUE!")
CYCLE = _Err("#CYCLE!")
ERROR = _Err("#ERROR!")


class _ParseError(Exception):
    pass


def _split_cell(name):
    m = _CELL_RE.match(name)
    if not m:
        return None
    col = 0
    for ch in m.group(1):
        col = col * 26 + (ord(ch) - 64)
    return int(m.group(2)), col


def _tokenize(src):
    tokens, pos = [], 0
    src = src.rstrip()
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise _ParseError()
        pos = m.end()
        num, word, op = m.groups()
        if num is not None:
            tokens.append(("NUM", float(num)))
        elif word is not None:
            up = word.upper()
            if up.isalpha():
                tokens.append(("FUNC", up))
            elif _split_cell(up):
                tokens.append(("REF", up))
            else:
                raise _ParseError()
        else:
            tokens.append(("OP", op))
    return tokens


class _Parser:
    def __init__(self, tokens):
        self.t = tokens
        self.i = 0

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else (None, None)

    def take(self):
        tok = self.peek()
        if tok[0] is None:
            raise _ParseError()
        self.i += 1
        return tok

    def expect(self, op):
        if self.take() != ("OP", op):
            raise _ParseError()

    def parse(self):
        node = self.expr()
        if self.i != len(self.t):
            raise _ParseError()
        return node

    def expr(self):
        node = self.term()
        while self.peek() in (("OP", "+"), ("OP", "-")):
            node = ("bin", self.take()[1], node, self.term())
        return node

    def term(self):
        node = self.unary()
        while self.peek() in (("OP", "*"), ("OP", "/")):
            node = ("bin", self.take()[1], node, self.unary())
        return node

    def unary(self):
        if self.peek() == ("OP", "-"):
            self.take()
            return ("neg", self.unary())
        return self.primary()

    def primary(self):
        kind, val = self.take()
        if kind == "NUM":
            return ("num", val)
        if kind == "REF":
            return ("ref", val)
        if kind == "FUNC":
            if val not in _FUNCTIONS:
                raise _ParseError()
            self.expect("(")
            args = [self.arg()]
            while self.peek() == ("OP", ","):
                self.take()
                args.append(self.arg())
            self.expect(")")
            return ("func", val, args)
        if (kind, val) == ("OP", "("):
            node = self.expr()
            self.expect(")")
            return node
        raise _ParseError()

    def arg(self):
        if self.peek()[0] == "REF" and self.peek(1) == ("OP", ":"):
            a = self.take()[1]
            self.take()
            kind, b = self.take()
            if kind != "REF":
                raise _ParseError()
            if self.peek() not in (("OP", ","), ("OP", ")")):
                raise _ParseError()
            return ("range", a, b)
        return self.expr()


def _range_cells(a, b, positions):
    (r1, c1), (r2, c2) = _split_cell(a), _split_cell(b)
    rlo, rhi = min(r1, r2), max(r1, r2)
    clo, chi = min(c1, c2), max(c1, c2)
    hits = [
        (pos, name)
        for name, pos in positions.items()
        if rlo <= pos[0] <= rhi and clo <= pos[1] <= chi
    ]
    hits.sort()
    return [name for _, name in hits]


def _collect_deps(node, positions, out):
    kind = node[0]
    if kind == "ref":
        if node[1] in positions:
            out.add(node[1])
    elif kind == "range":
        out.update(_range_cells(node[1], node[2], positions))
    elif kind == "neg":
        _collect_deps(node[1], positions, out)
    elif kind == "bin":
        _collect_deps(node[2], positions, out)
        _collect_deps(node[3], positions, out)
    elif kind == "func":
        for a in node[2]:
            _collect_deps(a, positions, out)


def _find_cycle_cells(deps):
    """Iterative Tarjan SCC; returns cells that lie on a cycle."""
    index, low, on_stack = {}, {}, set()
    stack, result, counter = [], set(), 0
    for root in deps:
        if root in index:
            continue
        work = [(root, iter(deps[root]))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(deps[w])))
                    advanced = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1 or v in deps[v]:
                    result.update(comp)
    return result


def _eval(node, values, positions):
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "ref":
        return values.get(node[1], 0.0) if node[1] in positions else 0.0
    if kind == "neg":
        v = _eval(node[1], values, positions)
        if isinstance(v, _Err):
            return v
        if isinstance(v, str):
            return VALUE
        return -v
    if kind == "bin":
        operands = []
        for sub in (node[2], node[3]):
            v = _eval(sub, values, positions)
            if isinstance(v, _Err):
                return v
            if isinstance(v, str):
                return VALUE
            operands.append(v)
        a, b = operands
        op = node[1]
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if b == 0:
            return DIV0
        return a / b
    if kind == "func":
        nums = []
        for arg in node[2]:
            if arg[0] == "range":
                for name in _range_cells(arg[1], arg[2], positions):
                    v = values[name]
                    if isinstance(v, _Err):
                        return v
                    if not isinstance(v, str):
                        nums.append(v)
            else:
                v = _eval(arg, values, positions)
                if isinstance(v, _Err):
                    return v
                if isinstance(v, str):
                    return VALUE
                nums.append(v)
        name = node[1]
        if name == "SUM":
            return float(sum(nums))
        if name == "AVG":
            return sum(nums) / len(nums) if nums else DIV0
        if not nums:
            return 0.0
        return min(nums) if name == "MIN" else max(nums)
    raise AssertionError(kind)


def evaluate(cells):
    positions = {name: _split_cell(name) for name in cells}
    values, asts, deps = {}, {}, {}

    for name, raw in cells.items():
        deps[name] = set()
        if raw.startswith("="):
            try:
                asts[name] = _Parser(_tokenize(raw[1:])).parse()
            except _ParseError:
                values[name] = ERROR
                continue
            _collect_deps(asts[name], positions, deps[name])
        elif _NUMBER_RE.match(raw.strip()):
            values[name] = float(raw.strip())
        else:
            values[name] = raw

    for name in _find_cycle_cells(deps):
        values[name] = CYCLE

    # Evaluate remaining formulas in dependency order (iterative post-order).
    for start in asts:
        if start in values:
            continue
        work = [(start, iter(deps[start]))]
        while work:
            v, it = work[-1]
            pushed = False
            for w in it:
                if w not in values:
                    work.append((w, iter(deps[w])))
                    pushed = True
                    break
            if pushed:
                continue
            work.pop()
            values[v] = _eval(asts[v], values, positions)

    return {k: (v.code if isinstance(v, _Err) else v) for k, v in values.items()}
