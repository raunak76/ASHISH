import unittest

from solution import evaluate


class PublicTests(unittest.TestCase):
    def test_plain_values(self):
        out = evaluate({"A1": "5", "A2": "-2.5", "A3": "hello", "A4": ""})
        self.assertEqual(out, {"A1": 5.0, "A2": -2.5, "A3": "hello", "A4": ""})

    def test_arithmetic_precedence(self):
        out = evaluate({"A1": "=1 + 2 * 3", "A2": "=(1 + 2) * 3", "A3": "=-2 * -3"})
        self.assertEqual(out, {"A1": 7.0, "A2": 9.0, "A3": 6.0})

    def test_references(self):
        out = evaluate({"A1": "10", "B1": "=A1 * 2", "C1": "=B1 + a1"})
        self.assertEqual(out["B1"], 20.0)
        self.assertEqual(out["C1"], 30.0)

    def test_missing_reference_is_zero(self):
        self.assertEqual(evaluate({"A1": "=Z99 + 1"}), {"A1": 1.0})

    def test_functions(self):
        cells = {"A1": "1", "A2": "2", "A3": "3", "B1": "=SUM(A1:A3)",
                 "B2": "=avg(A1:A3)", "B3": "=MAX(A1:A3, 10)", "B4": "=MIN(A1:A3)"}
        out = evaluate(cells)
        self.assertEqual((out["B1"], out["B2"], out["B3"], out["B4"]), (6.0, 2.0, 10.0, 1.0))

    def test_division_by_zero(self):
        out = evaluate({"A1": "=1/0", "A2": "=A1 + 1"})
        self.assertEqual(out, {"A1": "#DIV/0!", "A2": "#DIV/0!"})

    def test_text_in_arithmetic(self):
        out = evaluate({"A1": "abc", "A2": "=A1 + 1", "A3": "=A1"})
        self.assertEqual(out["A2"], "#VALUE!")
        self.assertEqual(out["A3"], "abc")

    def test_simple_cycle(self):
        out = evaluate({"A1": "=B1", "B1": "=A1", "C1": "=A1 + 1"})
        self.assertEqual(out, {"A1": "#CYCLE!", "B1": "#CYCLE!", "C1": "#CYCLE!"})

    def test_parse_error(self):
        out = evaluate({"A1": "=1 +", "A2": "=FOO(1)", "A3": "=A1"})
        self.assertEqual(out, {"A1": "#ERROR!", "A2": "#ERROR!", "A3": "#ERROR!"})


if __name__ == "__main__":
    unittest.main()
