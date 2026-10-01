import unittest

from solution import evaluate


class HiddenTests(unittest.TestCase):
    # --- Ranges ---------------------------------------------------------
    def test_range_skips_text_and_missing(self):
        out = evaluate({"A1": "4", "A2": "word", "A4": "6", "B1": "=AVG(A1:A5)"})
        self.assertEqual(out["B1"], 5.0)

    def test_direct_text_arg_is_value_error(self):
        out = evaluate({"A1": "word", "B1": "=SUM(A1)", "B2": "=SUM(A1:A1)"})
        self.assertEqual(out["B1"], "#VALUE!")
        self.assertEqual(out["B2"], 0.0)

    def test_reversed_range(self):
        out = evaluate({"A1": "1", "B2": "2", "C3": "=SUM(B2:A1)"})
        self.assertEqual(out["C3"], 3.0)

    def test_multi_letter_columns(self):
        out = evaluate({"Z1": "1", "AA1": "2", "AB1": "4", "AC1": "=SUM(Z1:AB1)"})
        self.assertEqual(out["AC1"], 7.0)

    def test_range_row_major_error_order(self):
        # Row-major visits B1 (row 1) before A2 (row 2); column-major would
        # hit A2 first and wrongly return #DIV/0!.
        cells = {"A2": "=1/0", "B1": "=Q1 + 1", "Q1": "x", "C1": "=SUM(A1:B2)"}
        out = evaluate(cells)
        self.assertEqual(out["B1"], "#VALUE!")
        self.assertEqual(out["C1"], "#VALUE!")

    def test_huge_range_is_fast(self):
        cells = {f"A{i}": str(i) for i in range(1, 2001)}
        cells["B1"] = "=SUM(A1:A999999999)"
        self.assertEqual(evaluate(cells)["B1"], 2001000.0)

    def test_empty_aggregates(self):
        out = evaluate({"A1": "=SUM(C1:C9)", "A2": "=MAX(C1:C9)",
                        "A3": "=MIN(C1:C9)", "A4": "=AVG(C1:C9)"})
        self.assertEqual(out, {"A1": 0.0, "A2": 0.0, "A3": 0.0, "A4": "#DIV/0!"})

    # --- Errors and evaluation order -----------------------------------
    def test_left_operand_decides(self):
        out = evaluate({"A1": "txt", "B1": "=A1 + 1/0", "B2": "=1/0 + A1"})
        self.assertEqual(out["B1"], "#VALUE!")
        self.assertEqual(out["B2"], "#DIV/0!")

    def test_plain_error_text_is_just_text(self):
        out = evaluate({"A1": "#DIV/0!", "B1": "=A1", "B2": "=SUM(A1:A1, 2)"})
        self.assertEqual(out, {"A1": "#DIV/0!", "B1": "#DIV/0!", "B2": 2.0})

    def test_parenthesised_text_passes_through(self):
        self.assertEqual(evaluate({"A1": "hi", "B1": "=((A1))"})["B1"], "hi")

    def test_negated_text(self):
        self.assertEqual(evaluate({"A1": "hi", "B1": "=-A1"})["B1"], "#VALUE!")

    # --- Cycles ----------------------------------------------------------
    def test_self_reference(self):
        self.assertEqual(evaluate({"A1": "=A1"}), {"A1": "#CYCLE!"})

    def test_self_reference_via_range(self):
        out = evaluate({"A1": "1", "A2": "=SUM(A1:A3)"})
        self.assertEqual(out["A2"], "#CYCLE!")

    def test_cycle_hidden_behind_earlier_error(self):
        # Cycle detection is static: B1 is never "reached" at runtime, but the
        # cells are still on a cycle.
        out = evaluate({"A1": "=1/0 + B1", "B1": "=A1"})
        self.assertEqual(out, {"A1": "#CYCLE!", "B1": "#CYCLE!"})

    def test_cell_pointing_into_cycle_is_not_itself_cycle_member(self):
        out = evaluate({"A1": "=B1", "B1": "=A1", "C1": "=SUM(D1:D2)", "D1": "5"})
        self.assertEqual(out["C1"], 5.0)

    # --- Parsing ---------------------------------------------------------
    def test_parse_errors(self):
        bad = ["=", "=+1", "=1 2", "=A0", "=.5", "=1e5", "=SUM()", "=A1:B2",
               "=SUM(A1:B2 + 1)", "=SUM", "=(1", "=1)"]
        cells = {f"A{i + 1}": f for i, f in enumerate(bad)}
        out = evaluate(cells)
        for k in cells:
            self.assertEqual(out[k], "#ERROR!", (k, cells[k]))

    def test_whitespace_and_case(self):
        out = evaluate({"A1": "  7 ", "B1": "=  sUm ( a1 , 3 )  "})
        self.assertEqual(out, {"A1": 7.0, "B1": 10.0})

    def test_double_negation(self):
        self.assertEqual(evaluate({"A1": "=--5"}), {"A1": 5.0})

    def test_number_like_text(self):
        out = evaluate({"A1": "1e5", "A2": "5.", "A3": "+5", "B1": "=A1"})
        self.assertEqual(out, {"A1": "1e5", "A2": "5.", "A3": "+5", "B1": "1e5"})

    # --- Scale -----------------------------------------------------------
    def test_long_chain_no_recursion_error(self):
        n = 20000
        cells = {f"A{i}": f"=A{i + 1} + 1" for i in range(1, n)}
        cells[f"A{n}"] = "0"
        self.assertEqual(evaluate(cells)["A1"], float(n - 1))

    def test_long_cycle(self):
        n = 20000
        cells = {f"A{i}": f"=A{i % n + 1}" for i in range(1, n + 1)}
        out = evaluate(cells)
        self.assertTrue(all(v == "#CYCLE!" for v in out.values()))


if __name__ == "__main__":
    unittest.main()
