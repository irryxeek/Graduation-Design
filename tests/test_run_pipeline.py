import argparse
import unittest

from src.run_pipeline import parse_var_weights


class RunPipelineArgsTest(unittest.TestCase):
    def test_parse_var_weights_accepts_csv(self):
        self.assertEqual(parse_var_weights("1,1,4"), [1.0, 1.0, 4.0])

    def test_parse_var_weights_returns_none_for_empty(self):
        self.assertIsNone(parse_var_weights(""))
        self.assertIsNone(parse_var_weights(None))

    def test_parse_var_weights_rejects_non_positive_values(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_var_weights("1,0,4")


if __name__ == "__main__":
    unittest.main()
