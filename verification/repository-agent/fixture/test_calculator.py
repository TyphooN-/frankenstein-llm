import unittest

from calculator import add


class CalculatorTests(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(7, add(3, 4))

    def test_negative(self):
        self.assertEqual(-5, add(-2, -3))

    def test_mixed(self):
        self.assertEqual(-1, add(2, -3))


if __name__ == "__main__":
    unittest.main()
