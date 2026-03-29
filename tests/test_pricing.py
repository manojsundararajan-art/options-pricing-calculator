"""
Unit Tests for Options Pricing

Verify correctness of all calculations.
"""
import sys

sys.path.append('../options-pricing-calculator')

import unittest
import numpy as np
from black_scholes import BlackScholesCalculator
from greeks import GreeksCalculator



class TestBlackScholes(unittest.TestCase):
    """Test Black-Scholes pricing."""
    
    def setUp(self):
        """Set up test options."""
        self.S = 100
        self.K = 100
        self.r = 0.05
        self.sigma = 0.20
        self.T = 1.0
        self.calc = BlackScholesCalculator(self.S, self.K, self.r, self.sigma, self.T)
    
    def test_call_price_positive(self):
        """Call price should be positive."""
        call = self.calc.call_price()
        self.assertGreater(call, 0)
    
    def test_put_price_positive(self):
        """Put price should be positive."""
        put = self.calc.put_price()
        self.assertGreater(put, 0)
    
    def test_put_call_parity(self):
        """C - P = S - K·e^(-r·T)"""
        call = self.calc.call_price()
        put = self.calc.put_price()
        lhs = call - put
        rhs = self.S - self.K * np.exp(-self.r * self.T)
        self.assertAlmostEqual(lhs, rhs, places=10)
    
    def test_call_increases_with_spot(self):
        """Higher stock price → higher call price."""
        calc1 = BlackScholesCalculator(100, 100, 0.05, 0.20, 1.0)
        calc2 = BlackScholesCalculator(110, 100, 0.05, 0.20, 1.0)
        self.assertGreater(calc2.call_price(), calc1.call_price())
    
    def test_put_decreases_with_spot(self):
        """Higher stock price → lower put price."""
        calc1 = BlackScholesCalculator(100, 100, 0.05, 0.20, 1.0)
        calc2 = BlackScholesCalculator(110, 100, 0.05, 0.20, 1.0)
        self.assertGreater(calc1.put_price(), calc2.put_price())


class TestGreeks(unittest.TestCase):
    """Test Greeks calculations."""
    
    def setUp(self):
        """Set up test Greeks."""
        self.greeks = GreeksCalculator(100, 100, 0.05, 0.20, 1.0)
    
    def test_delta_call_in_range(self):
        """Call delta should be between 0 and 1."""
        delta = self.greeks.delta_call()
        self.assertGreaterEqual(delta, 0)
        self.assertLessEqual(delta, 1)
    
    def test_delta_put_in_range(self):
        """Put delta should be between -1 and 0."""
        delta = self.greeks.delta_put()
        self.assertGreaterEqual(delta, -1)
        self.assertLessEqual(delta, 0)
    
    def test_gamma_positive(self):
        """Gamma should always be positive."""
        gamma = self.greeks.gamma()
        self.assertGreater(gamma, 0)
    
    def test_vega_positive(self):
        """Vega should always be positive."""
        vega = self.greeks.vega()
        self.assertGreater(vega, 0)


if __name__ == '__main__':
    unittest.main()