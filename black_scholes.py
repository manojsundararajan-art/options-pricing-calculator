"""
Black-Scholes Option Pricing Model

This module implements the Black-Scholes formula for pricing European options.

References:
    - Hull, J. C. (2017). Options, Futures, and Other Derivatives (10th ed.)
    - Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities
"""

import numpy as np
from scipy.stats import norm

class BlackScholesCalculator:
    """
    Black-Scholes option pricing calculator.
    
    Implements the Black-Scholes formula for pricing European options.
    
    Assumptions:
    -----------
    - European style options (exercise only at expiration)
    - Lognormal stock price distribution
    - Constant volatility
    - No arbitrage
    - No dividends
    - No transaction costs
    """
    
    def __init__(self, S, K, r, sigma, T):
        """
        Initialize Black-Scholes calculator.
        
        Parameters
        ----------
        S : float
            Current stock price
            Example: 100 means stock is trading at $100
            
        K : float
            Strike price (exercise price)
            Example: 105 means you can buy/sell at $105
            
        r : float
            Risk-free interest rate (annualized, as decimal)
            Example: 0.05 means 5% per year
            
        sigma : float
            Volatility (annualized, as decimal)
            Example: 0.20 means 20% per year
            This is the annualized standard deviation of stock returns
            
        T : float
            Time to expiration in years
            Example: 0.25 means 3 months (1/4 year)
            Example: 1.0 means 1 year
        """
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        
        # Validate inputs
        if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
            raise ValueError("S, K, sigma, T must all be positive. r can be >= 0.")
        if r < 0:
            raise ValueError("Interest rate cannot be negative")