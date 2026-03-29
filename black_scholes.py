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
        

    def compute_d1_d2(self):
        """
        Compute d1 and d2 from Black-Scholes formula.
        
        These are intermediate values used in the pricing formula.
        
        Formula
        -------
        d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
        d2 = d1 - σ·√T
        
        Returns
        -------
        d1 : float
            First parameter
        d2 : float
            Second parameter
            
        Example
        -------
        If S=100, K=100, r=0.05, sigma=0.20, T=1.0:
        d1 ≈ 0.3418
        d2 ≈ 0.1418
        """
        # Calculate square root of time (appears in both formulas)
        sqrt_T = np.sqrt(self.T)
        
        # Calculate ln(S/K) - the log ratio of spot to strike
        ln_S_K = np.log(self.S / self.K)
        
        # Calculate d1 numerator: ln(S/K) + (r + σ²/2)·T
        # Breaking it down:
        #   ln(S/K) - log ratio of stock price to strike
        #   r*T - risk-free return term
        #   0.5 * sigma^2 * T - variance adjustment term
        numerator = ln_S_K + (self.r + 0.5 * self.sigma ** 2) * self.T
        
        # Calculate d1 denominator: σ·√T
        denominator = self.sigma * sqrt_T
        
        # Calculate d1
        d1 = numerator / denominator
        
        # Calculate d2 = d1 - σ·√T
        d2 = d1 - self.sigma * sqrt_T
        
        return d1, d2

    def call_price(self):
        """
        Calculate European call option price using Black-Scholes.
        
        A call option gives the holder the right to BUY the stock at K.
        If stock is above K at expiration, it's worth S - K.
        
        Formula
        -------
        C = S·N(d1) - K·e^(-r·T)·N(d2)
        
        Where:
            N(x) = Cumulative standard normal distribution
            e^(-r·T) = Discount factor (present value)
        
        Intuition
        ---------
        - S·N(d1): Expected value of owning stock if option is exercised
        - K·e^(-r·T)·N(d2): Expected value of strike payment, discounted
        - The difference is what the option is worth today
        
        Returns
        -------
        float
            Price of the call option
            
        Example
        -------
        For S=100, K=100, r=0.05, sigma=0.20, T=1.0:
        Call price ≈ $10.45
        """
        # Get d1 and d2
        d1, d2 = self.compute_d1_d2()
        
        # Get N(d1) and N(d2) from normal distribution CDF
        # N(x) is the probability that a normal random variable is <= x
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        
        # Calculate discount factor e^(-r·T)
        # This converts future payment K to present value
        discount_factor = np.exp(-self.r * self.T)
        
        # Apply Black-Scholes formula
        call = self.S * N_d1 - self.K * discount_factor * N_d2
        
        return call

    def put_price(self):
        """
        Calculate European put option price using Black-Scholes.
        
        A put option gives the holder the right to SELL the stock at K.
        If stock is below K at expiration, it's worth K - S.
        
        Formula
        -------
        P = K·e^(-r·T)·N(-d2) - S·N(-d1)
        
        Alternative (via put-call parity):
        P = C - S + K·e^(-r·T)
        
        We use the direct formula for precision.
        
        Returns
        -------
        float
            Price of the put option
            
        Example
        -------
        For S=100, K=100, r=0.05, sigma=0.20, T=1.0:
        Put price ≈ $5.57
        """
        d1, d2 = self.compute_d1_d2()
        
        # N(-d1) and N(-d2) are probabilities in the opposite tail
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)
        
        discount_factor = np.exp(-self.r * self.T)
        
        # Apply Black-Scholes formula for put
        put = self.K * discount_factor * N_neg_d2 - self.S * N_neg_d1
        
        return put

    def verify_put_call_parity(self):
        """
        Verify put-call parity relationship.
        
        Relationship
        -----------
        C - P = S - K·e^(-r·T)
        
        This MUST hold by arbitrage principle.
        If it doesn't, there's money on the table.
        
        Returns
        -------
        dict
            Verification results
        """
        call = self.call_price()
        put = self.put_price()
        
        # Left side: Call - Put
        lhs = call - put
        
        # Right side: S - K·e^(-r·T)
        rhs = self.S - self.K * np.exp(-self.r * self.T)
        
        # Check if they're equal (within numerical tolerance)
        tolerance = 1e-10
        parity_holds = np.isclose(lhs, rhs, atol=tolerance)
        
        return {
            'call_price': call,
            'put_price': put,
            'lhs': lhs,
            'rhs': rhs,
            'parity_holds': parity_holds
        }

def calculate_call_price(S, K, r, sigma, T):
    """Quick function to price a call."""
    calc = BlackScholesCalculator(S, K, r, sigma, T)
    return calc.call_price()


def calculate_put_price(S, K, r, sigma, T):
    """Quick function to price a put."""
    calc = BlackScholesCalculator(S, K, r, sigma, T)
    return calc.put_price()


def calculate_option_prices(S, K, r, sigma, T):
    """Get both call and put prices."""
    calc = BlackScholesCalculator(S, K, r, sigma, T)
    return {
        'call': calc.call_price(),
        'put': calc.put_price()
    }

if __name__ == "__main__":
    # Test convenience functions
    call = calculate_call_price(100, 100, 0.05, 0.20, 1.0)
    put = calculate_put_price(100, 100, 0.05, 0.20, 1.0)
    both = calculate_option_prices(100, 100, 0.05, 0.20, 1.0)
    
    print(f"Call: ${call:.2f}")
    print(f"Put: ${put:.2f}")
    print(f"Both: {both}")