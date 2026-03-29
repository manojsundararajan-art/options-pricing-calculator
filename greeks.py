"""
Greeks Calculation Module

The Greeks are the sensitivities of option prices to changes in parameters.
They help traders manage risk.

Greeks:
    - Delta: How much option price changes if stock moves $1
    - Gamma: How fast Delta changes (acceleration)
    - Vega: How much option price changes if volatility moves 1%
    - Theta: How much option loses per day (time decay)
    - Rho: How much option changes if rates move 1%
"""

import numpy as np
from scipy.stats import norm
from black_scholes import BlackScholesCalculator

class GreeksCalculator:
    """
    Calculate the Greeks for European options.
    
    All Greeks are calculated using the Black-Scholes framework.
    """
    
    def __init__(self, S, K, r, sigma, T):
        """Initialize Greeks calculator."""
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        
        # Pre-compute once (used by multiple Greeks)
        self.calc = BlackScholesCalculator(S, K, r, sigma, T)
        self.d1, self.d2 = self.calc.compute_d1_d2()

    def delta_call(self):
        """
        Delta for a call: N(d1)
        
        Interpretation:
        - Delta = 0.6 means if stock moves $1, call moves $0.60
        - Delta = 0.5 means 50-50 chance of being in the money (roughly)
        - Delta ranges from 0 to 1 for calls
        """
        return norm.cdf(self.d1)
    
    def delta_put(self):
        """
        Delta for a put: -N(-d1)
        
        Interpretation:
        - Negative because puts go down when stock goes up
        - Delta ranges from -1 to 0 for puts
        """
        return -norm.cdf(-self.d1)
    
    def gamma(self):
        """
        Gamma: N'(d1) / (S·σ·√T)
        
        Interpretation:
        - How fast Delta changes
        - Always positive (convex payoff)
        - Highest for at-the-money options
        - Shows how much you're wrong if stock moves unexpectedly
        """
        sqrt_T = np.sqrt(self.T)
        N_prime_d1 = norm.pdf(self.d1)  # PDF, not CDF
        return N_prime_d1 / (self.S * self.sigma * sqrt_T)
    
    def vega(self, per_1_percent=True):
        """
        Vega: S·N'(d1)·√T
        
        Interpretation:
        - Sensitivity to volatility
        - Always positive (higher vol = higher option value)
        - Highest for at-the-money options
        
        Parameters
        ----------
        per_1_percent : bool
            If True, vega is for 1% volatility change
            If False, vega is raw (for decimal change)
        """
        sqrt_T = np.sqrt(self.T)
        N_prime_d1 = norm.pdf(self.d1)
        vega = self.S * N_prime_d1 * sqrt_T
        
        if per_1_percent:
            vega = vega / 100
        
        return vega
    
    def theta_call(self, per_day=True):
        """
        Theta for call: -[S·N'(d1)·σ] / (2·√T) - r·K·e^(-r·T)·N(d2)
        
        Interpretation:
        - How much option loses per day
        - Usually negative (time decay works against you)
        - Shows value erosion as expiration approaches
        
        Parameters
        ----------
        per_day : bool
            If True, theta per day (divide by 365)
            If False, theta annualized
        """
        sqrt_T = np.sqrt(self.T)
        N_prime_d1 = norm.pdf(self.d1)
        N_d2 = norm.cdf(self.d2)
        
        # Time decay component (always negative)
        term1 = -(self.S * N_prime_d1 * self.sigma) / (2 * sqrt_T)
        
        # Interest rate component
        discount = np.exp(-self.r * self.T)
        term2 = -self.r * self.K * discount * N_d2
        
        theta = term1 + term2
        
        if per_day:
            theta = theta / 365
        
        return theta
    
    def theta_put(self, per_day=True):
        """
        Theta for put: -[S·N'(d1)·σ] / (2·√T) + r·K·e^(-r·T)·N(-d2)
        
        Similar to call but interest rate component has opposite sign.
        """
        sqrt_T = np.sqrt(self.T)
        N_prime_d1 = norm.pdf(self.d1)
        N_neg_d2 = norm.cdf(-self.d2)
        
        term1 = -(self.S * N_prime_d1 * self.sigma) / (2 * sqrt_T)
        
        discount = np.exp(-self.r * self.T)
        term2 = self.r * self.K * discount * N_neg_d2
        
        theta = term1 + term2
        
        if per_day:
            theta = theta / 365
        
        return theta
    
    def rho_call(self, per_1_percent=True):
        """
        Rho for call: K·T·e^(-r·T)·N(d2)
        
        Interpretation:
        - Sensitivity to interest rates
        - Positive (higher rates = higher call prices)
        - Less important for short-dated options
        """
        N_d2 = norm.cdf(self.d2)
        discount = np.exp(-self.r * self.T)
        rho = self.K * self.T * discount * N_d2
        
        if per_1_percent:
            rho = rho / 100
        
        return rho
    
    def rho_put(self, per_1_percent=True):
        """
        Rho for put: -K·T·e^(-r·T)·N(-d2)
        
        Negative (higher rates = lower put prices)
        """
        N_neg_d2 = norm.cdf(-self.d2)
        discount = np.exp(-self.r * self.T)
        rho = -self.K * self.T * discount * N_neg_d2
        
        if per_1_percent:
            rho = rho / 100
        
        return rho
    
    def get_all_greeks(self, option_type='call'):
        """Get all Greeks at once."""
        if option_type.lower() == 'call':
            return {
                'delta': self.delta_call(),
                'gamma': self.gamma(),
                'vega': self.vega(),
                'theta': self.theta_call(),
                'rho': self.rho_call()
            }
        elif option_type.lower() == 'put':
            return {
                'delta': self.delta_put(),
                'gamma': self.gamma(),
                'vega': self.vega(),
                'theta': self.theta_put(),
                'rho': self.rho_put()
            }
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
def calculate_greeks(S, K, r, sigma, T, option_type='call'):
    """Quick function to get all Greeks."""
    calc = GreeksCalculator(S, K, r, sigma, T)
    return calc.get_all_greeks(option_type)


if __name__ == "__main__":
    greeks = GreeksCalculator(S=100, K=100, r=0.05, sigma=0.20, T=1.0)
    
    call_greeks = greeks.get_all_greeks('call')
    print("CALL GREEKS:")
    for greek, value in call_greeks.items():
        print(f"  {greek}: {value:.6f}")
    
    print("\nPUT GREEKS:")
    put_greeks = greeks.get_all_greeks('put')
    for greek, value in put_greeks.items():
        print(f"  {greek}: {value:.6f}")