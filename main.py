"""
Main Options Pricing Engine

High-level interface combining all modules.
"""

import pandas as pd
from black_scholes import BlackScholesCalculator
from greeks import GreeksCalculator
from visualizations import (
    plot_option_price_vs_spot,
    plot_greeks_vs_spot
)


class OptionsPricingEngine:
    """
    Main pricing engine combining Black-Scholes, Greeks, and visualizations.
    """
    
    def __init__(self, S, K, r, sigma, T):
        """Initialize engine."""
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        
        self.bs_calc = BlackScholesCalculator(S, K, r, sigma, T)
        self.greeks_calc = GreeksCalculator(S, K, r, sigma, T)
    
    def get_summary(self):
        """Get comprehensive summary of pricing and Greeks."""
        call_price = self.bs_calc.call_price()
        put_price = self.bs_calc.put_price()
        
        return {
            'parameters': {
                'S': self.S,
                'K': self.K,
                'r': self.r,
                'sigma': self.sigma,
                'T': self.T
            },
            'call': {
                'price': call_price,
                'delta': self.greeks_calc.delta_call(),
                'gamma': self.greeks_calc.gamma(),
                'vega': self.greeks_calc.vega(),
                'theta': self.greeks_calc.theta_call(),
                'rho': self.greeks_calc.rho_call()
            },
            'put': {
                'price': put_price,
                'delta': self.greeks_calc.delta_put(),
                'gamma': self.greeks_calc.gamma(),
                'vega': self.greeks_calc.vega(),
                'theta': self.greeks_calc.theta_put(),
                'rho': self.greeks_calc.rho_put()
            }
        }
    
    def print_summary(self):
        """Pretty print the summary."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("OPTIONS PRICING SUMMARY")
        print("="*70)
        
        params = summary['parameters']
        print(f"\nParameters:")
        print(f"  Stock Price (S):        ${params['S']:.2f}")
        print(f"  Strike Price (K):       ${params['K']:.2f}")
        print(f"  Risk-free Rate (r):     {params['r']*100:.2f}%")
        print(f"  Volatility (σ):         {params['sigma']*100:.2f}%")
        print(f"  Time to Expiry (T):     {params['T']:.4f} years")
        
        call = summary['call']
        print(f"\nCALL OPTION:")
        print(f"  Price:                  ${call['price']:.4f}")
        print(f"  Delta:                  {call['delta']:.4f}")
        print(f"  Gamma:                  {call['gamma']:.6f}")
        print(f"  Vega:                   {call['vega']:.4f}")
        print(f"  Theta:                  {call['theta']:.4f} (per day)")
        print(f"  Rho:                    {call['rho']:.4f}")
        
        put = summary['put']
        print(f"\nPUT OPTION:")
        print(f"  Price:                  ${put['price']:.4f}")
        print(f"  Delta:                  {put['delta']:.4f}")
        print(f"  Gamma:                  {put['gamma']:.6f}")
        print(f"  Vega:                   {put['vega']:.4f}")
        print(f"  Theta:                  {put['theta']:.4f} (per day)")
        print(f"  Rho:                    {put['rho']:.4f}")
        
        print("\n" + "="*70)


def price_options_from_dataframe(df):
    """Price multiple options from DataFrame."""
    results = []
    
    for idx, row in df.iterrows():
        engine = OptionsPricingEngine(
            S=row['S'],
            K=row['K'],
            r=row['r'],
            sigma=row['sigma'],
            T=row['T']
        )
        
        summary = engine.get_summary()
        call = summary['call']
        put = summary['put']
        
        results.append({
            'S': row['S'],
            'K': row['K'],
            'call_price': call['price'],
            'put_price': put['price'],
            'call_delta': call['delta'],
            'put_delta': put['delta'],
            'gamma': call['gamma'],
            'vega': call['vega'],
            'call_theta': call['theta'],
            'put_theta': put['theta']
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example 1: Single option
    print("\nExample 1: Single Option Pricing")
    engine = OptionsPricingEngine(S=100, K=105, r=0.05, sigma=0.20, T=1.0)
    engine.print_summary()
    
    # Example 2: Batch pricing
    print("\n\nExample 2: Batch Pricing")
    data = {
        'S': [100, 100, 100],
        'K': [95, 100, 105],
        'r': [0.05, 0.05, 0.05],
        'sigma': [0.20, 0.20, 0.20],
        'T': [1.0, 1.0, 1.0]
    }
    df = pd.DataFrame(data)
    results = price_options_from_dataframe(df)
    print(results[['K', 'call_price', 'put_price', 'call_delta']])
    
    # Example 3: Generate plots
    print("\n\nGenerating visualizations...")
    plot_option_price_vs_spot(K=100, r=0.05, sigma=0.20, T=1.0)
    plot_greeks_vs_spot(K=100, r=0.05, sigma=0.20, T=1.0)
    import matplotlib.pyplot as plt
    plt.show()

    