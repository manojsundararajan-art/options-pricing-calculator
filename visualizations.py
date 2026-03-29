"""
Visualization Module for Options Pricing

Creates plots to visualize option prices and Greeks.
"""

import numpy as np
import matplotlib.pyplot as plt
from black_scholes import BlackScholesCalculator
from greeks import GreeksCalculator

def plot_option_price_vs_spot(K, r, sigma, T, spot_range=None, figsize=(12, 6)):
    """
    Plot call and put prices across a range of spot prices.
    
    This shows the nonlinear relationship between spot price and option value.
    """
    if spot_range is None:
        spot_range = (K * 0.7, K * 1.3)
    
    # Generate spot prices
    spots = np.linspace(spot_range[0], spot_range[1], 100)
    
    call_prices = []
    put_prices = []
    call_payoff = []
    put_payoff = []
    
    # Calculate for each spot price
    for S in spots:
        calc = BlackScholesCalculator(S, K, r, sigma, T)
        call_prices.append(calc.call_price())
        put_prices.append(calc.put_price())
        call_payoff.append(max(S - K, 0))
        put_payoff.append(max(K - S, 0))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Call plot
    ax1.plot(spots, call_prices, 'b-', linewidth=2, label='Call Price')
    ax1.plot(spots, call_payoff, 'r--', linewidth=1, label='Payoff at Expiration')
    ax1.axvline(K, color='k', linestyle=':', alpha=0.5, label='Strike')
    ax1.fill_between(spots, 0, call_prices, alpha=0.1, color='blue')
    ax1.set_xlabel('Stock Price ($)')
    ax1.set_ylabel('Option Price ($)')
    ax1.set_title('Call Option Price vs Stock Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Put plot
    ax2.plot(spots, put_prices, 'g-', linewidth=2, label='Put Price')
    ax2.plot(spots, put_payoff, 'r--', linewidth=1, label='Payoff at Expiration')
    ax2.axvline(K, color='k', linestyle=':', alpha=0.5, label='Strike')
    ax2.fill_between(spots, 0, put_prices, alpha=0.1, color='green')
    ax2.set_xlabel('Stock Price ($)')
    ax2.set_ylabel('Option Price ($)')
    ax2.set_title('Put Option Price vs Stock Price')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def plot_greeks_vs_spot(K, r, sigma, T, spot_range=None, figsize=(14, 10)):
    """
    Plot all Greeks across a range of spot prices.
    
    Shows how Greeks change as the stock price moves.
    """
    if spot_range is None:
        spot_range = (K * 0.7, K * 1.3)
    
    spots = np.linspace(spot_range[0], spot_range[1], 100)
    
    # Storage for each Greek
    deltas_call, deltas_put, gammas, vegas, thetas_call, thetas_put = [], [], [], [], [], []
    rhos_call, rhos_put = [], []
    
    for S in spots:
        greeks = GreeksCalculator(S, K, r, sigma, T)
        deltas_call.append(greeks.delta_call())
        deltas_put.append(greeks.delta_put())
        gammas.append(greeks.gamma())
        vegas.append(greeks.vega())
        thetas_call.append(greeks.theta_call())
        thetas_put.append(greeks.theta_put())
        rhos_call.append(greeks.rho_call())
        rhos_put.append(greeks.rho_put())
    
    # 6 subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # Delta
    axes[0, 0].plot(spots, deltas_call, 'b-', linewidth=2, label='Call')
    axes[0, 0].plot(spots, deltas_put, 'r-', linewidth=2, label='Put')
    axes[0, 0].axvline(K, color='k', linestyle=':', alpha=0.5)
    axes[0, 0].set_ylabel('Delta')
    axes[0, 0].set_title('Delta vs Stock Price')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gamma
    axes[0, 1].plot(spots, gammas, 'g-', linewidth=2)
    axes[0, 1].axvline(K, color='k', linestyle=':', alpha=0.5)
    axes[0, 1].set_ylabel('Gamma')
    axes[0, 1].set_title('Gamma vs Stock Price')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Vega
    axes[1, 0].plot(spots, vegas, 'purple', linewidth=2)
    axes[1, 0].axvline(K, color='k', linestyle=':', alpha=0.5)
    axes[1, 0].set_ylabel('Vega')
    axes[1, 0].set_title('Vega vs Stock Price')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Theta
    axes[1, 1].plot(spots, thetas_call, 'b-', linewidth=2, label='Call')
    axes[1, 1].plot(spots, thetas_put, 'r-', linewidth=2, label='Put')
    axes[1, 1].axvline(K, color='k', linestyle=':', alpha=0.5)
    axes[1, 1].set_ylabel('Theta')
    axes[1, 1].set_title('Theta vs Stock Price (per day)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Rho
    axes[2, 0].plot(spots, rhos_call, 'b-', linewidth=2, label='Call')
    axes[2, 0].plot(spots, rhos_put, 'r-', linewidth=2, label='Put')
    axes[2, 0].axvline(K, color='k', linestyle=':', alpha=0.5)
    axes[2, 0].set_xlabel('Stock Price ($)')
    axes[2, 0].set_ylabel('Rho')
    axes[2, 0].set_title('Rho vs Stock Price')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Hide the last subplot
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    return fig, axes


if __name__ == "__main__":
    print("Generating plots...")
    
    # fig1, _ = plot_option_price_vs_spot(K=100, r=0.05, sigma=0.20, T=1.0)
    # fig1.savefig('option_prices.png', dpi=150)
    # print("Saved: option_prices.png")
    
    fig2, _ = plot_greeks_vs_spot(K=100, r=0.05, sigma=0.20, T=1.0)
    fig2.savefig('./figures/greeks.png', dpi=150)
    print("Saved: greeks.png")
    
    plt.show()