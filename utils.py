import numpy as np
import pandas as pd

def get_peaks_bottoms(prices: pd.Series, window: int = 5):
    peaks, bottoms = [], []

    # Iterate through price windows
    for i in range(window, len(prices) - window):
        price = prices.iloc[i]
        
        max_price = prices.iloc[i - window:i + window].max()
        min_price = prices.iloc[i - window:i + window].min()

        if price == max_price:
            peaks.append(price)

        if price == min_price:
            bottoms.append(price)

    # Return peak, bottom prices found
    return peaks, bottoms

def get_peaks_bottoms_weights(prices: pd.Series, peaks_bottoms: list[float], tolerance: float = 0.02):
    weights = []

    # Iterate through peaks and bottoms
    for peak_bottom in peaks_bottoms:
        # Calculate number of prices within tolerance of peak or bottom
        weight = len(prices[(prices >= peak_bottom * (1 - tolerance)) & (prices <= peak_bottom * (1 + tolerance))])
        weights.append(weight)

    # Return weighted peaks and bottoms
    return weights

def get_first_peak_bottom_above_below(close_btc, peaks_bottoms, weights):
    # Find first peak or bottom above and below close price
    above = next((pb for pb in peaks_bottoms if pb > close_btc), None)
    below = next((pb for pb in reversed(peaks_bottoms) if pb < close_btc), None)
    
    # Get weights of above and below peaks or bottoms
    above_weight = weights[peaks_bottoms.index(above)] if above else None
    below_weight = weights[peaks_bottoms.index(below)] if below else None
    
    return above, above_weight, below, below_weight

def get_closest_strike(strikes: list[str], value: float) -> float:
    # Convert strikes to floats
    strikes = [float(strike) for strike in strikes]
    # Return strike closest to value
    return min(strikes, key=lambda x:abs(x-value))