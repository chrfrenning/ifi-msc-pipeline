import numpy as np
from scipy.signal import savgol_filter
from .myconfig import SAVGOL_WINDOW_SIZE, SAVGOL_POLYORDER



#
# Normalization
#

def standardize_signal(signal):
    return (signal - signal.mean()) / signal.std()

def normalize_signal(signal):
    return (signal - signal.min()) / (signal.max() - signal.min())



#
# Baseline correction
#

def baseline_correction(somnofy_df, column_name):
    #return savgol_filter(somnofy_df[column_name], SAVGOL_WINDOW_SIZE, SAVGOL_POLYORDER)
    y = somnofy_df[column_name].values
    baseline = savgol_filter(y, SAVGOL_WINDOW_SIZE, SAVGOL_POLYORDER)
    return y - baseline



#
# Sliding cross correlation for alignment of two signals with imperfect time sync
#

# from 18_cross_correlation.py by farzanmn
def estimate_lag(signal1, signal2, max_lag=2000):
    full_corr = np.correlate(signal1, signal2, mode='full')
    center = len(full_corr) // 2
    restricted_corr = full_corr[center - max_lag : center + max_lag + 1]
    lag = restricted_corr.argmax() - max_lag
    return lag

# somnofy recordings start and stop, and we have multiple of these per continous psg recording,
# so we need to try and align as best as possible the shorter recordings to the continous psg recording
# by searching and maximizing the correlation coefficient between the two signals
def align_short_recording_to_continuous(continuous_signal, short_signal, approximate_start_time, search_window=300, max_lag=100):
    # Define search region in continuous signal
    search_start = max(0, approximate_start_time - search_window//2)
    search_end = min(len(continuous_signal) - len(short_signal), approximate_start_time + search_window//2)
    
    best_correlation = -np.inf
    best_start_idx = approximate_start_time
    
    # Search within the window
    for start_idx in range(search_start, search_end):
        # Extract segment from continuous signal
        continuous_segment = continuous_signal[start_idx:start_idx + len(short_signal)]
        
        if len(continuous_segment) != len(short_signal):
            continue
            
        # Use estimate_lag function for fine alignment
        lag = estimate_lag(continuous_segment, short_signal, max_lag=max_lag)
        
        # Apply the lag and calculate correlation
        if lag > 0:
            aligned_continuous = continuous_segment[lag:]
            aligned_short = short_signal[:-lag] if lag < len(short_signal) else []
        elif lag < 0:
            aligned_continuous = continuous_segment[:lag]
            aligned_short = short_signal[-lag:]
        else:
            aligned_continuous = continuous_segment
            aligned_short = short_signal
            
        if len(aligned_continuous) > 0 and len(aligned_short) > 0:
            # Calculate correlation coefficient
            try:
                # Standardize signals
                std_continuous = np.std(aligned_continuous)
                std_short = np.std(aligned_short)
                
                if std_continuous == 0 and std_short == 0:
                    # Both signals are constant
                    correlation = 1.0 if np.mean(aligned_continuous) == np.mean(aligned_short) else 0.0
                elif std_continuous == 0 or std_short == 0:
                    # One signal is constant
                    correlation = 0.0
                else:
                    # Normal correlation calculation
                    correlation = np.corrcoef(aligned_continuous, aligned_short)[0, 1]
                    
                # Handle NaN
                if np.isnan(correlation):
                    correlation = 0.0
                    
            except Exception:
                correlation = 0.0
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_start_idx = start_idx + lag
    
    return best_start_idx, best_correlation