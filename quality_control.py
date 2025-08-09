from scipy.fft import rfft, rfftfreq
from scipy.stats import pearsonr
import numpy as np

def evaluate_sequenz(reference, filtered, fs=1000):
    # --- RMSE ---
    rmse = np.sqrt(np.mean((reference- filtered) ** 2))
    mse = np.mean((reference - filtered) ** 2)

    # --- SNR (dB) ---
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean((filtered - reference) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else np.inf

    # --- Korrelation ---
    r, _ = pearsonr(reference, filtered)

    

    return rmse,mse,snr,r,

def evaluate_filtering(reference, filtered, settings):
    if reference is None:
        return {
            "rmse": -1,
            "mse": -1,
            "snr": -1,
            "correlation": -1
        }



    samples_active = int((settings.active_ms / 1000) * settings.fs)
    samples_pause = int((settings.pause_ms / 1000) * settings.fs)
    block_len = samples_active + samples_pause
    total_samples = len(filtered)

    rmse_list = []
    mse_list = []
    snr_list = []
    corr_list = []

    

    for start in range(0, total_samples, block_len):
        end = start + samples_active
        if end > total_samples:
            break

        ref_seg = reference[start:end]
        filt_seg = filtered[start:end]

        rmse, mse, snr, r = evaluate_sequenz(ref_seg, filt_seg, settings.fs)

        # Speichern in Listen
        rmse_list.append(rmse)
        mse_list.append(mse)
        snr_list.append(snr)
        corr_list.append(r)

        t_start = start / settings.fs

        
    return {
        "rmse": np.mean(rmse_list),
        "mse": np.mean(mse_list),
        "snr": np.mean(snr_list),
        "correlation": np.mean(corr_list),
    }



def print_evaluation_results(label, metrics):
    print(f"\n--- Signal: {label} ---")
    print(f"RMSE:        {metrics['rmse']:.3f}")
    print(f"MSE:        {metrics['mse']:.3f}")
    print(f"SNR (dB):    {metrics['snr']:.2f}")
    print(f"Korrelation: {metrics['correlation']:.3f}")