import numpy as np
import pandas as pd


def get_active_mask_between_nans(signal):
    """
    Gibt ein Boolean-Array zurück, das True ist für alle nicht-nan-Werte
    zwischen dem ersten und letzten NaN im Signal.
    """
    signal = np.array(signal, dtype='float64')
    is_nan = np.isnan(signal)
    
    # Alles initial auf False setzen
    is_active = np.full_like(signal, False, dtype=bool)

    if not np.any(is_nan):
        # Es gibt keine NaNs → alles bleibt False
        return is_active

    # Position des ersten und letzten NaNs
    first_nan_idx = np.argmax(is_nan)
    last_nan_idx = len(signal) - 1 - np.argmax(is_nan[::-1])

    # Bereich zwischen den beiden NaNs prüfen
    for i in range(first_nan_idx + 1, last_nan_idx):
        if not is_nan[i]:
            is_active[i] = True

    return is_active



def extract_data_segments(signal, time_array, pause_threshold=1e-3, min_active_length=10):
    """
    Findet aktive Segmente im Signal basierend auf Übergängen von NaN → Wert.
    
    Rückgabe:
    - segments: Liste von Arrays mit Signalabschnitten
    - start_segment: Liste mit Startzeitpunkten (Sekunden)
    - end_segment: Liste mit Endzeitpunkten (Sekunden)
    """
    signal = np.array(signal, dtype='float64')
    time_array = np.array(time_array, dtype='float64')

    if len(signal) != len(time_array):
        raise ValueError("Signal und Zeitachse müssen gleich lang sein.")

    is_active=get_active_mask_between_nans(signal)

    segments = []
    start_segment = []
    end_segment = []

    start = None
    for i in range(len(is_active)):
        if is_active[i] and start is None:
            start = i
        elif not is_active[i] and start is not None:
            if i - start >= min_active_length:
                segments.append(signal[start:i])
                start_segment.append(time_array[start])
                end_segment.append(time_array[i - 1])
            start = None

    if start is not None and len(signal) - start >= min_active_length:
        segments.append(signal[start:])
        start_segment.append(time_array[start])
        end_segment.append(time_array[-1])

    return segments, start_segment, end_segment

def extract_sequence(signal,time_array, t_start, t_end):
    if len(signal) != len(time_array):
        print(f"Länge Signal:{len(signal)}, Länge Time:{len(time_array)}")
        raise ValueError(f"Signal und Zeitachse müssen gleich lang sein.")

    mask = (time_array >= t_start) & (time_array <= t_end)
    segment = signal[mask]
    t_segment = time_array[mask] - t_start  # Zeitachse auf 0 setzen

    return segment, t_segment


# Signal-Generatoren mit Start- und Endzeiten
def relaxation_signal(t, B0, Tau):
    return B0 * np.exp(-t / Tau)

def relaxation_signal_offset(t, B0, Tau,c):
    return B0 * np.exp(-t / Tau) + c

def white_noise(t, amplitude, start, end):
    signal = np.zeros_like(t)
    mask = (t >= start) & (t <= end)
    signal[mask] = np.random.normal(0, amplitude/3, size=np.sum(mask))
    return signal
#Erzeugt zufällige Werte aus einer Normalverteilung mit Mittelwert 0 und Standardabweichung 𝐴 (Amplitude)


def white_noise_option1(t, A0,f_max,delta_f, start, end):
    signal = np.zeros_like(t)
    mask = (t >= start) & (t <= end)
    N=int(f_max/delta_f)
    
    for k in range (1, N+1):
        phi_k = np.random.uniform(0, 2 * np.pi)
        alpha_k = np.sin(phi_k)
        beta_k = np.cos(phi_k)
        component = (
            A0/3*(np.sqrt(2 / N) * alpha_k * np.cos(2 * np.pi * k * delta_f * t) +
            np.sqrt(2 / N) * beta_k * np.sin(2 * np.pi * k * delta_f * t))
        )
        signal += component

     
    return signal
    
def sinus_störung(t, freq, amplitude, start, end):
    """
    Erzeugt ein Sinus-Störsignal, das ab `start` beginnt (mit weichem Anfang) und bei `end` endet.
    """
    signal = np.zeros_like(t)
    mask = (t >= start) & (t <= end)
    signal[mask] = amplitude * np.sin(2 * np.pi * freq * (t[mask] - start))
    return signal

def linear_drift(t, start, end, amplitude):
    drift = np.zeros_like(t)
    mask = (t >= start) & (t <= end)
    if np.any(mask):
        drift[mask] = np.linspace(0, amplitude, np.sum(mask))
    return drift

def impulse_noise(t, period, amplitude, start, end, impulse_times=None, impulse_width=None):
    #Erzeugt ein Impulsrauschen mit variabler Verschiebung und optionaler Breite.
    if impulse_times is None:  
        impulse_times = [0]  # Standardwert setzen


    impulses = np.zeros_like(t)
    num_periods = int((end - start) / period)  # Anzahl der möglichen Perioden

    for i in range(1, num_periods + 1):
        # Wähle die Impulsverschiebung für diese Periode (zyklisch durch die Liste)
        impulse_time = impulse_times[i % len(impulse_times)]
        time_point = i * period - impulse_time

        if start <= time_point <= end:
            idx = np.argmin(np.abs(t - time_point))
            if impulse_width is None:
                impulses[idx] = amplitude  # Einzelner Punktimpuls
            else:
                # Erzeuge einen breiten Impuls
                width_mask = (t >= (t[idx] - impulse_width / 2)) & (t <= (t[idx] + impulse_width / 2))
                impulses[width_mask] = amplitude

    return impulses


def init_disturbsignals(t,signal_params):
    
    disturbed_Signals=[]

    
    labels = []

    for params in signal_params:
        s_type = params["type"]

        if s_type == 1:
            disturbed_Signals.append(white_noise(t, params["amplitude"], params["start"], params["end"]))
            labels.append("Weißes Rauschen")

        elif s_type == 2:
            disturbed_Signals.append(linear_drift(t, params["start"], params["end"],params["amplitude"]))
            labels.append("Linearer Drift")

        elif s_type == 3:
            disturbed_Signals.append(impulse_noise(
                t, params["period"], params["amplitude"], params["start"], params["end"],
                impulse_times=params["impulse_times"], impulse_width=params["impulse_width"]
            ))
            labels.append("Impulsrauschen")

        elif s_type == 4:
            for freq, amp, start, end in params["sinusoids"]:
                disturbed_Signals.append(sinus_störung(t, freq, amp, start, end))
                labels.append(f"Sinusrauschen {freq} Hz")

        elif s_type == 5:
            disturbed_Signals.append( white_noise_option1(t,params["amplitude"],params["f_max"],params["delta_f"], params["start"], params["end"]))
            labels.append("Weißes Rauschen")

    return disturbed_Signals, labels







