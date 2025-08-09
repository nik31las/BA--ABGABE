import numpy as np
from real_data_extraction import read_signal_from_csv, remove_500Hz, create_time_vector
from create_signals import extract_data_segments





def cut_signal_by_time(t, signal, start_time, end_time):
    
    T_total=end_time-start_time
    if len(t) != len(signal):
        raise ValueError("Zeit- und Signalvektor müssen gleich lang sein.")

    if start_time >= end_time:
        raise ValueError("Startzeit muss kleiner als Endzeit sein.")

    mask = (t >= start_time) & (t <= end_time)
    t_cut = t[mask]
    signal_cut = signal[mask]

    if len(t_cut) == 0:
        raise ValueError("Keine Daten im angegebenen Zeitbereich.")

    return t_cut, signal_cut,T_total


def average_adjacent_sequences(t, signal, fs, active_ms, pause_ms):
    samples_active = int((active_ms / 1000) * fs)
    samples_pause = int((pause_ms / 1000) * fs)
    samples_block = samples_active + samples_pause

    total_samples = len(signal)
    result = np.copy(signal)

    for start in range(0, total_samples - 2 * samples_block + 1, 2 * samples_block):
        seg1 = signal[start:start + samples_active]
        seg2 = signal[start + samples_block:start + samples_block + samples_active]
        if len(seg1) == len(seg2):
            avg = (seg1 + seg2) / 2
            result[start:start + samples_active] = avg
            result[start + samples_block:start + samples_block + samples_active] = avg

    return t, result



def apply_data_editing_to_signal(t, signal, edit_config, settings, index=None,filter=False):
    
    start_cut = edit_config.get("start_cut")
    end_cut = edit_config.get("end_cut")
    avg_sequenzen = edit_config.get("avg_sequenzen")
    T_total=settings.T_total
    if start_cut and end_cut:
        if start_cut < settings.t_dataPoints[0] or end_cut > settings.t_dataPoints[-1]:
            print(f"⚠️ Achtung: Der Bereich {start_cut}s – {end_cut}s liegt außerhalb der Daten ({settings.t_dataPoints[0]}s – {settings.t_dataPoints[-1]}s)\n Keine Daten zugeschnitten")
        else:
            t, signal, T_total = cut_signal_by_time(t, signal, start_cut, end_cut)
            if index is not None:
                print(f"Signal {index+1} wurde zugeschnitten: {start_cut}s – {end_cut}s")
    if avg_sequenzen and settings is not None and filter:
        t, signal = average_adjacent_sequences(t, signal, settings.fs, settings.active_ms, settings.pause_ms)
        if index is not None:
            print(f"➕ Signal {index+1}: benachbarte Sequenzen wurden gemittelt.")

    
    return t, signal,(start_cut,end_cut,T_total)