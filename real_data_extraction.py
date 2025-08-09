import pandas as pd
import numpy as np
import os
import sys



from create_signals import extract_data_segments

def get_mean_segment_from_file(spalten_index=3, 
                               fs=1000, 
                               target_length=20,
                               filepath = r'D:\MEchatronik\Bachlorarbeit\Aaron-Material\messdaten_unshielded_mrx\omg_log_20201208_195736.csv'):
    """
    Liest eine CSV-Datei ein, extrahiert aktive Segmente und berechnet
    den Mittelwert-Verlauf über die ersten `target_length` Punkte aller gültigen Segmente.

    Rückgabe:
    - mean_per_point: 1D-Numpy-Array mit `target_length` Werten (repräsentative Sequenz)
    """
    # === Signal laden und vorbereiten ===
    signal = read_signal_from_csv(filepath, spalten_index)
    signal = remove_500Hz(signal)
    t_total = create_time_vector(len(signal), fs)

    # === Aktive Segmente extrahieren ===
    list_signals, list_starts, list_ends = extract_data_segments(signal, t_total)

    # === Nur Segmente mit ausreichender Länge verwenden ===
    aligned_segments = [
        segment[:target_length]
        for segment in list_signals
        if len(segment) >= target_length
    ]

    # Prüfen, ob genug Daten vorhanden sind
    if len(aligned_segments) == 0:
        raise ValueError("Keine Segmente mit ausreichender Länge gefunden.")

    # In NumPy-Array umwandeln: shape = (n_segments, target_length)
    aligned_array = np.array(aligned_segments)

    # Mittelwert berechnen
    mean_per_point = np.mean(aligned_array, axis=0)

    return mean_per_point


def edit_given_data(signal, invert, cut_last_point, settings,remove_empty_measurement,spalte):
    """
    Bearbeitet das Signal segmentweise:
    - Invertiert aktive Sequenzen bei Bedarf.
    - schneidet letzten Punkt ggf. ab

    Parameter:
    - signal: np.ndarray – Eingangssignal mit abwechselnd aktiven + Pausendaten
    - invert: bool – ob aktive Sequenzen invertiert werden sollen
    - cut_last_point: bool – ob der letzte Punkt jeder aktiven Sequenz gelöscht werden soll
    - settings: SignalSettings – enthält Struktur der Sequenzen

    Rückgabe:
    - bearbeitetes Signal (np.ndarray)
    """
    signal = signal.copy()

    num_blocks=len(signal)//settings.samples_block

    if remove_empty_measurement:
            
        filepath = r'D:\MEchatronik\Bachlorarbeit\Aaron-Material\messdaten_unshielded_mrx\omg_log_20201208_195736.csv'
        empty_seq=get_mean_segment_from_file(spalte,settings.fs,settings.samples_active,filepath)
    else :   
        empty_seq=None

    for i in range(num_blocks):
        start = i * settings.samples_block
        end = start + settings.samples_active

        if end > len(signal):
            break  # Sicherheitsabbruch bei unvollständigem Block

        # Nur aktiven Teil bearbeiten
        segment = signal[start:end]

        

        if cut_last_point and len(segment) >= 2:
            segment[-1] = np.nan

        if remove_empty_measurement:
            
            segment[:-1] = segment[:-1]  -empty_seq

        if invert:
            # Horizontal spiegeln (invertieren entlang der y-Achse)
            segment = -segment

        signal[start:end] = segment

    return signal
    

def compute_mean(signal, settings,option):
    """
    Berechnet den mittleren Peak-to-Peak-Wert über alle aktiven Sequenzen eines Signals.

    Parameter:
    - signal: np.ndarray – das Signal, aus dem die Werte extrahiert werden sollen
    - settings: Objekt mit .samples_active, .samples_block, .num_blocks

    Rückgabe:
    - gemittelter Peak-to-Peak-Wert (float)
    """
    
    values = []
    num_blocks = len(signal) // settings.samples_block
    for i in range(num_blocks):
        start = i * settings.samples_block
        end = start + settings.samples_active

        if end > len(signal):
            break

        block = signal[start:end]
        nonzero_block = block[block != 0]

        if len(nonzero_block) > 0:
            if option == "pp":
                ptp = np.ptp(nonzero_block)
                values.append(ptp)
            if option =="off":
                active_block = signal[start:end]
                nonzero_values = active_block[active_block != 0.0]
                mid = len(nonzero_values) // 2
                rear_half = nonzero_values[mid:]  # zweite Hälfte (Einfluzss Relaxation geringer)
                mean_off = np.mean(rear_half)
                values.append(mean_off)



    if values:
        return np.mean(values)
    else:
        return 0.0
def read_single_relaxation(filepath):
    """
    Liest eine einzelne Relaxationskurve aus einer CSV-Datei mit den Spalten:
    Time_ms, Signal_nT

    Parameter:
    - filepath: Pfad zur CSV-Datei

    Rückgabe:
    - signal: np.ndarray – 1D-Array mit Signalwerten
    """
    df = pd.read_csv(filepath)
    
    if df.shape[1] < 2:
        raise ValueError("Die CSV-Datei muss mindestens zwei Spalten enthalten (Zeit, Signal).")
    
    try:
        signal = df.iloc[:, 1].astype(float).to_numpy()
    except Exception as e:
        raise ValueError(f"Fehler beim Einlesen der Signalspalte: {e}")
    
    return signal

def align_signal_to_given_relaxation(given_relaxation,total_disturbed_relaxation,settings):

    
    ref_relaxation=read_single_relaxation(given_relaxation)

    
    edit_aligned_signal=np.full(len(total_disturbed_relaxation),np.nan)
    num_blocks = len(total_disturbed_relaxation) // settings.samples_block
    mean_off = compute_mean(total_disturbed_relaxation,settings,"off")
    for i in range(num_blocks):
        start = i * settings.samples_block
        end = start + settings.samples_active
        
        edit_aligned_signal[start:end] = ref_relaxation + mean_off
        
    return edit_aligned_signal  

def align_signals_to_reference(
    signal_to_align, 
    time_to_align,
    reference_start_time, 
    reference_length, 
    settings,
    reference_factor=1,
    total_disturbed_relaxation=None, 
    referenz_reduktion=32.75, 
    scale_addaption=False
):
    

    """
    
    
    Schneidet ein Signal und den zugehörigen Zeitvektor so zu, dass sie
    zum gegebenen Startzeitpunkt und zur gewünschten Länge passen.
    
    Parameters:
    - signal_to_align: np.ndarray → z. B. relaxation_totalSignal
    - time_to_align: np.ndarray → z. B. ref_t_total
    - reference_start_time: float → Startzeit des Referenzsignals
    - reference_length: int → Länge des Referenzsignals (Anzahl Datenpunkte)
    - fs: int → Abtastrate in Hz
    
    Returns:
    - aligned_signal: np.ndarray
    - aligned_time: np.ndarray
    """
    # Finde den Index des Startzeitpunkts im Zeitvektor
    idx_start = np.argmin(np.abs(time_to_align - reference_start_time))
    idx_end = idx_start + reference_length

    # Sicherheit: keine Indexüberschreitung
    if idx_end > len(signal_to_align):
        idx_end = len(signal_to_align)
        idx_start = max(0, idx_end - reference_length)
    
    if scale_addaption:
        mean_peak_peak = compute_mean(total_disturbed_relaxation,settings,"pp")
        edit_aligned_signal = (signal_to_align[idx_start:idx_end] - referenz_reduktion)  #/reference_factor + ref_off
        ref_mean_peak_peak = compute_mean(edit_aligned_signal,settings,"pp")
        skalierungsfaktor = (mean_peak_peak/2) / ref_mean_peak_peak
        edit_aligned_signal = edit_aligned_signal * skalierungsfaktor

        mean_off = compute_mean(total_disturbed_relaxation,settings,"off")

    else:
        edit_aligned_signal = (signal_to_align[idx_start:idx_end] - referenz_reduktion)/reference_factor  #/reference_factor + ref_off
        mean_off = compute_mean(total_disturbed_relaxation,settings,"off")
    aligned_time = time_to_align[idx_start:idx_end]
    
    num_blocks = len(signal_to_align) // settings.samples_block

    for i in range(num_blocks):
        start = i * settings.samples_block
        end = start + settings.samples_active
        
        edit_aligned_signal[start:end] = edit_aligned_signal[start:end] + mean_off
        
    

    

    return edit_aligned_signal, aligned_time




def remove_500Hz(signal):
    
    if len(signal) > 1:
        if signal[1] > signal[0]:
            diff = (signal[1] - signal[0])/2
            signal[1::2] -= diff

            signal[::2] += diff
            
        else:
            diff = (signal[0] - signal[1])/2
            signal[::2] -= diff

            signal[1::2] += diff
            
    return signal

def read_signal_from_csv(filepath, column_index):
    daten = pd.read_csv(filepath, sep=';', comment='#', header=None, skiprows=1)
    signal = daten.iloc[:, column_index].to_numpy()
    return signal



def create_time_vector(signal_length, fs):
    T_total = signal_length / fs
    return np.linspace(0, T_total, signal_length, endpoint=False)


def find_signal_start_time(reconstructed_signal, time_array):
    if len(reconstructed_signal) != len(time_array):
        raise ValueError("Signal und Zeitachse müssen gleich lang sein.")
    for idx, val in enumerate(reconstructed_signal):
        if not np.isnan(val):
            return time_array[idx]
    return None


def extract_signal_from_next_sequence_after_time(
    list_signals, list_starts, list_ends,
    settings,
    t_start_min: float,
    duration_desired: float,
    full_time_array,
    total_length,
    invert=False,
    cut_last_point=False,
    remove_empty_measurement=False,
    spalte=3
    

):
    

    t_end_target = None
    reconstructed = np.full(total_length, np.nan, dtype=float)

    for i, (start, end) in enumerate(zip(list_starts, list_ends)):
        if start >= t_start_min:
            start_idx = i
            t_start_actual = start

            # Anzahl gewünschter Punkte
            n_desired = int(duration_desired * settings.fs)

            # Auf nächstkleinstes Vielfaches von 30 kürzen
            n_desired_adjusted = (n_desired // 30) * 30 -1

            # Neue Zielzeit basierend auf gekürzter Punktzahl
            
            desired_t_end_target = t_start_actual + (n_desired_adjusted / settings.fs)
            if desired_t_end_target < list_ends[-1]:
                t_end_target= desired_t_end_target
            else:
                raise ValueError(f"Gewünschte Sequenzlänge mit einem Ende von {desired_t_end_target} zu lang - letzte Datenwert {list_ends[-1]}")

            break
    else:
        raise ValueError(f"Keine Sequenz gefunden, die nach {t_start_min}s startet.")
    

    if remove_empty_measurement:
        filepath = r'D:\MEchatronik\Bachlorarbeit\Aaron-Material\messdaten_unshielded_mrx\omg_log_20201208_195736.csv'
        empty_seq=get_mean_segment_from_file(spalte,settings.fs,settings.samples_active,filepath)
    else :   
        empty_seq=None
    for seq, start, end in zip(
        list_signals[start_idx:], list_starts[start_idx:], list_ends[start_idx:]
    ):
        if start > t_end_target:
            break

        

        overlap_start = max(start, t_start_actual)
        overlap_end = min(end, t_end_target)
        mask = (full_time_array >= overlap_start) & (full_time_array <= overlap_end)
        indices = np.where(mask)[0]

        start_in_seq = int((overlap_start - start) * settings.fs)
        end_in_seq = start_in_seq + len(indices)
        
        
        

        if end_in_seq <= len(seq):
            values_to_insert = seq[start_in_seq:end_in_seq].copy()

            # Optional: letzten Punkt durch vorletzten ersetzen
            if cut_last_point and len(values_to_insert) > 1:
                values_to_insert[-1] = np.nan
                

            

            if remove_empty_measurement:
                values_to_insert[:-1] = values_to_insert[:-1]  -empty_seq
                
            if invert:
                values_to_insert = -values_to_insert
            reconstructed[indices] = values_to_insert
    
    
    return reconstructed, t_start_actual, t_end_target

def trim_signal_to_valid_range(signal, time_array, t_start, t_end):
    
    if len(signal) != len(time_array):
        raise ValueError("Signal und Zeitachse müssen gleich lang sein.")

    mask = (time_array >= t_start) & (time_array <= t_end)
    trimmed_signal = signal[mask]
    trimmed_time = time_array[mask]

    
    return trimmed_signal, trimmed_time

def check_signal_block_pattern_detailed(signal, settings,min_blocks=1):#, active_len=21, pause_len=10, min_blocks=1):
    """

    Rückgabe:
        result (dict): Enthält:
            - 'pattern_ok' (bool): True, wenn alle geprüften Blöcke korrekt.
            - 'total_blocks' (int): Anzahl geprüfter Blöcke.
            - 'failed_blocks' (int): Anzahl fehlerhafter Blöcke.
            - 'total_active_errors' (int): Summe ungültiger Werte in aktiven Bereichen.
            - 'total_pause_errors' (int): Summe ungültiger Werte in Pausenbereichen.
    """
    #block_size = active_len + pause_len
    total_blocks = len(signal) // settings.samples_block

    result = {
        'pattern_ok': True,
        'total_blocks': min(total_blocks, max(min_blocks, total_blocks)),
        'failed_blocks': 0,
        'total_active_errors': 0,
        'total_pause_errors': 0,
        'length_signal': len(signal)
    }

    for i in range(result['total_blocks']):
        
        start = i * settings.samples_block
        
        # Extrahieren des aktiven und des Pausen-Blocks
        active_block = signal[start : start + settings.samples_active]
        pause_block = signal[start + settings.samples_active : start + settings.samples_block]

     
        active_errors = np.count_nonzero(np.isnan(active_block))
        
    
        pause_errors = np.count_nonzero(~np.isnan(pause_block))

        # Prüfe auf Fehler
        if active_errors > 0 or pause_errors > 0:
            result['failed_blocks'] += 1
            result['total_active_errors'] += active_errors
            result['total_pause_errors'] += pause_errors
            result['pattern_ok'] = False

    return result

def process_signal_from_file(
    filepath,
    spalten_index,
    fs,
    t_start_wunsch,
    dauer_wunsch,
    settings,
    invert=False,
    cut_last_point=False,
    apply_remove_500Hz=False,
    remove_empty_measurement=False,
    given_signal_data=None,
    given_timedata=None,
    spalte=3,
    find_opt_run=False
):
    if given_signal_data is None:
        signal = read_signal_from_csv(filepath, spalten_index)
    else:
        signal=given_signal_data

    if given_timedata:
        t_start, t_end, trimmed_time=given_timedata
    if apply_remove_500Hz:
        signal=remove_500Hz(signal)
    
    

    if given_signal_data is None:
        t_total = create_time_vector(len(signal), fs)
        list_signals, list_starts, list_ends = extract_data_segments(signal, t_total)

        reconstructed, t_start, t_end = extract_signal_from_next_sequence_after_time(
            list_signals,
            list_starts,
            list_ends,
            settings,
            t_start_min=t_start_wunsch,
            duration_desired=dauer_wunsch,
            full_time_array=t_total,
            total_length=len(t_total),
            invert=invert,
            cut_last_point=cut_last_point,
            remove_empty_measurement=remove_empty_measurement,
            spalte=spalte
            
        )
        trimmed_signal, trimmed_time = trim_signal_to_valid_range(reconstructed, t_total, t_start, t_end)
    else:
        trimmed_signal=edit_given_data(signal=given_signal_data,invert=invert,cut_last_point=cut_last_point,settings=settings,remove_empty_measurement=remove_empty_measurement,spalte=spalte)
    


    
    
    if not find_opt_run:
        # PRüft ob die STruktur des Signals passend ist
        # deaktiviert für Optimierungsskripte (Terminal-Spam)
        info = check_signal_block_pattern_detailed(trimmed_signal,settings)

        print("Musterprüfung:")
        print(f"- OK?                    {info['pattern_ok']}")
        print(f"- Länge Signal           {info['length_signal']}")
        print(f"- Geprüfte Blöcke:       {info['total_blocks']}")
        print(f"- Fehlende Blöcke:       {info['failed_blocks']}")
        print(f"- Fehlerhafte Aktive:    {info['total_active_errors']}")
        print(f"- Fehlerhafte Pausen:    {info['total_pause_errors']}")


    return trimmed_signal, t_start, t_end,trimmed_time
