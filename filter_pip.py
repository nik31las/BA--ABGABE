import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Callable,Tuple
from scipy.optimize import curve_fit
#from create_signals import extract_active_sequences,reconstruct_with_pauses
from split_unite_signal import  extract_active_sequences, reconstruct_signal_with_pauses,reshape_flat_to_blocks,select_every_nth_sequence

from scipy.signal import butter, filtfilt, iirnotch, medfilt, detrend,savgol_filter
from create_signals import relaxation_signal_offset,relaxation_signal

def savitzky_golay(settings, signal, savgol_window, savgol_poly,savgol_mode):
    cval = None
    if "constant=" in savgol_mode:
        parts = savgol_mode.split("=")
        savgol_mode = "constant"
        try:
            cval = float(parts[1].strip())
        except (IndexError, ValueError):
            print("⚠️ Ungültiger constant-Wert. cval bleibt None.")

    

    fs = 1 / (settings.t_dataPoints[1] - settings.t_dataPoints[0])  # 1/(deltaT) = freqquenz
    total_samples = len(signal)

    

    # Fenster- und Polyorder anpassen
    savgol_window = max(3, int(round(savgol_window)))
    if savgol_window % 2 == 0:
        savgol_window += 1

    savgol_poly = int(round(savgol_poly))
    if savgol_poly >= savgol_window:
        
        savgol_poly = savgol_window - 1
    savgol_poly = max(1, savgol_poly)

    # Initialisiere Ergebnis
    filtered_total = np.copy(signal)

    # Sequenzen blockweise filtern
    for start in range(0, total_samples, settings.samples_block):
        end = start + settings.samples_active
        if end > total_samples:
            break
        
        if "addzero" in savgol_mode:
            pause_end=end +settings.samples_pause
            pause_segment = np.zeros(settings.samples_pause)
            segment = np.concatenate((signal[start:end], pause_segment))
            savgol_mode="interp"
        else:
            segment = signal[start:end]

        if len(segment) >= savgol_window:
            try:
                if cval==None:
                    filtered_segment = savgol_filter(segment, window_length=savgol_window, polyorder=savgol_poly,mode=savgol_mode)
                else:
                    filtered_segment = savgol_filter(segment, window_length=savgol_window, polyorder=savgol_poly,mode=savgol_mode,cval=cval)

            except Exception as e:
                #print(f"⚠️ Savitzky-Golay-Fehler in Block {start}-{end}: {e}")
                filtered_segment = segment
        else:
            filtered_segment = segment  # zu kurz → nicht filtern

        filtered_total[start:end] = filtered_segment[:settings.samples_active]

    return filtered_total


def notch_filterung(signal,notch_freqs, notch_q,notch_odd_sequence,settings):
    
    if isinstance(notch_q, list) and len(notch_q) == 1 and isinstance(notch_q[0], list):
        notch_q = notch_q[0]

    if isinstance(notch_freqs, list) and len(notch_freqs) == 1 and isinstance(notch_freqs[0], list):
        notch_freqs = notch_freqs[0]
        
    
    for f0, q in zip(notch_freqs, notch_q):
        
        ##############################################
        # Auswählen, ob gerade oder ungerade Sequenzen
        start_index= 1    #(odd = 0, even =1)
        ##############################################
        if notch_odd_sequence: 
            active_only_0= select_every_nth_sequence(signal,settings,nth=2,start_index=start_index)
        

            combined_sequences = active_only_0 + active_only_0
        else:
            combined_sequences = extract_active_sequences(signal,settings)

        signal_active_only = np.concatenate(combined_sequences)
        
        if q == 0:
            filtered=signal
            continue
        
        b, a = iirnotch(f0, q, settings.fs)
        filtered_active_only = filtfilt(b, a, signal_active_only)
        filtered_blocks = reshape_flat_to_blocks(filtered_active_only, settings)
        filtered=reconstruct_signal_with_pauses(filtered_blocks,settings)

        #Wichtig, für Serielle Filterung!!
        signal = filtered
        
        
    return filtered
    
def apply_curvefit_per_sequence(
    settings,
    signal: np.ndarray,
    model: Callable,
    p0: Optional[List[float]] = None,
    bounds: Optional[Tuple[List[float], List[float]]] = None,
    method: Optional[str] = None,
    others: Optional[List[str]]=None
) -> np.ndarray:
    
    
    if isinstance(p0, list) and len(p0) == 1 and isinstance(p0[0], list):
        p0 = p0[0]

    off=False
    if others:
        B0=60
        Tau=0.001
        c=0
        for string in others:

            if 'useoff' in string:
                off = True
                if bounds is None or len(bounds[0]) != 3 or len(bounds[1]) != 3:
                    bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
                    print("Grenzwerte anpassen auf Modelparameter")
                
                if p0 is None or len(p0) !=3:
                    p0=[B0, Tau,c]
                    print("AStartwerte anpassen auf Modelparameter")
                model=relaxation_signal_offset
    
    
    if not off:
        if bounds is None or len(bounds[0]) != 2 or len(bounds[1]) != 2:
            bounds = ([0, 0], [np.inf, np.inf])
            print("Grenzwerte anpassen auf Modelparameter")
        
        if p0 is None or len(p0) !=2:
            p0=[B0, Tau]
            print("Startwerte anpassen auf Modelparameter") 
        model=relaxation_signal
    
    """
    Wendet Curve-Fitting blockweise auf die aktiven Signalteile an.
    """
    
    fitted_signal = np.copy(signal)
    total_samples = len(signal)

    for block_idx, start in enumerate(range(0, total_samples, settings.samples_block)):
        end = start + settings.samples_active
        if end > total_samples:
            print(f"⚠️ Block {block_idx}: Bereich [{start}:{end}] überschreitet Signalende – übersprungen.")
            break

        x_data = np.arange(0, settings.samples_active) / settings.fs
        y_data = signal[start:end]
        ##########################################################
        #### Ausgabe der Fitting PArameter ggf. zur Kontrolle ####
        #### oder zum Bestimmen von Offsets ######################
        #print(f"\n🔹 Block {block_idx}: x_data=({x_data[0]:.4f}…{x_data[-1]:.4f}), y_mean={np.mean(y_data):.2f}")
        ##########################################################
        
        
        if others:
            for addon in others:
                if isinstance(addon, str) and "guess_p0" in addon.strip().lower():
                    # Modell deaktivieren oder auf eine alternative Version setzen
                    # Hier: Schätze Startwerte grob aus y_data
                    B0_guess = np.max(y_data) - np.min(y_data)
                    if off:
                        c_guess = np.min(y_data)
                        p0 = [B0_guess, Tau, c_guess]  # oder z. B. model.disable()
                    else:
                        p0 = [B0_guess, Tau]
                    #print(f"Geschärtzte Werte: {p0}")
                
        
        
        # Curve-Fitting Parameter vorbereiten
        fit_kwargs = {}
        fit_kwargs["max_nfev"] = 50000
        # Methode prüfen
        if method == "lm":
            # Bei 'lm' keine bounds erlaubt
            if p0 is not None:
                fit_kwargs["p0"] = p0
            fit_kwargs["method"] = "lm"
        else:
            # Alle anderen Methoden
            if p0 is not None:
                fit_kwargs["p0"] = p0
            if bounds is not None:
                fit_kwargs["bounds"] = bounds
            if method is not None:
                fit_kwargs["method"] = method

        try:
            ##################################################
            #Falls bessere Ausgabe zur Kontrolle benötigt wird
            #print("→ Fitting-Parameter:")
            #for key, val in fit_kwargs.items():
                #print(f"   {key} = {val}")
            #print(f" others: {others}")
            ##################################################
            popt, _ = curve_fit(model, x_data, y_data, **fit_kwargs)
            fitted_segment = model(x_data, *popt)
            #print(f"   → Erfolgreich gefittet: Parameter = {popt}")
        except Exception as e:
            print(f"❌ Fehler bei Block {block_idx}: {e}")
            fitted_segment = y_data

        
        fitted_signal[start:end] = fitted_segment

    
    return fitted_signal




def filter_pipeline(
    settings,
    signal: np.ndarray,
    apply_median: bool = True,
    median_kernel_size: int = 5,
    apply_notch: bool = True,
    notch_freqs: Optional[List[float]] = [50.0, 150.0],
    notch_q: float = 30.0,
    notch_odd_sequence: bool = True,
    detrend_signal: bool = True,
    detrend_method: str = "highpass",  # "highpass" oder "linear"
    detrend_cutoff: float = 0.1,
    apply_bandpass: bool = True,
    bandpass_low: float = 1.0,
    bandpass_high: float = 200.0,
    bandpass_order: int = 4,
    apply_savgol: bool = False,
    savgol_window: int = 11,
    savgol_poly: int = 3,
    savgol_cut_signal_sequenz: bool = True,
    savgol_mode: str ="interp",
    apply_curvefit: bool = True,
    curvefit_model: Optional[Callable] = None,  # z. B. exp_model(t, a, b, c)
    curvefit_p0: Optional[List[float]] = None,
    curvefit_bounds: Tuple[List[float], List[float]] = ([-np.inf], [np.inf]),
    curvefit_method: str = "trf",
    curvefit_others: List[str]=None
    
) -> np.ndarray:
    

    filtered = np.copy(signal)

    
    '''
    --------------------------------------
    Hier wären mögliche zusätzlichen Filter 
    (deaktiviert bzw. noch nicht implementiert)
    if apply_median:
    # Detrend
    if detrend_signal:      
    # Bandpassfilter
    if apply_bandpass:
    --------------------------------------
    '''
        
    
    # Notchfilter

    if apply_notch and notch_freqs:
        filtered=notch_filterung(signal,notch_freqs, notch_q,notch_odd_sequence,settings)
    
    
    # Savitzky-Golay-Filter
    if apply_savgol:
        savgol_cut_signal_sequenz = True
        if savgol_cut_signal_sequenz:
            #einzelne Sequenzen
            filtered = savitzky_golay(settings, filtered,savgol_window, savgol_poly,savgol_mode)
        else:
            #Gesamtes Signal
            filtered=savgol_filter(filtered,window_length=savgol_window, polyorder=savgol_poly)

    if apply_curvefit and curvefit_model:
        filtered = apply_curvefit_per_sequence(
            settings,
            signal=filtered,
            model=curvefit_model,
            p0=curvefit_p0,
            bounds=curvefit_bounds,
            method=curvefit_method,
            others=curvefit_others
        )

    return filtered



def apply_filters(
    settings,
    signal: np.ndarray,
    apply_median=True,
    median_kernel_size=5,
    apply_notch=True,
    notch_freqs=[50.0],
    notch_q=[30.0],
    notch_odd_sequence=True,
    detrend_signal=True,
    detrend_method="highpass",
    detrend_cutoff=0.1,
    apply_bandpass=True,
    bandpass_low=1.0,
    bandpass_high=200.0,
    bandpass_order=4,
    apply_savgol=False,
    savgol_window=11,
    savgol_poly=3,
    savgol_cut_signal_sequenz=True,
    savgol_mode="interp",
    apply_curvefit: bool = True,
    curvefit_model: Optional[Callable] = None,  # z. B. anderer Datentyp, aber umgewansdelt wird hier Funktionen wie - exp_model(t, a, b, c) gespeichert
    curvefit_p0: Optional[List[float]] = None,
    curvefit_bounds: Tuple[List[float], List[float]] = ([-np.inf], [np.inf]),
    curvefit_method: str = "trf",
    curvefit_others: List[str]=None
) -> np.ndarray:
    return filter_pipeline(
        settings,
        signal,
        apply_median=apply_median,
        median_kernel_size=median_kernel_size,
        apply_notch=apply_notch,
        notch_freqs=notch_freqs,
        notch_q=notch_q,
        notch_odd_sequence=notch_odd_sequence,
        detrend_signal=detrend_signal,
        detrend_method=detrend_method,
        detrend_cutoff=detrend_cutoff,
        apply_bandpass=apply_bandpass,
        bandpass_low=bandpass_low,
        bandpass_high=bandpass_high,
        bandpass_order=bandpass_order,
        apply_savgol=apply_savgol,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
        savgol_cut_signal_sequenz=savgol_cut_signal_sequenz,
        savgol_mode=savgol_mode,
        apply_curvefit= apply_curvefit,
        curvefit_model= curvefit_model,  
        curvefit_p0= curvefit_p0,
        curvefit_bounds= curvefit_bounds,
        curvefit_method=curvefit_method,
        curvefit_others= curvefit_others
    )


