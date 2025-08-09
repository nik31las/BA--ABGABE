import numpy as np
from create_signals import relaxation_signal,relaxation_signal_offset

def get_bool_input(prompt, default=False):
                antwort = input(f"{prompt} (j/n) [{'j' if default else 'n'}]: ").strip().lower()
                if antwort in {"j", "ja", "y"}:
                    return True
                elif antwort in {"n", "nein"}:
                    return False
                return default


def get_curvefit_parameters(param_names):
    """
    Fragt Startwerte (p0) und optionale Grenzen (bounds) für curve_fit-Parameter ab.

    Args:
        param_names (list): Liste der Parameternamen, z. B. ["B0", "Tau"]

    Returns:
        Tuple:
            p0 (list of float or None)
            bounds (tuple of (list, list) or None)
    """
    print("\n🔧 Curve-Fit Parameter-Eingabe:")

    # Startwerte abfragen
    use_p0 = input(" Startwerte (p0) angeben? (j/n) [n]: ").strip().lower() == "j"
    p0 = []
    if use_p0:
        for name in param_names:
            value_str = input(f"    Startwert für {name} [leer für ungültig]: ").strip()
            try:
                value = float(value_str)
                p0.append(value)
            except ValueError:
                print(f"    ⚠️ Ungültiger Wert für {name} – Startwerte werden verworfen.")
                p0 = None
                break
    else:
        p0 = None

    # Bounds abfragen
    use_bounds = input(" Grenzen (bounds) angeben? (j/n) [n]: ").strip().lower() == "j"
    bounds = None
    if use_bounds:
        lower = []
        upper = []
        for name in param_names:
            low_str = input(f"    Untergrenze für {name} [leer für -inf]: ").strip()
            up_str = input(f"    Obergrenze für {name} [leer für +inf]: ").strip()
            try:
                low = float(low_str) if low_str else -float("inf")
                up = float(up_str) if up_str else float("inf")
                lower.append(low)
                upper.append(up)
            except ValueError:
                print(f"    ⚠️ Ungültige Grenze für {name} – Grenzen werden ignoriert.")
                lower, upper = None, None
                break

        if lower is not None and upper is not None:
            bounds = (lower, upper)

    return p0, bounds

def get_choice(prompt, default="j"):
        eingabe = input(prompt).strip().lower()
        return eingabe if eingabe in ("j", "n", "p") else default

def ask_filter_settings(find_opt=False):
    #print(f"\nFiltereinstellungen für {signal_name if signal_name else 'alle Signale'}:")

    # find opt -> wenn Optimierung wird PArameter einstellung geskippt (wird später gemacht)
    if find_opt:
        print("Nur Auswahl der Filter - j,p wählen den Filter aus\n" \
        "Parameter definition bzw. Bereichsdefnition wird später ausgeführt")


    
    ###########################################################################
    #################### nicht vollständig implementierte Filter ##############
    ################## lediglich Variabeln bereits vordefiniert ###############
    # Medianfilter
    #median_choice = get_choice("  Medianfilter verwenden? (j/n/p): ", default="n")
    apply_median=False
    median_kernel_size = 5
    median_choice="n"
    # Driftentfernung
    #detrend_choice = get_choice("  Driftentfernung (Detrend) verwenden? (j/n/p): ", default="n")
    detrend_choice="n"
    detrend_signal=False
    detrend_method = "highpass"
    detrend_cutoff = 0.1
    # Bandpass
    #bandpass_choice = get_choice("  Bandpassfilter verwenden? (j/n/p): ", default="n")
    bandpass_choice="n"
    apply_bandpass = bandpass_choice != "n"
    bandpass_low = 1.0
    bandpass_high = 200.0
    bandpass_order = 4
    apply_bandpass=False
    ###########################################################################
    ###########################################################################

    # Notchfilter
    notch_choice = get_choice("  Notchfilter verwenden? (j/n/p): ", default="n")
    notch_freqs = []
    notch_q = []
    notch_odd_sequence=False
    notch_number=1
    if notch_choice == "j":
        notch_freqs = [50.0]
        notch_q= [3]
        apply_notch = True
    elif notch_choice == "p":
        apply_notch = True
        if not find_opt:
            notch_number = int(input("    Anzahl an Notchfiltern") or 1)
            
            for i in range(notch_number):
                notch_freqs.append(float(input("    Frequenz in Hz: [50] ") or 50.0))
                notch_q.append(float(input("    Q-Faktor [3]: ") or 30.0))
                notch_odd_sequence_choice= get_choice("  nur jede 2. Sequenz verwenden? (j/n): ", default="j")
                if notch_odd_sequence_choice=="j":
                    notch_odd_sequence=True
    else:
        apply_notch = False
    

    # Savitzky-Golay-Filter
    savgol_choice = get_choice("  Savitzky-Golay-Filter verwenden? (j/n/p): ", default="n")
    
    savgol_window = 11
    savgol_poly = 3
    savgol_cut_signal_sequenz=True
    savgol_mode="interp"

    if savgol_choice == "j":
        apply_savgol=True
    elif savgol_choice == "p":
        apply_savgol=True

        if not find_opt:
            savgol_window = int(input("    Fensterlänge (ungerade Zahl, z. B. 11) [11]: ") or 11)
            if savgol_window % 2 == 0:
                print(f"    Hinweis: {savgol_window} ist gerade – abgerundet auf {savgol_window - 1}")
                savgol_window -= 1
            savgol_poly = int(input("    Polynomgrad (z. B. 3) [3]: ") or 3)
            savgol_mode = str(input("   Modus für Verarbeitung festlegen [interp] (mirror, constant, nearest, wrap or interp) : ").strip().lower() or "interp")
            #savgol_cut_signal_sequenz_in = input("    Signal auf einzelne Sequenz zuschneiden (j/n) [j]: ").strip().lower()
            savgol_cut_signal_sequenz = True #if savgol_cut_signal_sequenz_in in {"", "j", "ja", "y"} else False
    else:
        apply_savgol = False
    # Curve-Fitting
    curvefit_choice = get_choice("  Curve-Fit verwenden? (j/n/p): ", default="n")
    

    # Standardmodell
    curvefit_model = relaxation_signal  # Muss als Funktion definiert sein: def relaxation_signal(t, B0, Tau): ...
    curvefit_p0 = [60.0, 0.001]           # Default: realistische Startwerte für exp-Fit
    
    
    curvefit_bounds = ([0, 0], [np.inf, np.inf])   # Default: keine Grenzen
    curvefit_method = "trf" #None #"lm"            # Default: stabiler, moderner Algorithmus
    curvefit_others = []               # Noch nicht verwendet – optional später parsen
    param_names = ["B0", "Tau"]

    if curvefit_choice =="j":
        apply_curvefit= True
        
        
    elif curvefit_choice == "p":
        apply_curvefit= True
        if not find_opt:
            # Weitere Parameter – als String gespeichert
            others_in = input("    Voreinstellungen (optional, z. B. useoff für Offset, guess_p0 für Startwerte Schätzen) durch Komma trennen [leer]: \n"
            "Für das Verarbeiten von REALEN DATEN wird die Einstelung useoff stark empfohln").strip()
            if others_in:
                curvefit_others = [s.strip() for s in others_in.split(",") if s.strip()]
                for addon in curvefit_others:
                    if "useoff" in addon.strip().lower():
                        # Modell deaktivieren oder auf eine alternative Version setzen
                        curvefit_model = relaxation_signal_offset  # oder z. B. model.disable()
                        param_names = ["B0", "Tau","c"]
                        print("Modell mit Offset: 'useoff'")
                        curvefit_bounds = ([0, 0, -np.inf], [np.inf , np.inf, np.inf])#([59.0,0.00095],[61,0.00105])   # Default: keine Grenzen
                        curvefit_p0 = [60.0, 0.001,0] 
            # Interaktive Eingabe von p0 und bounds
            
            p0_input, bounds_input = get_curvefit_parameters(param_names)

            if p0_input is not None:
                curvefit_p0 = p0_input
            if bounds_input is not None:
                curvefit_bounds = bounds_input

            # Methode abfragen
            method_in = input("    Fitting-Methode (trf / dogbox / lm) [trf]: ").strip().lower()
            if method_in in {"trf", "dogbox", "lm"}:
                curvefit_method = method_in
    else:
        apply_curvefit = False
            



    return {
        "apply_median": apply_median,
        "median_kernel_size": median_kernel_size,
        "apply_notch": apply_notch,
        "notch_freqs": notch_freqs,
        "notch_q": notch_q,
        "notch_odd_sequence":notch_odd_sequence,
        "detrend_signal": detrend_signal,
        "detrend_method": detrend_method,
        "detrend_cutoff": detrend_cutoff,
        "apply_bandpass": apply_bandpass,
        "bandpass_low": bandpass_low,
        "bandpass_high": bandpass_high,
        "bandpass_order": bandpass_order,
        "apply_savgol": apply_savgol,
        "savgol_window": savgol_window,
        "savgol_poly": savgol_poly,
        "savgol_mode": savgol_mode,
        "savgol_cut_signal_sequenz": savgol_cut_signal_sequenz,
        "apply_curvefit": apply_curvefit,
        "curvefit_model": curvefit_model,  # z. B. exp_model(t, a, b, c)
        "curvefit_p0": curvefit_p0,
        "curvefit_bounds": curvefit_bounds,
        "curvefit_method":curvefit_method,
        "curvefit_others": curvefit_others
    }

def ask_data_edit_settings(T_total,real_data=False,given_Data_choice=False):
    if not given_Data_choice:
        edit_Data_Choice = input("Soll der Datensatz bearbeitet werden (j/n)? [n]: ").strip().lower() or "n"
        edit_Data = False
        cut_Data=False
        colum_filtering=False
        remove_empty_measurement=False
        if edit_Data_Choice == "j":

            if not real_data:
                print("Nur j oder n Eingeben, Variieren der Datenbearebitungsmethoden nur mit realen Daten möglich")
            
            edit_Data=True
            cut_data_Choice = input("Soll Datensatz zugeschnitten werden (j/n)? [n]: ").strip().lower() or "n"
            if cut_data_Choice == "j":
                cut_Data=True
                start_cut = float(input("Startpunkt Zuschneiden [0.0]: ") or 0.0)
                end_cut = float(input(f"Endpunkt Zuschneiden [{T_total}]: ") or T_total)
            else:
                start_cut = None
                end_cut = None
            
            if real_data:
                pre_avg_sequenzen = get_bool_input("Zwei benachbarte Sequenzen VOR Filterung mitteln? (j/n) [n]: ",default= False)
            
            
            avg_sequenzen = get_bool_input("Zwei benachbarte Sequenzen NACH Filterung mitteln? (j/n) [n]: ",default=False)

            if real_data:
                

                colum_filtering   = get_bool_input("Spalten davor filtern?", default=False)
                cut_last_datapoint = get_bool_input("Letzten Datenpunkt abschneiden?", default=True)
                invert             = get_bool_input("Gegebene Daten invertieren?", default=False)
                apply_remove_500Hz = get_bool_input("500 Hz entfernen?", default=True)
                remove_empty_measurement = get_bool_input("Leermessung der MRX-Aparatur abziehen?", default=True)
                return {
                    "edit_Data": edit_Data,
                    "cut_Data": cut_Data,
                    "start_cut": start_cut,
                    "end_cut": end_cut,
                    "pre_avg_sequenzen": pre_avg_sequenzen,
                    "avg_sequenzen": avg_sequenzen,
                    "colum_filtering": colum_filtering,
                    "cut_last_datapoint": cut_last_datapoint,
                    "invert": invert,
                    "apply_remove_500Hz": apply_remove_500Hz,
                    "remove_empty_measurement":remove_empty_measurement
                }
            else:

                return {
                    "edit_Data": edit_Data,
                    "cut_Data": cut_Data,
                    "start_cut": start_cut,
                    "end_cut": end_cut,
                    "pre_avg_sequenzen": False,
                    "avg_sequenzen": avg_sequenzen,
                    "colum_filtering": False,
                    "cut_last_datapoint": True,
                    "invert": False,
                    "apply_remove_500Hz": True,
                    "remove_empty_measurement":True
                }

    else:
        edit_Data = True
    # Fallback
    return {
        "edit_Data": edit_Data,
        "cut_Data": False,
        "start_cut": None,
        "end_cut": None,
        "pre_avg_sequenzen": False,
        "avg_sequenzen": False,
        "colum_filtering": False,
        "cut_last_datapoint": True,
        "invert": False,
        "apply_remove_500Hz": True,
        "remove_empty_measurement":False
    }

            


def get_interference_signal(T_total):
    print("Welche Störsignale sollen verwendet werden?")
    print("[Enter] → Alle verwenden")
    print("Oder gib eine Auswahl ein (z. B. 1 4 für White Noise & Sinus):")
    print("1: Weißes Rauschen (veraltet)\n2: Linearer Drift (deaktiviert)\n3: Impulsrauschen (deaktiviert)\n4: Sinus-Störungen\n5: Weißes Rauschen Ansatz äqui-Freq")

    selection = input("Auswahl: ").replace(",", " ").strip()
    if selection == "":
        selected_signals = [4, 5]
    else:
        try:
            selected_signals = sorted(set(int(i) for i in selection.split() if i in {"1", "2", "3", "4","5"}))
        except ValueError:
            print("Ungültige Eingabe – alle Signale werden verwendet.")
            selected_signals = [1, 2, 3, 4,5]

    signal_params = []

    default_start = 0.0
    default_end = T_total
    for s in selected_signals:
        print(f"\n--- Parameter für Signal {s} ---")
        
        if s == 1:  # Weißes Rauschen
            # Standardwerte definieren
            default_amp = 1.0 #in nT
            

            choice = input("  Standardparameter für Weißes Rauschen? ([Enter] für Standard, 'p' für eigene Werte): ").strip().lower()
            if choice == "p":
                amp = float(input(f"    Amplitude [{default_amp}]: ") or default_amp)
                start = float(input(f"    Startzeit [{default_start}]: ") or default_start)
                end = float(input(f"    Endzeit [{default_end}]: ") or default_end)
            else:
                amp = default_amp
                start = default_start
                end = default_end

            signal_params.append({
                "type": 1,
                "amplitude": amp,
                "start": start,
                "end": end
            })

        

        elif s == 4:  # Sinus
            # Standardwerte definieren
            default_n = 1
            default_freq = 50.0
            default_amp = 50.0

            choice = input("  Standard-Sinusstörungen verwenden? ([Enter] für Standard, 'p' für eigene Werte): ").strip().lower()
            sinusoids = []

            if choice == "p":
                n = int(input(f"  Anzahl Sinusse [{default_n}]: ") or default_n)
                for i in range(n):
                    print(f"    Sinus {i+1}:")
                    freq = float(input(f"      Frequenz [{default_freq}]: ") or default_freq)
                    amp = float(input(f"      Amplitude [{default_amp}]: ") or default_amp)
                    start = float(input(f"      Startzeit [{default_start}]: ") or default_start)
                    end = float(input(f"      Endzeit [{default_end}]: ") or default_end)
                    sinusoids.append((freq, amp, start, end))
            else:
                sinusoids = [(default_freq, default_amp, default_start, default_end)]

            signal_params.append({
                "type": 4,
                "sinusoids": sinusoids
            })
        elif s == 5:  # Weißes Rauschen
            # Standardwerte definieren
            default_amp = 1.0 #in nT
            default_f_max=500
            default_delta_f=1

            choice = input("  Standardparameter für Weißes Rauschen? ([Enter] für Standard, 'p' für eigene Werte): ").strip().lower()
            if choice == "p":
                amp = float(input(f"    Amplitude [{default_amp}]: ") or default_amp)
                f_max = float(input(f"    Maximale Frequenz [{default_f_max}]: ") or default_f_max)
                delta_f = float(input(f"   FRequenzabstand [{default_delta_f}]: ") or default_delta_f)
                start = float(input(f"    Startzeit [{default_start}]: ") or default_start)
                end = float(input(f"    Endzeit [{default_end}]: ") or default_end)
            else:
                amp = default_amp
                f_max=default_f_max
                delta_f=default_delta_f
                start = default_start
                end = default_end

            signal_params.append({
                "type": 1,
                "amplitude": amp,
                "f_max":f_max,
                "delta_f": delta_f,
                "start": start,
                "end": end
            })

            '''
            ----
            s=2 und s=3 für Lineareer Drift und Impulsstörung vordefniert
            ----
            '''
    return selected_signals,signal_params

def get_user_signal_selection(settings):
    
    # TESTUMGEBUNG
    selected_signals,signal_params=get_interference_signal(settings.T_total)
    
    
    default_start=0.6 
    default_end=0.72
    default_Title="Signalverarbeitung gestörter Relaxationskurve"
    
    print("\n Bereich für die Plots festlegen (in Sekunden)")
    plot_start_input = input(f"  Startzeit [Standard {default_start}s]: ").strip()
    plot_end_input = input(f"  Endzeit   [Standard {default_end}s]: ").strip()
    plot_title = input(f"  Plot Titel   [Standard: {default_Title}]: ").strip() or default_Title
    plot_reference_in = input("    Referenzsignal plotten (j/n) [j]: ").strip().lower()
    plot_reference = True if plot_reference_in in {"", "j", "ja", "y"} else False

    
    
    plot_start = float(plot_start_input) if plot_start_input else default_start
    plot_end = float(plot_end_input) if plot_end_input else default_end
    
    if plot_start >= plot_end:
        print("Ungültiger Bereich, Standardwerte werden verwendet.")
        plot_start, plot_end = default_start, default_end

    ##########################################################
    ### Hier findet die Strategie wahö der Filterung statt ###
    ##########################################################
    filter_configs = []
    data_edit_configs=[]
    
    # Manuelle Abfrage 
    
    signal_names = {
        1: "Weißes Rauschen",
        2: "Linearer Drift",
        3: "Impulsrauschen",
        4: "Sinus-Störungen"
    }
    filter_configs.append(ask_filter_settings())

    data_edit_configs.append(ask_data_edit_settings(settings.T_total))

    ################
    #Reste von FFT Analyse
    plot_FFT = False
    fft_params = None
    ################

    return selected_signals, filter_configs, data_edit_configs,signal_params, plot_start, plot_end,plot_title,plot_reference,plot_FFT, fft_params



def get_data_filer_settings(settings,pre_filtering=False,real_data=False):
    
    # REALE DATEN
    default_start=0.6 
    default_end=0.72

    if real_data:
        default_start=8.42
        default_end=8.6
    default_Title="Signalverarbeitung gestörter Relaxationskurve"
    
    print("\n Bereich für die Plots festlegen (in Sekunden)")
    plot_start_input = input(f"  Startzeit [Standard {default_start}s]: ").strip()
    plot_end_input = input(f"  Endzeit   [Standard {default_end}s]: ").strip()
    plot_title = input(f"  Plot Titel   [Standard: {default_Title}]: ").strip() or default_Title
    plot_reference_in = input("    Referenzsignal plotten (j/n) [j]: ").strip().lower()
    plot_reference = True if plot_reference_in in {"", "j", "ja", "y"} else False

    
    
    plot_start = float(plot_start_input) if plot_start_input else default_start
    plot_end = float(plot_end_input) if plot_end_input else default_end
    
    if plot_start >= plot_end:
        print("Ungültiger Bereich, Standardwerte werden verwendet.")
        plot_start, plot_end = default_start, default_end

    
    filter_configs = []
    data_edit_configs=[]
    
    # Manuelle Abfrage 
    
    signal_names = {
        1: "Weißes Rauschen",
        2: "Linearer Drift",
        3: "Impulsrauschen",
        4: "Sinus-Störungen"
    }
    filter_configs.append(ask_filter_settings())

    data_edit_configs.append(ask_data_edit_settings(settings.T_total,real_data=real_data))

    if pre_filtering:
        return filter_configs, data_edit_configs, None, None,None,None,None, None

    plot_FFT = False
    fft_params = None
    

    return filter_configs, data_edit_configs, plot_start, plot_end,plot_title,plot_reference,plot_FFT, fft_params
