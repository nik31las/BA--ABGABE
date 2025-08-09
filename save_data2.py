import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle




def get_filename_edit(data_edit_configs):
    config = data_edit_configs[0]

    if not config["edit_Data"]:
        return ""

    addon = "edit"

    if config.get("start_cut") is not None or config.get("end_cut") is not None:
        addon += "_cut"

    if config.get("avg_sequenzen"):
        addon += "_avg"

    return addon



def sanitize_filename(name):
    """Hilfsfunktion zur sicheren Ordner-/Dateibenennung."""
    return str(name).replace(" ", "_").replace(".", "_").replace(":", "").replace("/", "_")

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_safe(v) for v in obj)
    elif isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    elif callable(obj):
        return obj.__name__
    return obj



def format_bounds_for_filename(bounds, digits=3):
    """Wandelt Bounds in eine kurze string-Darstellung für Dateinamen um."""
    try:
        def short(val):
            # Runden und "." → "p" für Dateinamen-Kompatibilität
            return f"{round(val, digits)}".replace(".", "p").replace("-", "m")

        lower = "_".join([short(v) for v in bounds[0]])
        upper = "_".join([short(v) for v in bounds[1]])
        return f"{lower}__{upper}"
    except Exception:
        return "unknown_bounds"
    


def select_sig_folder(signal_params):
    """
    WR: Weißes Rauschen
    D: Drift
    I: Impuls
    S: Sinus
    Erzeugt eine kurze, strukturierte Ordnerbeschreibung aus den Signalparametern.
    """
    def shorten_signal_description(sig_param):
        if sig_param["type"] == 1:
            return f"WR_a{sig_param['amplitude']}_t{sig_param['start']}-{sig_param['end']}"
        elif sig_param["type"] == 2:
            return f"D{sig_param['start']}-{sig_param['end']}"
        elif sig_param["type"] == 3:
            return f"I_a{sig_param['amplitude']}_p{sig_param['period']}_t{sig_param['start']}-{sig_param['end']}"
        elif sig_param["type"] == 4:
            sin_strs = [f"{f}HzA{a}_t{s}-{e}" for f, a, s, e in sig_param["sinusoids"]]
            return "S_" + "_".join(sin_strs)
        else:
            return f"T{sig_param['type']}"

    stör_beschreibungen = [shorten_signal_description(p) for p in signal_params]
    sig_folder = sanitize_filename("__".join(stör_beschreibungen))
    return sig_folder

def shorten_filename(label_combined):
    label_combined = label_combined.replace("Sinusrauschen", "S")
    label_combined = label_combined.replace("Weißes", "W")
    label_combined = label_combined.replace("_Rauschen", "R")
    label_combined = label_combined.replace("Impulsrauschen", "I")
    label_combined = label_combined.replace("LinearerDrift", "D")

    return label_combined

def select_file_name(filter_param,labels,plot_start,plot_end,data_edit_configs):
    # Filterparameter & Dateinamen
    
    filter_types = []
    filename_parts = []

    if filter_param.get("apply_notch", False):
        filter_types.append("Notch")
        freqs = "_".join([f"{f}Hz" for f in filter_param.get("notch_freqs", [])])
        q = filter_param.get("notch_q", "Q?")
        filename_parts.append(f"Freq{freqs}_Q{q}")

    if filter_param.get("apply_bandpass", False):
        filter_types.append("Bandpass")
        low = filter_param.get("bandpass_low", "?")
        high = filter_param.get("bandpass_high", "?")
        order = filter_param.get("bandpass_order", "?")
        filename_parts.append(f"BP_{low}-{high}_ord{order}")

    if filter_param.get("apply_median", False):
        filter_types.append("Median")
        k = filter_param.get("median_kernel_size", "?")
        filename_parts.append(f"Med_k{k}")

    if filter_param.get("detrend_signal", False):
        filter_types.append("Driftentfernung")
        method = filter_param.get("detrend_method", "??")
        cutoff = filter_param.get("detrend_cutoff", "?")
        filename_parts.append(f"Detr_{method}_{cutoff}Hz")

    if filter_param.get("apply_savgol", False):
        filter_types.append("Savgol")
        window = filter_param.get("savgol_window", "??")
        poly = filter_param.get("savgol_poly", "?")
        cut = filter_param.get("savgol_cut_signal_sequenz", "?")
        mode= filter_param.get("savgol_mode","?")
        filename_parts.append(f"w_{window}_p{poly}_cut{cut}_mode-{mode}_plot{plot_start}-{plot_end}")


    if filter_param.get("apply_curvefit", False):
        filter_types.append("Fitting")
        model = filter_param.get("curvefit_model", "??")
        p0 = filter_param.get("curvefit_p0", "?")
        bounds = filter_param.get("curvefit_bounds", "?")
        
        method= filter_param.get("curvefit_method","?")
        filename_parts.append(f"method{method}_plot{plot_start}-{plot_end}")

    if not filter_types:
        filter_types = ["Unfiltered"]
        filename_parts.append("None")
    
    if data_edit_configs and data_edit_configs[0]["edit_Data"]:
        filename_parts.append(get_filename_edit(data_edit_configs))

    # 3. Finale Dateibenennung
    label_combined = sanitize_filename("_".join(labels))
    label_combined= shorten_filename(label_combined)
    file_label = label_combined + "_" + "_".join(filename_parts)
    filter_subfolder = "FILTER__" + "_".join(filename_parts)

    return filter_types, file_label, filter_subfolder

def save_plot(folder, label, fig, addon):
    os.makedirs(folder, exist_ok=True)  # <- Ordner sicherstellen
    fig.savefig(os.path.join(folder, label + f"_{addon}.pdf"))
    with open(os.path.join(folder, label + f"_{addon}.fig.pickle"), "wb") as f:
        pickle.dump(fig, f)
    plt.close(fig)

def save_evaluation_results(base_folder,file_label,filter_param,ev_results):
    with open(os.path.join(base_folder, f"{file_label}_filter_params.json"), 'w') as f:
            json.dump(make_json_safe(filter_param), f, indent=4)

def get_fallback_paths(output_dir):
    """
    Erzeugt einen Kurzpfad-Fallback, falls signal_params fehlt oder Pfad ungültig ist.
    
    Returns:
    - base_folder: Pfad zum Basisordner
    - freq_folder: Pfad zum Frequenz-Unterordner
    - time_folder: Pfad zum Zeit-Unterordner
    """
    short_base = os.path.join(output_dir, "Results_SHORT")
    base_folder = os.path.join(short_base, "Fallback")
    freq_folder = os.path.join(base_folder, "Frequenz")
    time_folder = os.path.join(base_folder, "Zeit")

    os.makedirs(freq_folder, exist_ok=True)
    os.makedirs(time_folder, exist_ok=True)
    os.makedirs(base_folder, exist_ok=True)

    return base_folder, freq_folder, time_folder

def check_path_length(path_parts, file_label):
    # Erstelle eine temporäre Liste, die alle Pfadteile und den Dateinamen enthält.
    full_path_parts = path_parts + [file_label]
    
    # Füge alle Teile zu einem Pfad-String zusammen und gib seine Länge zurück.
    potential_path = os.path.sep.join(full_path_parts)
    return len(potential_path)


def save_signal_analysis(output_dir, t_gesamt, relaxation_gesamt, gestörte_relaxation_gesamt, filtered_gesamt,
                         labels, evaluation_results, fig_s_FFTdisturbedSignal, figure_signals,
                         signal_params, filter_configs,plot_start,plot_end, fs=1000,data_edit_config=None):
    #kontrolliert ob alle VAriabeln gesetzt sind für auotmaitsche Namensspeicherung
    ordinary_save=True

    """
    Strukturierter Export für kombiniertes Störsignal (nur ein Störsignal-Gesamtsignal),
    mit zusammengeführtem Namen aller enthaltenen Störsignale.
    """
    if signal_params is None or filter_configs is None:
        base_folder, freq_folder, time_folder = get_fallback_paths(output_dir)
        ordinary_save=False

    if ordinary_save:   
        sig_folder=select_sig_folder(signal_params)

        filter_param = filter_configs[0]

        filter_types, file_label,filter_subfolder  =select_file_name(filter_param,labels,plot_start,plot_end,data_edit_config)
    else:
        filter_param = {}  # Leere oder Default-Parameter als Platzhalter
        file_label = "Fallback_Save"
        filter_types = ["Unfiltered"]
        filter_subfolder = "Default"
     

    for f_type in filter_types:
        if ordinary_save:

            path_parts = [output_dir, sig_folder, sanitize_filename(f_type), sanitize_filename(filter_subfolder)]
            MAX_PATH_LENGTH = 250

            if check_path_length(path_parts,file_label) > MAX_PATH_LENGTH:
                print("⚠️ Potenzieller Pfad ist zu lang. Wechsle zu Fallback.")
                base_folder, freq_folder, time_folder = get_fallback_paths(output_dir)
            else:
                # Der Pfad ist kurz genug, jetzt bauen wir ihn korrekt mit os.path.join
                base_folder = os.path.join(*path_parts) # os.path.join kann eine Liste von Argumenten annehmen
                freq_folder = os.path.join(base_folder, "Frequenz")
                time_folder = os.path.join(base_folder, "Zeit")
            # Ordnerstruktur erzeugen
            
            try:
                os.makedirs(freq_folder, exist_ok=True)
                os.makedirs(time_folder, exist_ok=True)
                #os.makedirs(base_folder, exist_ok=True)
            except FileNotFoundError as e:
                print(f"\n⚠️ Pfad zu lang oder ungültig – wechsle zu Kurzpfad.\n📁 Ursprünglicher Pfad:\n{base_folder}\n")
                # Fallback: Kurzer Speicherpfad
                base_folder, freq_folder, time_folder = get_fallback_paths(output_dir)
                print(f"✅ Wechsle zu Fallback-Pfad: {base_folder}")
    
                # Versuche, die Fallback-Pfade zu erstellen
                try:
                    os.makedirs(freq_folder, exist_ok=True)
                    os.makedirs(time_folder, exist_ok=True)
                except Exception as fallback_e:
                    print(f"❌ Fehler beim Erstellen der Fallback-Pfade: {fallback_e}")

        # CSV-Daten speichern
        csv_path = os.path.join(base_folder, f"{file_label}_data.csv")
        
        # Alle Arrays in 1D umwandeln
        t_gesamt = np.ravel(t_gesamt)
        relaxation_gesamt = np.ravel(relaxation_gesamt)

        gestörte_relaxation_gesamt= np.ravel(gestörte_relaxation_gesamt)
        filtered_gesamt = np.ravel(filtered_gesamt)

        # Prüfen, ob alle dieselbe Länge haben
        lengths = [len(t_gesamt), len(relaxation_gesamt), len(gestörte_relaxation_gesamt),len(filtered_gesamt)]
        if len(set(lengths)) != 1:
            raise ValueError(f"Nicht alle Signale haben dieselbe Länge: {lengths}")

        # CSV-Daten generieren und speichern
        data_array = np.column_stack((t_gesamt, relaxation_gesamt, gestörte_relaxation_gesamt , filtered_gesamt))
        np.savetxt(csv_path, data_array, delimiter=",", header="Zeit,Relaxation,GestoerteRElaxation,Gefiltert,Stoersignal", comments='')


        # JSON-Parameter speichern
        with open(os.path.join(base_folder, f"{file_label}_filter_params.json"), 'w') as f:
            json.dump(make_json_safe(filter_param), f, indent=4)
        with open(os.path.join(base_folder, f"{file_label}_signal_params.json"), 'w') as f:
            json.dump(make_json_safe(signal_params), f, indent=4)
        
        #Save Evaluation Results
        save_evaluation_results(base_folder,file_label,filter_param,evaluation_results)
        # Frequenzplot
        if fig_s_FFTdisturbedSignal:
            freq_addon="frequenz"
            save_plot(freq_folder,file_label,fig_s_FFTdisturbedSignal[0],freq_addon)
            

        # Zeitplot
        if figure_signals:
            fig_addon="zeitverlauf"
            save_plot(time_folder,file_label,figure_signals,fig_addon)
            

    print(f"\n Kombinierte Analyse gespeichert unter: {os.path.abspath(output_dir)}")
