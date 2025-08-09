import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
import os
import json
from collections import Counter
import ast

from create_signals import init_disturbsignals, relaxation_signal, extract_sequence
from filter_pip import filter_pipeline, apply_filters
from choose_logic import get_interference_signal,ask_filter_settings,ask_data_edit_settings
from quality_control import evaluate_filtering, print_evaluation_results
from visu import plot_signals, compareFrequencySpectra
from save_data2 import save_signal_analysis,save_evaluation_results,select_file_name,select_sig_folder,sanitize_filename
from signals_v2 import create_full_signal_with_pauses
from data_edit_methods import cut_signal_by_time, apply_data_editing_to_signal

from signals_v2 import SignalSettings
from choose_variation_logic import input_range,generate_filter_parameter_grid




def make_serializable(obj):
    """Rekursiv alle nicht-serialisierbaren Objekte entfernen oder umwandeln."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items() if not callable(v)}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif callable(obj):
        return str(obj)  # oder einfach: return None
    else:
        return obj
    

def save_summary_as_json(summary_data, filename_base, output_dir="."):
    """
    Speichert die gegebene Ergebnis-Zusammenfassung als JSON-Datei.

    Parameter:
    - summary_data: dict, die Rückgabe von run_dynamic_filter_param_tests_with_evaluation()
    - filename_base: Dateiname ohne Endung (z. B. "sinus_test_q_variation")
    - output_dir: Pfad zum Speicherordner (Standard: aktuelles Verzeichnis)
    """
    filename = f"{filename_base}.json"
    path = os.path.join(output_dir, filename)

    # Konvertieren zu einer serialisierbaren Struktur
    summary_serializable = make_serializable(summary_data)

    with open(path, "w") as f:
        json.dump(summary_serializable, f, indent=4)

    print(f"✅ Ergebnisse gespeichert unter: {os.path.abspath(path)}")



def run_dynamic_filter_param_tests_with_evaluation(filter_configs, signal_params,data_edit_configs, settings,repeat_calc=1,analyse_params=None):
    parameter_grid = generate_filter_parameter_grid(filter_configs)
    filter_config =filter_configs[0]

    keys, values = zip(*parameter_grid.items())
    combinations = list(itertools.product(*values))
    print(f"\n🔁 {len(combinations)} Kombinationen werden getestet...\n")

    evaluation_history = []
    best_tracker = {
        "rmse": [],
        "mse": [],
        "snr": [],
        "correlation": []
    }

    for repeat in range(repeat_calc):
        print(f"\n▶ Wiederholung {repeat + 1}/{repeat_calc}")
        for i, combo in enumerate(combinations, 1):
            test_config = filter_config.copy()
            for k, v in zip(keys, combo):
                test_config[k] = [v] if isinstance(filter_config[k], list) else v

            

            relaxation_gesamt, _, _, filtered_gesamt, labels, _ ,settings,edit_data= create_full_signal_with_pauses(
                signal_params,
                [test_config],
                settings,
                plot_FFT=False,
                fft_params=None,
                data_edit_configs=data_edit_configs
            )
            if data_edit_configs and data_edit_configs[0]["edit_Data"]:
                edit_relaxation_totalSignal,edit_total_disturbed_relaxation,edit_total_filtered,edit_total_noise,start_cut,end_cut,edit_T_total,t_edit=edit_data
                edit_settings=SignalSettings()
                edit_settings.T_total=float(edit_T_total)
                if data_edit_configs[0]["cut_Data"]:
                    edit_settings.update_time_vector(t_cut=t_edit,T_total=edit_T_total,new_start=start_cut,new_end=end_cut)
            

                evaluation_results = evaluate_filtering(edit_relaxation_totalSignal, edit_total_filtered, edit_settings)
            else:
                evaluation_results = evaluate_filtering(relaxation_gesamt,filtered_gesamt,settings)

            entry = {
                "config": test_config.copy(),
                    "rmse": evaluation_results["rmse"],
                    "mse": evaluation_results["mse"],
                    "snr": evaluation_results["snr"],
                    "correlation": evaluation_results["correlation"]
            }
            

            evaluation_history.append(entry)

        # Beste Konfigurationen dieser Wiederholung extrahieren
        best_rmse = min(evaluation_history, key=lambda x: x["rmse"])
        best_mse = min(evaluation_history, key=lambda x: x["mse"])
        best_snr = max(evaluation_history, key=lambda x: x["snr"])
        best_corr = min(evaluation_history, key=lambda x: abs(x["correlation"] - 1))

        best_tracker["rmse"].append(best_rmse["config"])
        best_tracker["mse"].append(best_mse["config"])
        best_tracker["snr"].append(best_snr["config"])
        best_tracker["correlation"].append(best_corr["config"])

        print("\n📊 Häufigkeit der besten Parameter pro Kennwert:")
    
    results_analysed_params = {}
    for key in best_tracker:
        print(f"\n🔹 Beste {key.upper()}-Konfigurationen:")
        counter_map = {param: Counter() for param in (analyse_params or best_tracker[key][0].keys())}
        if analyse_params is not None:
            for cfg in best_tracker[key]:
                for param in list(counter_map):  # aktuelle Schlüssel durchgehen
                    if param =="curvefit_bounds":
                        continue
                    val = cfg.get(param)

                    # Fall: Liste von Listen → z. B. [[4.0, 10.0]]
                    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
                        val = val[0]  # auf [4.0, 10.0] reduzieren

                    if isinstance(val, list):
                        for i, v in enumerate(val):
                            sub_key = f"{param}_{i+1}"
                            if sub_key not in counter_map:
                                counter_map[sub_key] = Counter()
                            counter_map[sub_key][v] += 1
                    else:
                        counter_map[param][val] += 1

            # Ausgabe und Speicherung
            merged_counters = {}
            for param, counter in counter_map.items():
                print(f"  → {param}:")
                for val, count in sorted(counter.items()):
                    print(f"     {val}: {count}x")
                merged_counters[param] = dict(counter)

            results_analysed_params[key] = merged_counters

    # Beste Konfigurationen extrahieren
    def best_by(metric, key_fn, reverse=False):
        return min(evaluation_history, key=lambda x: key_fn(x[metric]), default=None) if not reverse else max(evaluation_history, key=lambda x: key_fn(x[metric]), default=None)

    best_rmse = best_by("rmse", lambda x: x)
    best_mse = best_by("mse", lambda x: x)
    best_snr = best_by("snr", lambda x: x, reverse=True)
    best_correlation = best_by("correlation", lambda x: abs(x - 1), reverse=False)

    results_summary = {
        "best_rmse": best_rmse,
        "best_mse": best_mse,
        "best_snr": best_snr,
        "best_correlation": best_correlation,
        #"all_evaluations": evaluation_history,
        "results_analysed_params":results_analysed_params
    }

    return results_summary


if __name__ == "__main__":


    settings = SignalSettings()
    settings.T_total=float(input(f"Gesamtzeiteingeben [Defaul={settings.T_total}] :") or settings.T_total)


    filter_configs = []
    data_edit_configs=[]
    # Auswahl durch User
    selected_signals,signal_params = get_interference_signal(settings.T_total)

    signal_names = {
                1: "Weißes Rauschen",
                2: "Linearer Drift",
                3: "Impulsrauschen",
                4: "Sinus-Störungen"
            }
    
    filter_configs.append(ask_filter_settings(find_opt=True))
    data_edit_configs.append(ask_data_edit_settings(settings.T_total))


    #print(filter_configs)


    repeat_calc=int(input("Soll der Test mehrfach durchfeührt werden (Zahl) [1]: ")or 1)

    #if repeat_calc>1:
    analyse_params_input = input("Welche Parameter sollen untersucht werden (z. B. savgol_window,savgol_poly,notch_q)? [Alle]: ").strip()
    analyse_params = [p.strip() for p in analyse_params_input.split(",") if p.strip()] if analyse_params_input else None

    summary=run_dynamic_filter_param_tests_with_evaluation(filter_configs,signal_params,data_edit_configs,settings,repeat_calc,analyse_params)
    
    
    print("\n✅ Beste Parameterkombinationen:")

    # RMSE
    print("🔹 Niedrigstes RMSE:")
    print(f"   RMSE-Wert:   {summary['best_rmse']['rmse']:.4f}")
    print(f"   Konfiguration: {summary['best_rmse']['config']}")

    # MSE
    print("\n🔹 Niedrigstes MSE:")
    print(f"   MSE-Wert:    {summary['best_mse']['mse']:.6f}")
    print(f"   Konfiguration: {summary['best_mse']['config']}")

    # SNR
    print("\n🔹 Höchstes SNR:")
    print(f"   SNR-Wert:    {summary['best_snr']['snr']:.2f} dB")
    print(f"   Konfiguration: {summary['best_snr']['config']}")

    # Korrelationskoeffizient
    print("\n🔹 Beste Korrelation (näher an 1):")
    print(f"   Korrelationswert: {summary['best_correlation']['correlation']:.4f}")
    print(f"   Konfiguration:     {summary['best_correlation']['config']}")


    save_input = input("\nZuum Speichern Geben sie 's' ein (sonst ENTER): ").strip().lower()

    if save_input =="s":
        file_name=input("Filename: ")
        
        ######
        #TODO#
        ######
        save_folder=r"D:\MEchatronik\Bachlorarbeit\WORK\Testsignale\secondSignals\FindOpti_Testumgebung"
        save_summary_as_json(summary, file_name,output_dir=save_folder)
