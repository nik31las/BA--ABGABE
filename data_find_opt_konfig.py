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
from data_analysis2 import filter_given_data,SignalSettings
from data_edit_methods import cut_signal_by_time, apply_data_editing_to_signal
from find_opt_konfig import save_summary_as_json
from choose_variation_logic import input_range,generate_filter_parameter_grid,generate_edit_data_parameter_grid


def iterate_column_filtering(combinations,configs,keys,
                             filter_combinations,filter_config,filter_keys,data_test_config,data_edit_configs,evaluation_history,settings,total_combination,counter,start_value=0):
    for i, combo in enumerate(combinations, 1):
        if total_combination < 25:
            print(f"Column_Setting: {configs[0]}")
        #print(f"Column_Setting {i}/{len(combinations)}")#: {test_config}")
        #holt sich eine vollständige data_config ab, nciht das irgenwelche WErte ggf nicht gesetzt sind
        test_config = configs[0].copy()

        #iteriert über die einzelnenen einträge der aktuellen Kombination
        for u, w in zip(keys, combo):
            #setzt diesen Wert gleich den neuen WErt der Kombination
            original_val = configs[0][u]
            test_config[u] = [w] if isinstance(original_val, list) else w

        

        ##### Hier rein Formale Gründe wegen Lsite als Datentyp ####
        column_test_configs=[]
        column_test_configs.append(test_config)
        ############################################################



        evaluation_history,counter = iterate_filter_configs_and_evaluate(filter_combinations,filter_config,filter_keys,data_test_config,data_edit_configs,evaluation_history,settings,column_configs=column_test_configs,total_combination=total_combination,counter=counter,start_value=start_value)

    return evaluation_history,counter





def iterate_filter_configs_and_evaluate(filter_combinations,filter_config,filter_keys,data_test_config,data_edit_configs,evaluation_history,settings, column_configs=None,total_combination=-1,counter=-1,start_value=0):
    for i, combo in enumerate(filter_combinations, 1):
        test_config = filter_config.copy()
        for k, v in zip(filter_keys, combo):
            test_config[k] = [v] if isinstance(filter_config[k], list) else v

        
        if total_combination < 25:
            print(f"🔍 Kombination {counter}/{total_combination}: {test_config}")
        else:   
            print(f"🔍 Kombination {counter}/{total_combination}")#: {test_config}")

        counter = counter +1

        if column_configs:
            relaxation_gesamt,gestörte_relaxation_gesamt, filtered_gesamt, labels, fig_s_FFTdisturbedSignal,settings,edit_data,t_start_effektiv, t_end_effektiv,t_total = filter_given_data(
                filter_configs=[test_config],
                settings=settings,
                plot_FFT=False, 
                fft_params=None,
                data_edit_configs=[data_test_config],
                t_start_wunsch=start_value,
                find_opt=True,
                col_filter_configs=column_configs
            )
            
        else:
            relaxation_gesamt,gestörte_relaxation_gesamt, filtered_gesamt, labels, fig_s_FFTdisturbedSignal,settings,edit_data,t_start_effektiv, t_end_effektiv,t_total = filter_given_data(
                filter_configs=[test_config],
                settings=settings,
                plot_FFT=False, 
                fft_params=None,
                data_edit_configs=[data_test_config],
                t_start_wunsch=start_value,
                find_opt=True
            )
        if data_edit_configs and data_test_config["edit_Data"]:
            edit_relaxation_totalSignal,edit_total_disturbed_relaxation,edit_total_filtered,edit_total_noise,start_cut,end_cut,edit_T_total,t_edit=edit_data
            edit_settings=SignalSettings()
            edit_settings.T_total=float(edit_T_total)
            if data_test_config["cut_Data"]:
                edit_settings.update_time_vector(new_num_points=len(edit_relaxation_totalSignal),T_total=edit_T_total,new_start=start_cut,new_end=end_cut)
        

            evaluation_results = evaluate_filtering(edit_relaxation_totalSignal, edit_total_filtered, edit_settings)
        else:
            evaluation_results = evaluate_filtering(relaxation_gesamt,filtered_gesamt,settings)

        entry = {
            "config": test_config.copy(),
            "data_config": data_test_config.copy(),
            "rmse": evaluation_results["rmse"],
            "mse": evaluation_results["mse"],
            "snr": evaluation_results["snr"],
            "correlation": evaluation_results["correlation"]
        }

        if column_configs:
            entry["column_configs"] = column_configs[0].copy()

        evaluation_history.append(entry)

        

    return evaluation_history,counter




def run_data_with_evaluation(filter_configs, signal_params, settings,repeat_calc=1,analyse_params=None,start_value=0):
    parameter_grid = generate_filter_parameter_grid(filter_configs)
    filter_config =filter_configs[0]
    
    
    data_edit_configs=[]
    
    data_edit_configs.append(ask_data_edit_settings(settings.T_total,real_data=True,given_Data_choice=True)) ## Nur zum Initalisieren der DAten deswegen given_Data_choice= True

    #Datenbearebeitungs MEthoden und zu MEthode Spaltenbearbeitungs - Filterungseinstelung
    data_edit_parameter_grid,column_filtering_parameter_grid,column_filter_configs = generate_edit_data_parameter_grid(settings.T_total, filter_configs, real_data=True)

    filter_keys, filter_values = zip(*parameter_grid.items())
    filter_combinations = list(itertools.product(*filter_values))
    print(f"\n🔁 Filter  {len(filter_combinations)} Kombinationen werden getestet...\n")

    data_keys, data_values = zip(*data_edit_parameter_grid.items())
    data_combinations = list(itertools.product(*data_values))

    print(f"\n🔁 Dataedit  {len(data_combinations)} Kombinationen werden getestet...\n")

    if column_filtering_parameter_grid :
        column_filter_keys, column_filter_values = zip(*column_filtering_parameter_grid.items())
        column_filter_combinations = list(itertools.product(*column_filter_values))
        print(f"\n🔁 Column-Filter  {len(column_filter_combinations)} Kombinationen werden getestet...\n")
    
        
        total_combination=len(filter_combinations)*len(data_combinations)*len(column_filter_combinations)
    else:
        total_combination=len(filter_combinations)*len(data_combinations)
    print(f"\n\n\n Insgesamt {total_combination} Kombinationen")

    counter = 0

    evaluation_history = []
    best_tracker = {
        "rmse": [],
        "mse": [],
        "snr": [],
        "correlation": []
    }

    for repeat in range(repeat_calc):
        #iterriert über alle data_combinationen
        for j, d_combo in enumerate(data_combinations, 1):

            #holt sich eine vollständige data_config ab, nciht das irgenwelche WErte ggf nicht gesetzt sind
            data_test_config = data_edit_configs[0].copy()

            print(f"\n▶ Wiederholung {repeat + 1}/{repeat_calc}")
            
            if total_combination < 25:
                print(f"Dataedit_Setting : {data_test_config}")
            #iteriert über die einzelnenen einträge der einen Kombination
            for u, w in zip(data_keys, d_combo):
                
                
                #setzt diesen Wert gleich den neuen WErt der Kombination
                original_val = data_edit_configs[0][u]
                data_test_config[u] = [w] if isinstance(original_val, list) else w
                


                

            if column_filtering_parameter_grid :
                
                
                # Hier wird die Filterung mit den Colum Filtereigenschaften durchgeführt und Iteriert
                evaluation_history,counter = iterate_column_filtering(combinations=column_filter_combinations,
                                                                configs=column_filter_configs,
                                                                keys=column_filter_keys,
                                                                filter_combinations=filter_combinations,
                                                                filter_config=filter_config,
                                                                filter_keys=filter_keys,
                                                                data_test_config=data_test_config,
                                                                data_edit_configs=data_edit_configs,
                                                                evaluation_history=evaluation_history,
                                                                settings=settings,
                                                                total_combination=total_combination,
                                                                counter=counter,
                                                                start_value=start_value)

            else:
                evaluation_history,counter = iterate_filter_configs_and_evaluate(filter_combinations,filter_config,filter_keys,data_test_config,data_edit_configs,evaluation_history,settings,total_combination=total_combination,counter=counter,start_value=start_value)
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
        if analyse_params is not None:
            counter_map = {param: Counter() for param in (analyse_params or best_tracker[key][0].keys())}

            for cfg in best_tracker[key]:
                for param in list(counter_map):  # aktuelle Schlüssel durchgehen
                    if param =="curvefit_bounds":
                        print("hi")
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
    start_value=float(input(f"Startzeitpunkt [Default={8.2}] :") or 8.2)

    filter_configs = []
    
    # Auswahl durch User
    #print("Auswählen welche Filter verwendet werden soll, die VAriation der PArameter wird anschließend definiert")
    filter_configs.append(ask_filter_settings(find_opt=True))
    


    #print(filter_configs)

    repeat_calc= int(input("Soll der Test mehrfach durchfeührt werden (Zahl) [1]: ")or 1)

    #if repeat_calc>1:
    analyse_params_input = input("Welche Parameter sollen untersucht werden (z. B. savgol_window,savgol_poly,notch_q)? [Alle]: ").strip()
    if analyse_params_input == "":
        analyse_params=None
    else:
        analyse_params = [p.strip() for p in analyse_params_input.split(",") if p.strip()] if analyse_params_input else None

    summary=run_data_with_evaluation(filter_configs,None,settings,repeat_calc,analyse_params,start_value=start_value)
    
    
    print("\n✅ Beste Parameterkombinationen:")

    # RMSE
    print("🔹 Niedrigstes RMSE:")
    print(f"   RMSE-Wert:   {summary['best_rmse']['rmse']:.4f}")
    print(f"   Filter-Konfiguration: {summary['best_rmse']['config']}")
    print(f"   Datenbearbeitung-Konfiguration: {summary['best_rmse']['data_config']}")
    if 'column_configs' in summary['best_rmse']:
        if summary['best_rmse']['column_configs']:
            print(f"   Spaltenfilterung-Konfiguration: {summary['best_rmse']['column_configs']}")

    # MSE
    print("\n🔹 Niedrigstes MSE:")
    print(f"   MSE-Wert:    {summary['best_mse']['mse']:.6f}")
    print(f"   Filter-Konfiguration: {summary['best_mse']['config']}")
    print(f"   Datenbearbeitung-Konfiguration: {summary['best_mse']['data_config']}")
    if 'column_configs' in summary['best_mse']:
        if summary['best_mse']['column_configs']:
            print(f"   Spaltenfilterung-Konfiguration: {summary['best_mse']['column_configs']}")

    # SNR
    print("\n🔹 Höchstes SNR:")
    print(f"   SNR-Wert:    {summary['best_snr']['snr']:.2f} dB")
    print(f"   Filter-Konfiguration: {summary['best_snr']['config']}")
    print(f"   Datenbearbeitung-Konfiguration: {summary['best_snr']['data_config']}")
    if 'column_configs' in summary['best_snr']:
        if summary['best_snr']['column_configs']:
            print(f"   Spaltenfilterung-Konfiguration: {summary['best_snr']['column_configs']}")

    # Korrelationskoeffizient
    print("\n🔹 Beste Korrelation (näher an 1):")
    print(f"   Korrelationswert: {summary['best_correlation']['correlation']:.4f}")
    print(f"   Filter-Konfiguration:     {summary['best_correlation']['config']}")
    print(f"   Datenbearbeitung-Konfiguration:     {summary['best_correlation']['data_config']}")
    if 'column_configs' in summary['best_correlation']:
        if summary['best_correlation']['column_configs']:
            print(f"   Spaltenfilterung-Konfiguration: {summary['best_correlation']['column_configs']}")


    save_input = input("\nZuum Speichern Geben sie 's' ein (sonst ENTER): ").strip().lower()

    if save_input =="s":
        file_name=input("Filename: ")
        #notch_folder=r"D:\MEchatronik\Bachlorarbeit\WORK\Testsignale\firstSignals\FindOpti\sinus\notch" #input("Folder:")
        ######
        #TODO#
        ######
        save_folder=r"D:\MEchatronik\Bachlorarbeit\WORK\Testsignale\secondSignals\FindOpti_Data"
        save_summary_as_json(summary, file_name,output_dir=save_folder)
