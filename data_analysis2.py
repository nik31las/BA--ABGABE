import numpy as np
import matplotlib.pyplot as plt
import sys

from create_signals import init_disturbsignals, relaxation_signal, extract_sequence
from filter_pip import filter_pipeline, apply_filters
from choose_logic import get_user_signal_selection,get_data_filer_settings,ask_filter_settings
from quality_control import evaluate_filtering, print_evaluation_results
from visu import plot_signals, compareFrequencySpectra,plot_edited_signals_with_blocks
from save_data2 import save_signal_analysis
from data_edit_methods import apply_data_editing_to_signal
from real_data_extraction import process_signal_from_file,align_signals_to_reference,align_signal_to_given_relaxation
from data_edit_methods import average_adjacent_sequences


from dataclasses import dataclass,field

@dataclass
class SignalSettings:
    fs: int = 1000
    T_total: float = 1.5
    active_ms: int = 20
    pause_ms: int = 10

    total_samples: int = field(init=False)
    samples_active: int = field(init=False)
    samples_pause: int = field(init=False)
    samples_block: int = field(init=False)
    t_dataPoints: np.ndarray = field(init=False)

    def __post_init__(self):
        self.total_samples = int(self.T_total * self.fs)
        self.samples_active = int((self.active_ms / 1000) * self.fs) #+ 1
        self.samples_pause = int((self.pause_ms / 1000) * self.fs) #- 1
        self.samples_block = self.samples_active + self.samples_pause
        self.t_dataPoints = np.linspace(0, self.T_total, self.total_samples, endpoint=False)
    
    def update_time_vector(self, new_num_points,T_total,new_start,new_end):
        """Setzt t_dataPoints auf neue Start-/Endzeit mit aktueller Abtastrate."""
        self.total_samples = new_num_points 
        self.t_dataPoints = np.linspace(new_start, new_end, new_num_points, endpoint=False)
        self.T_total = T_total
        



def filter_given_data(
        filter_configs,
        settings, 
        plot_FFT=True, fft_params=(0,300, 0, 0.02),data_edit_configs=None,find_opt=False, col_filter_configs=None,t_start_wunsch=0
    ):

    fig_s_FFTdisturbedSignal=[]
    labels= []
    edit_data=[]
    

    
    
    
    
                                                                                              
    dauer_wunsch = settings.T_total   
    ##############
    # Referenzierzungseinstellungen                                                            
    ref_data_measurements=False    # Variante mit Skalierung                                                                                         
    ref_data_given_relaxation=True # aktuelle Variante
    ##############
        
    ######                                
    #TODO#
    ######                                                    ######
    given_relaxation=r'D:\MEchatronik\Bachlorarbeit\WORK\Testsignale\firstSignals\reference_data\m4Clean_with_time.csv'
    ref_factor=1000
    scale_addaption=False

    if data_edit_configs:
        filter_column       = data_edit_configs[0]["colum_filtering"]
        cut_last_point      = data_edit_configs[0]["cut_last_datapoint"]
        invert              = data_edit_configs[0]["invert"]
        apply_remove_500Hz  = data_edit_configs[0]["apply_remove_500Hz"]
        pre_avg_sequenzen   = data_edit_configs[0]["pre_avg_sequenzen"]  # wird zum Entfernen von 50 Hz verwendet
        remove_empty_measurement = data_edit_configs[0]["remove_empty_measurement"]
    else:
        invert = True             # Signal um die Nulllinie spiegeln

        cut_last_point = True     # letzten Punkt jeder Sequenz entfernen

        filter_column=False        # Spalten filtern

        apply_remove_500Hz=True
        remove_empty_measurement=True

    # ###=== Parameter definieren === ### # noch abfrage implementieren
    ######                                
    #TODO#
    ######  
    # gestörter Datensatz, welche gefiltert werden soll
    csv_dateipfad = r'D:\MEchatronik\Bachlorarbeit\WORK\Testsignale\secondSignals\Daten\omg_log_20201208_202942.csv'
    spalten_index = 3         # 0-basiert, z. B. 3 = vierte Spalte (grad in nT)

    ######                                
    #TODO#
    ######
    ##### ###### Referenz Daten ####### #####
    # Einstellungen angepasst an Datensatz ##
    ref_csv_dateipfad = r'"D:\MEchatronik\Bachlorarbeit\WORK\Testsignale\secondSignals\Daten\omg_log_20201208_200454.csv"'
    ref_spalten_index = 3         # 0-basiert, z. B. 3 = vierte Spalte (grad in nT)
    ref_invert = True             # Signal um die Nulllinie spiegeln                                                        
    ref_cut_last_point = cut_last_point # True     # letzten Punkt jeder Sequenz entfernen                                  
    ref_apply_remove_500Hz =apply_remove_500Hz                                                                              
    rev_remove_empty_measurement=remove_empty_measurement                                                                   
    


    if filter_column:
        column_1=1
        column_2=2
        col_invert=False

        # Diese if Abfragen verhindern, dass bestimmte Bearbeitungsmethoden doppelt ausgeführt werden,
        # wenn die Spalten bereits gefiltert werden -> sonst Gefahr von Datenverfälschung
        if cut_last_point:
            cut_last_point=False
            col_cut_last_point=True
        else:
            cut_last_point=False

        if apply_remove_500Hz:
            col_apply_remove_500Hz=True
            apply_remove_500Hz=False
        
        if remove_empty_measurement:
            remove_empty_measurement=False
            col_remove_empty_measurement=True
        else:
            col_remove_empty_measurement=False

        
        # 
        if not find_opt:
            col_filter_configs=[]
            col_filter_configs.append(ask_filter_settings(find_opt))
            


        col1_total_disturbed_relaxation, col1_t_start_effektiv, col1_t_end_effektiv,col1_t_total = process_signal_from_file(
            filepath=csv_dateipfad,
            spalten_index=column_1,
            fs=settings.fs,
            t_start_wunsch=t_start_wunsch,
            dauer_wunsch=dauer_wunsch, # settings.T_total,
            settings=settings,
            invert=col_invert,
            cut_last_point=col_cut_last_point,
            apply_remove_500Hz=col_apply_remove_500Hz,
            remove_empty_measurement=col_remove_empty_measurement,
            spalte=1,
            find_opt_run=find_opt
        )

        col2_total_disturbed_relaxation, col2_t_start_effektiv, col2_t_end_effektiv,col2_t_total = process_signal_from_file(
            filepath=csv_dateipfad,
            spalten_index=column_2,
            fs=settings.fs,
            t_start_wunsch=t_start_wunsch,
            dauer_wunsch=dauer_wunsch, # settings.T_total,
            settings=settings,
            invert=col_invert,
            cut_last_point=col_cut_last_point,
            apply_remove_500Hz=col_apply_remove_500Hz,
            remove_empty_measurement=col_remove_empty_measurement,
            spalte=2,
            find_opt_run=find_opt
        )
        

        ################
        # Mittelung davor
        if pre_avg_sequenzen:
            _,col1_total_disturbed_relaxation=average_adjacent_sequences(col1_t_total,col1_total_disturbed_relaxation,settings.fs, settings.active_ms, settings.pause_ms)
            _,col2_total_disturbed_relaxation=average_adjacent_sequences(col2_t_total,col2_total_disturbed_relaxation,settings.fs, settings.active_ms, settings.pause_ms)
        
        col1_total_filtered = apply_filters(settings,col1_total_disturbed_relaxation, **col_filter_configs[0])
        col2_total_filtered = apply_filters(settings,col2_total_disturbed_relaxation, **col_filter_configs[0])

        col2_1_signal=col2_total_filtered-col1_total_filtered
        
    else:
        col2_1_signal=None
        col1_t_start_effektiv=None
        col1_t_end_effektiv=None
        col1_t_total=None






    # === Aufruf ===
    total_disturbed_relaxation, t_start_effektiv, t_end_effektiv,t_total = process_signal_from_file(
        filepath=csv_dateipfad,
        spalten_index=spalten_index,
        fs=settings.fs,
        t_start_wunsch=t_start_wunsch,
        dauer_wunsch=dauer_wunsch, # settings.T_total,
        settings=settings,
        invert=invert,
        cut_last_point=cut_last_point,
        apply_remove_500Hz=apply_remove_500Hz,
        remove_empty_measurement=remove_empty_measurement,
        given_signal_data=col2_1_signal,
        given_timedata=(col1_t_start_effektiv, col1_t_end_effektiv,col1_t_total),
        spalte=3,
        find_opt_run=find_opt
    )


    # === Referenz ===
    if ref_data_measurements:
        raw_relaxation_totalSignal, ref_t_start_effektiv, ref_t_end_effektiv,ref_t_total = process_signal_from_file(
            filepath=ref_csv_dateipfad,
            spalten_index=ref_spalten_index,
            fs=settings.fs,
            t_start_wunsch=t_start_wunsch,
            dauer_wunsch=dauer_wunsch, # settings.T_total,
            settings=settings,
            invert=ref_invert,
            cut_last_point=ref_cut_last_point,
            apply_remove_500Hz=ref_apply_remove_500Hz,
            remove_empty_measurement=rev_remove_empty_measurement,
            spalte=3,
            find_opt_run=find_opt
        )

        relaxation_totalSignal, ref_t_total_aligned = align_signals_to_reference(
            raw_relaxation_totalSignal,
            ref_t_total,
            reference_start_time=t_start_effektiv,
            reference_length=len(total_disturbed_relaxation),
            settings=settings,
            reference_factor=ref_factor,
            total_disturbed_relaxation=total_disturbed_relaxation,
            scale_addaption=scale_addaption
        )

    elif ref_data_given_relaxation:
         relaxation_totalSignal = align_signal_to_given_relaxation(
             given_relaxation,
             total_disturbed_relaxation,
             settings=settings)
    else:
        relaxation_totalSignal=None


    # === Ergebnisse weiterverarbeiten ===
    if not find_opt:
        print(f"Effektiver Startzeitpunkt: {t_start_effektiv:.3f} s")
        print(f"Effektives Ende:           {t_end_effektiv:.3f} s")
    
    settings.update_time_vector(new_num_points=len(total_disturbed_relaxation),T_total=dauer_wunsch,new_start=t_start_effektiv,new_end=t_end_effektiv)

    total_filtered = apply_filters(settings,total_disturbed_relaxation, **filter_configs[0])
    

    
    
    if data_edit_configs and data_edit_configs[0]["edit_Data"]:
        if relaxation_totalSignal is not None:
            t_edit, edit_relaxation_totalSignal,cut_settings = apply_data_editing_to_signal(settings.t_dataPoints, relaxation_totalSignal, data_edit_configs[0],settings=settings)
        else:
            edit_relaxation_totalSignal=None
        t_edit, edit_total_disturbed_relaxation,cut_settings = apply_data_editing_to_signal(settings.t_dataPoints, total_disturbed_relaxation, data_edit_configs[0],settings=settings)
        _, edit_total_filtered,_ = apply_data_editing_to_signal(settings.t_dataPoints, total_filtered, data_edit_configs[0],settings=settings,filter=True)
        
        start_cut, end_cut , T_total= cut_settings
        edit_data=[edit_relaxation_totalSignal,edit_total_disturbed_relaxation,edit_total_filtered,None,start_cut,end_cut,T_total,t_edit]
    
    
    return relaxation_totalSignal, total_disturbed_relaxation, total_filtered, labels,fig_s_FFTdisturbedSignal, settings,edit_data,t_start_effektiv, t_end_effektiv,t_total

    

if __name__ == "__main__":
    settings = SignalSettings()
    
    settings.T_total=float(input(f"Gesamtzeiteingeben [Default={settings.T_total}] :") or settings.T_total)
    
    default_start=8.2
    start_value=float(input(f"Startzeitpunkt [Default={8.2}] :") or default_start)
    
    ######
    #TODO#
    ######
    #Zum Speichern
    output_dir= r"D:\MEchatronik\Bachlorarbeit\WORK\Testsignale\secondSignals\Results"
    # Auswahl durch User
    print("Filterung nach Subtrahierung der Daten zur Relaxationskurve:")
    filter_configs,data_edit_configs, plot_start, plot_end,plot_title,plot_reference,in_plot_FFT, in_fft_params = get_data_filer_settings(settings,real_data=True)
    
    

    relaxation_gesamt,gestörte_relaxation_gesamt, filtered_gesamt, labels, fig_s_FFTdisturbedSignal,settings,edit_data,t_start_effektiv, t_end_effektiv,t_total = filter_given_data(
        filter_configs=filter_configs,
        settings=settings,
        plot_FFT=in_plot_FFT, 
        fft_params=in_fft_params,
        data_edit_configs=data_edit_configs,
        t_start_wunsch=start_value
    )

    
    evaluation_results=evaluate_filtering(relaxation_gesamt,filtered_gesamt,settings)
    ########################
    # PArameter übersicht
    # relaxation_gesamt : einfach nur normale Relaxtion
    # störsignale_gesamt nur Störsignal, aber alles zusammengefasst
    # filtered_gesamt: wiederum kommplettes gefiltertes Signal
    ########################
    
    
    if relaxation_gesamt is None:
        plot_reference= False
    
    
    
    figure_signals = plot_signals(
        settings, 
        relaxation_gesamt,
        gestörte_relaxation_gesamt, 
        filtered_gesamt, 
        labels, 
        plot_start,
        plot_end,
        plot_reference=plot_reference,
        title=plot_title
    )

    print("Kennzahlen Filterung:")
    print_evaluation_results(labels, evaluation_results)


    if data_edit_configs and data_edit_configs[0]["edit_Data"]:
        
        edit_relaxation_totalSignal,edit_total_disturbed_relaxation,edit_total_filtered,edit_total_noise,start_cut,end_cut,edit_T_total,t_edit=edit_data
        edit_settings=SignalSettings()
        edit_settings.T_total=float(edit_T_total)
        if data_edit_configs[0]["cut_Data"]:
            edit_settings.update_time_vector(new_num_points=len(t_edit) ,T_total=edit_T_total,new_start=start_cut,new_end=end_cut)
            
        
        edit_fig = plot_edited_signals_with_blocks(
            t_data=t_edit,
            relaxation_total=edit_relaxation_totalSignal,
            disturbed_signal=edit_total_disturbed_relaxation,
            filtered_signal=edit_total_filtered,
            settings=edit_settings,  # wichtig!
            labels=["Nach Bearbeitung"],
            plot_start=plot_start,
            plot_end=plot_end,
            plot_reference=True,
            title=plot_title
        )
        
        edit_evaluation_results=evaluate_filtering(edit_relaxation_totalSignal,edit_total_filtered,edit_settings)
        print("KEnnzahlen nach Datennachbearbeitung:")
        print_evaluation_results(labels,edit_evaluation_results)
        save_input_edit = input("\nZuum Speichern der Edit-Daten Geben sie 's' ein (sonst ENTER): ").strip().lower()

        if save_input_edit =="s":
            save_signal_analysis(output_dir, t_edit, edit_relaxation_totalSignal, edit_total_disturbed_relaxation, edit_total_filtered,
                                labels, edit_evaluation_results, fig_s_FFTdisturbedSignal, edit_fig,
                                None, filter_configs,plot_start,plot_end, fs=settings.fs,data_edit_config=data_edit_configs)

   

    
    save_input = input("\nZuum Speichern Geben sie 's' ein (sonst ENTER): ").strip().lower()

    if save_input =="s":
        save_signal_analysis(output_dir, t_total, relaxation_gesamt, gestörte_relaxation_gesamt, filtered_gesamt,
                            labels, evaluation_results, fig_s_FFTdisturbedSignal, figure_signals,
                            None, filter_configs,plot_start,plot_end, fs=settings.fs)

    user_input = input("\nGib 'q' oder ENTER ein zum Beenden").strip().lower()
    if user_input == "q":
        print("Fenster wird geschlossen. Programmende.")
        plt.close('all')
        sys.exit()


