import numpy as np
import matplotlib.pyplot as plt
import sys

from create_signals import init_disturbsignals, relaxation_signal, extract_sequence
from filter_pip import filter_pipeline, apply_filters
from choose_logic import get_user_signal_selection
from quality_control import evaluate_filtering, print_evaluation_results
from visu import plot_signals, compareFrequencySpectra,plot_edited_signals_with_blocks
from save_data2 import save_signal_analysis
from data_edit_methods import apply_data_editing_to_signal



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
    
    def update_time_vector(self, t_cut,T_total,new_start,new_end):
        """Setzt t_dataPoints auf neue Start-/Endzeit mit aktueller Abtastrate."""
        num_points = len(t_cut)
        self.t_dataPoints = np.linspace(new_start, new_end, num_points, endpoint=False)
        self.T_total = T_total
        self.total_samples = len(t_cut)



def create_full_signal_with_pauses(
        signal_params, filter_configs,
        settings, 
        plot_FFT=True, fft_params=(0,300, 0, 0.02),data_edit_configs=None
    ):

    fig_s_FFTdisturbedSignal=[]
    labels= []
    edit_data=[]
    

    
    #
    total_noise= np.zeros(settings.total_samples)

    # 1. Relaxationssignal mit Pausen

    relaxation_totalSignal = np.full(settings.total_samples, np.nan, dtype=float)
    num_blocks = settings.total_samples // settings.samples_block

    for i in range(num_blocks):
        start_idx = i * settings.samples_block
        end_idx = start_idx + settings.samples_active
        t_segment = np.linspace(0, settings.active_ms / 1000, settings.samples_active, endpoint=False)
        segment = relaxation_signal(t_segment, 60, 1e-3)
        relaxation_totalSignal[start_idx:end_idx] = segment

    # 2. Störsignal kontinuierlich erzeugen (Beispiel: Sinusse)
    disturbed_Signals, labels = init_disturbsignals(settings.t_dataPoints, signal_params)


    #3 Gesamtesgestoertes Signal erzeugen
    for signal in disturbed_Signals:

        total_noise += signal # zusammengesetztes Signal
        
    total_disturbed_relaxation= relaxation_totalSignal +total_noise


    
    
    # 4. Filter anwenden (nur ein Filter-Set für alles)
    
    total_filtered = apply_filters(settings,total_disturbed_relaxation, **filter_configs[0])
    


    
    
    if data_edit_configs and data_edit_configs[0]["edit_Data"]:
        
        t_edit, edit_relaxation_totalSignal,cut_settings = apply_data_editing_to_signal(settings.t_dataPoints, relaxation_totalSignal, data_edit_configs[0],settings=settings)
        _, edit_total_disturbed_relaxation,_ = apply_data_editing_to_signal(settings.t_dataPoints, total_disturbed_relaxation, data_edit_configs[0],settings=settings)
        _, edit_total_filtered,_ = apply_data_editing_to_signal(settings.t_dataPoints, total_filtered, data_edit_configs[0],settings=settings,filter=True)
        _, edit_total_noise,_ = apply_data_editing_to_signal(settings.t_dataPoints, total_noise, data_edit_configs[0],settings=settings,filter=True)
        
        start_cut, end_cut , T_total= cut_settings
        edit_data=[edit_relaxation_totalSignal,edit_total_disturbed_relaxation,edit_total_filtered,edit_total_noise,start_cut,end_cut,T_total,t_edit]
        


    
    
    # 5. Optional: Frequenzvergleich plotten
    ### Deaktiviert
    '''if plot_FFT:
        sequenz_start=fft_params[2]
        sequenz_end=fft_params[3]

        active_disturbed, t_active = extract_sequence(total_disturbed_relaxation,settings.t_dataPoints,sequenz_start, sequenz_end)
        active_filtered, _ = extract_sequence(total_filtered, settings.t_dataPoints,sequenz_start, sequenz_end)
        active_relaxation, _ = extract_sequence(relaxation_totalSignal,settings.t_dataPoints,sequenz_start, sequenz_end)

        fig = compareFrequencySpectra(
        signal1=active_disturbed,
        signal2=active_filtered,
        compareSignal=active_relaxation,
        fs=settings.fs,
        label1="Gestört ",
        label2="Gefiltert",
        comparelabel="Relaxation",
        title="Frequenzvergleich - einzelne Sequenz",
        fft_params=fft_params
        ############
        #Hier Range bzw. Params einfügen 
        )
        fig_s_FFTdisturbedSignal.append(fig)'''
    
    

    return relaxation_totalSignal, total_disturbed_relaxation, total_noise, total_filtered, labels,fig_s_FFTdisturbedSignal, settings,edit_data

    

if __name__ == "__main__":
    settings = SignalSettings()
    
    settings.T_total=float(input(f"Gesamtzeiteingeben [Defaul={settings.T_total}] :") or settings.T_total)
    
    ######
    #TODO#
    ######
    #Zum Speichern
    output_dir= r"D:\MEchatronik\Bachlorarbeit\WORK\Testsignale\secondSignals\Results_Testumgebung"
    # Auswahl durch User
    selected_signals, filter_configs,data_edit_configs, signal_params, plot_start, plot_end,plot_title,plot_reference,in_plot_FFT, in_fft_params = get_user_signal_selection(settings)
    


    relaxation_gesamt,gestörte_relaxation_gesamt, störsignale_gesamt, filtered_gesamt, labels, fig_s_FFTdisturbedSignal,settings,edit_data = create_full_signal_with_pauses(
        signal_params,
        filter_configs,
        settings=settings,
        plot_FFT=in_plot_FFT, 
        fft_params=in_fft_params,
        data_edit_configs=data_edit_configs
    )

     
    evaluation_results=evaluate_filtering(relaxation_gesamt,filtered_gesamt,settings)
    #################################################################################
    # PArameter übersicht
    # relaxation_gesamt : einfach nur normale Relaxtion
    # störsignale_gesamt nur Störsignal, aber alles zusammengefasst
    # filtered_gesamt: wiederum kommplettes gefiltertes Signal
    #################################################################################
    
    
    # Plotten

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

    print("KEnnzahlen Filterung:")
    print_evaluation_results(labels, evaluation_results)


    if data_edit_configs and data_edit_configs[0]["edit_Data"]:
        
        edit_relaxation_totalSignal,edit_total_disturbed_relaxation,edit_total_filtered,edit_total_noise,start_cut,end_cut,edit_T_total,t_edit=edit_data
        edit_settings=SignalSettings()
        edit_settings.T_total=float(edit_T_total)
        if data_edit_configs[0]["cut_Data"]:
            edit_settings.update_time_vector(t_cut=t_edit,T_total=edit_T_total,new_start=start_cut,new_end=end_cut)
            
        
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
            save_signal_analysis(output_dir, edit_settings.t_dataPoints, edit_relaxation_totalSignal, edit_total_disturbed_relaxation, edit_total_filtered,
                                labels, edit_evaluation_results, fig_s_FFTdisturbedSignal, edit_fig,
                                signal_params, filter_configs,plot_start,plot_end, fs=settings.fs,data_edit_config=data_edit_configs)

   

    
    save_input = input("\nZuum Speichern Geben sie 's' ein (sonst ENTER): ").strip().lower()

    if save_input =="s":
        save_signal_analysis(output_dir, settings.t_dataPoints, relaxation_gesamt, gestörte_relaxation_gesamt, filtered_gesamt,
                            labels, evaluation_results, fig_s_FFTdisturbedSignal, figure_signals,
                            signal_params, filter_configs,plot_start,plot_end, fs=settings.fs)

    user_input = input("\nGib 'q' oder ENTER ein zum Beenden").strip().lower()
    if user_input == "q":
        print("Fenster wird geschlossen. Programmende.")
        plt.close('all')
        sys.exit()


