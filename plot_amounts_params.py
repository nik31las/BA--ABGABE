import json
import sys
import matplotlib.pyplot as plt
import os
from visu import  plot_param_grouped #,plot_param
from save_data2 import save_plot

def load_results_summary(path, file):
    filepath = os.path.join(path, file + ".json")
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

x_label=input("\nParameter Plotbeschriftung: ").strip() or "Polynomordnung p" #"Randbehandlung mode"#"Fensterbreite w"#
param=input("\nParametername Datensatz: ").strip() or "savgol_poly" #"savgol_mode"#"savgol_window"#

y_label="Häufigkeit"
file="addzero_interp_1pT"
path = rf"D:\MEchatronik\Bachlorarbeit\WORK\Testsignale\firstSignals\FindOpti\weissRauschen\savgol2"

results_summary = load_results_summary(path,file)

#plot_param(results_summary, metric="snr", param="savgol_window")
fig = plot_param_grouped(results_summary, param=param,custom_title="Vergleich Kennzahlen",x_label=x_label,y_label=y_label )

save_input = input("\nZuum Speichern Geben sie 's' ein (sonst ENTER): ").strip().lower()

if save_input == "s":
    save_plot(path, file, fig, param)
    print(f"gespeichert Ordner {path}")
    print(f"Filenam {file}{param}.pdf")


user_input = input("\nGib 'q' oder ENTER ein zum Beenden").strip().lower()
if user_input == "q":
    print("Fenster wird geschlossen. Programmende.")
    plt.close('all')
    sys.exit()