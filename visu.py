import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq,fftshift
import numpy as np

def plot_param_grouped(results_summary, param="savgol_window", custom_title=None,x_label="Parameter",y_label="Häufigkeit"):
    """
    Plottet die Häufigkeit eines Parameters gruppiert über alle Metriken in einem gemeinsamen Plot.

    Parameter:
        results_summary (dict): Ergebnisdatenstruktur (aus JSON oder Lauf).
        param (str): Zu analysierender Parameter.
        custom_title (str or None): Optionaler Plot-Titel.
    """
    analysed = results_summary.get("results_analysed_params", {})
    if not analysed:
        print("⚠️ Keine analysierten Parameter vorhanden.")
        return

    # Sammle alle auftretenden Werte des Parameters über alle Metriken
    all_param_values = set()
    for metric_data in analysed.values():
        if param in metric_data:
            all_param_values.update(metric_data[param].keys())

    if not all_param_values:
        print(f"⚠️ Keine Daten zu '{param}' gefunden.")
        return

    try:
        sorted_values = sorted(all_param_values, key=lambda x: float(x))
    except ValueError:
        sorted_values = sorted(all_param_values)  # alphabetisch für Strings
    x = np.arange(len(sorted_values))
    width = 0.2

    metrics = list(analysed.keys())
    colors = plt.cm.Set2.colors

    fig = plt.figure(figsize=(10, 5))

    for idx, metric in enumerate(metrics):
        param_counts = analysed[metric].get(param, {})
        heights = [param_counts.get(v, 0) for v in sorted_values]
        plt.bar(x + idx * width, heights, width, label=metric.upper(), color=colors[idx % len(colors)])

    # Achsenbeschriftungen
    plt.xticks(x + width * (len(metrics) - 1) / 2, sorted_values, rotation=0,fontsize=20)  # ← horizontal
    plt.yticks(fontsize=20)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)

    # Titel
    if custom_title:
        plt.title(custom_title, fontsize=26)
    else:
        plt.title(f"Häufigkeit von '{param}' bei bester Metrik", fontsize=26)

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title="Metrik", fontsize="xx-large",loc='upper center',bbox_to_anchor=(0.820, 1.0))
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    return fig




def plot_signals(settings, relaxation_total, disturbed_Signal_total, filtered_Signal_total, labels: list, plot_start, plot_end,plot_reference=True,title="Signalvergleich: Referenz vs. Störung vs. Gefiltert"):
    
    if relaxation_total is None:
        plot_reference=False

    print("\n🔍 Filter-Zusammenfassung vor dem Plot:")
    for i, label in enumerate(labels, start=1):
        print(f"Signal {i}: {label}")
    print("\n→ Plot startet...\n")

    fig, ax = plt.subplots(figsize=(14, 5))
    first_plot = True  # ← oben vor der Schleife definieren

    
    for start in range(0, len(settings.t_dataPoints), settings.samples_block):
        end = start + settings.samples_active
        if end > len(settings.t_dataPoints):
            break

        # Nur anzeigen, wenn innerhalb des gewünschten Zeitbereichs
        if settings.t_dataPoints[end - 1] < plot_start or settings.t_dataPoints[start] > plot_end:
            continue
        
        
        segment_mask = (settings.t_dataPoints >= plot_start) & (settings.t_dataPoints <= plot_end)
        t_seg = settings.t_dataPoints[start:end][segment_mask[start:end]]
        

        
        if plot_reference:
            ax.plot(t_seg, relaxation_total[start:end][segment_mask[start:end]], 'r--',
                    label="Referenzsignal (Relaxation)" if first_plot else "")
        ax.plot(t_seg, disturbed_Signal_total[start:end][segment_mask[start:end]], 'b-',
                label="Gestörtes Signal" if first_plot else "")
        ax.plot(t_seg, filtered_Signal_total[start:end][segment_mask[start:end]], 'g-',
                label="Gefiltertes Signal" if first_plot else "", linewidth=2)

        first_plot = False
        
    ax.set_xlabel("Zeit (s)", fontsize=20)
    ax.set_ylabel("Magnetfeld (nT)", fontsize=20)
    ax.set_title(title, fontsize=26)
    ax.grid(True)
    ax.legend(loc='upper right', fontsize="xx-large")
    ax.tick_params(axis='both', labelsize=20)

    plt.tight_layout()
    
    plt.show(block=False)
    plt.pause(0.1)

    return fig





def plot_edited_signals_with_blocks(t_data, relaxation_total, disturbed_signal, filtered_signal, settings, labels, plot_start, plot_end, plot_reference=True, title="Signalvergleich (bearbeitet, aktiv)"):
    if relaxation_total is None:
        plot_reference=False

    print("\n🔍 Filter-Zusammenfassung (nur aktive Blöcke im Plotbereich, bearbeitet):")
    for i, label in enumerate(labels, start=1):
        print(f"Signal {i}: {label}")
    print("\n→ Plot startet...\n")

    fig, ax = plt.subplots(figsize=(14, 5))
    first_plot = True

    total_len = len(t_data)
    for start in range(0, total_len, settings.samples_block):
        end = start + settings.samples_active
        if end > total_len:
            break

        t_block = t_data[start:end]
        if len(t_block) == 0 or t_block[-1] < plot_start or t_block[0] > plot_end:
            continue  # Block komplett außerhalb des gewünschten Zeitbereichs

        # Maske für Zeitbereich innerhalb des Blocks
        mask = (t_block >= plot_start) & (t_block <= plot_end)
        if not np.any(mask):
            continue  # nichts zu plotten in diesem Block

        t_seg = t_block[mask]
        if plot_reference:
            relax_seg = relaxation_total[start:end][mask]
        disturb_seg = disturbed_signal[start:end][mask]
        filter_seg = filtered_signal[start:end][mask]

        if plot_reference:
            ax.plot(t_seg, relax_seg, 'r--', label="Referenzsignal (Relaxation)" if first_plot else "")
        ax.plot(t_seg, disturb_seg, 'b-', label="Gestörtes Signal" if first_plot else "")
        ax.plot(t_seg, filter_seg, 'g-', label="Gefiltertes Signal" if first_plot else "", linewidth=2)

        first_plot = False

    ax.set_xlabel("Zeit (s)", fontsize=20)
    ax.set_ylabel("Magnetfeld (nT)", fontsize=20)
    ax.set_title(title, fontsize=26)
    ax.grid(True)
    ax.legend(loc='upper right', fontsize="xx-large")
    ax.tick_params(axis='both', labelsize=20)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    return fig


def compareFrequencySpectra(signal1, signal2=None, compareSignal=None,
                            fs=5000,
                            label1="Signal 1", label2="Signal 2", comparelabel="Vergleichssignal",
                            title="Frequenzspektrum", threshold_factor=0.1,
                            fft_params=(0,300, 0, 0.02)
                            ):
    """
    Zeigt das Frequenzspektrum eines Signals – optional mit Vergleichssignalen.
    Frequenzdarstellung begrenzt auf max_freq Hz.
    """
    n = len(signal1)
    xf = fftshift(fftfreq(n, 1/fs))
    yf1 = fftshift(np.abs(fft(signal1)))
    yf2 = None
    yfc = None

    # Optional Vergleichssignale berechnen
    if signal2 is not None:
        if len(signal2) != n:
            raise ValueError("Beide Signale müssen die gleiche Länge haben.")
        yf2 = fftshift(np.abs(fft(signal2)))

    if compareSignal is not None:
        if len(compareSignal) != n:
            raise ValueError("Beide Signale müssen die gleiche Länge haben.")
        yfc = fftshift(np.abs(fft(compareSignal)))
    
    # Frequenzbegrenzung auf 0–max_freq
    mask = (xf >= fft_params[0]) & (xf <= fft_params[1])
    xf = xf[mask]
    yf1 = yf1[mask]
    if yf2 is not None:
        yf2 = yf2[mask]
    if yfc is not None:
        yfc = yfc[mask]
    
    # Plot vorbereiten
    plt.figure(figsize=(10, 4))
    plt.plot(xf, yf1, label=label1, linewidth=2, linestyle='-')
    if yf2 is not None:
        plt.plot(xf, yf2, label=label2, linewidth=1.8, linestyle='--')
    if yfc is not None:
        plt.plot(xf, yfc, label=comparelabel, linewidth=1.4, linestyle='-.')

    # Labels & Anzeige
    plt.title(title)
    plt.xlabel("Frequenz (Hz)]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    # Dominante Frequenzen
    print(f"\n {title}:")
    dom1 = xf[yf1 > threshold_factor * np.max(yf1)]
    print(f"→ Dominante Frequenzen ({label1}): {np.round(dom1, 2)} Hz")

    if yf2 is not None:
        dom2 = xf[yf2 > threshold_factor * np.max(yf2)]
        print(f"→ Dominante Frequenzen ({label2}): {np.round(dom2, 2)} Hz")

    if yfc is not None:
        dom3 = xf[yfc > threshold_factor * np.max(yfc)]
        print(f"→ Dominante Frequenzen ({comparelabel}): {np.round(dom3, 2)} Hz")

    return plt.gcf()

