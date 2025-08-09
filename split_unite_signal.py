import numpy as np
# Funktionen
#


def select_every_nth_sequence(signal, settings, nth=2, start_index=0):
    """
    Gibt jede n-te aktive Sequenz ausgehend vom gegebenen Startindex zurück.
    start_index = 0 → ungerade Blöcke (Index 0, 2, 4, ...)
    start_index = 1 → gerade Blöcke (Index 1, 3, 5, ...)
    """
    all_sequences = extract_active_sequences(signal, settings)
    return all_sequences[start_index::nth]


def extract_active_sequences(signal, settings):
    total_blocks = settings.total_samples // settings.samples_block
    active_sequences = []

    for i in range(total_blocks):
        start = i * settings.samples_block
        end = start + settings.samples_active
        if end <= len(signal):
            active_sequences.append(signal[start:end])

    return active_sequences


def reshape_flat_to_blocks(flat_signal, settings):
    """
    Teilt ein 1D-Array wieder in Blöcke der Länge samples_active auf.
    """
    block_len = settings.samples_active
    num_blocks = len(flat_signal) // block_len
    blocks = [flat_signal[i * block_len:(i + 1) * block_len] for i in range(num_blocks)]
    return blocks


def reconstruct_signal_with_pauses(sequences, settings, fill_value=np.nan):
    full_signal = []

    for seq in sequences:
        full_signal.extend(seq)
        full_signal.extend([fill_value] * settings.samples_pause)

    full_signal = full_signal[:settings.total_samples]
    return np.array(full_signal)