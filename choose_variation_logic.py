import itertools
import numpy as np
import ast
from choose_logic import ask_filter_settings


def estimate_total_combinations(parameter_grid):
    total = 1
    for values in parameter_grid.values():
        if isinstance(values, (list, np.ndarray)):
            total *= len(values)
    return total

def parse_liste_s_input(user_input, default_val=None, default_start=None, default_end=None, default_step=None):
    if not user_input.lower().startswith("liste-s:"):
        return None  # oder raise ValueError, je nach Anwendungsfall

    raw_data = user_input[8:].strip()

    # Versuch: Bereichsmodus mit Zahlen
    try:
        start, end, step = map(float, raw_data.split(","))
        return np.arange(start, end + step, step)
    except ValueError:
        pass  # Falls keine drei Zahlen: als Liste weiterverarbeiten

    # Strings als Einträge, "+" wird als Komma innerhalb des Strings ersetzt
    raw_elements = [entry.strip() for entry in raw_data.split(",") if entry.strip()]
    result = [elem.replace("+", ",") for elem in raw_elements]

    return result


def parse_number_range(user_input):
    start, end, step = map(float, user_input.split(","))
    return np.arange(start, end + step, step)

def parse_input_range_list_mode(user_input, default_val):
    try:
        user_input = user_input.strip()[6:].strip()  # Entferne "Liste:"
        entries = [entry.strip() for entry in user_input.split(";")]
        value_lists = []

        for entry in entries:
            if entry.startswith("="):
                val = float(entry[1:].strip())
                value_lists.append([val])
            else:
                try:
                    start, end, step = map(float, entry.split(","))
                    value_lists.append(list(np.arange(start, end + step, step)))
                except ValueError as e:
                    print(f"❌ Ungültiger Bereich in Liste: {entry} → {e}")
                    return default_val

        # Alle Kombinationen
        combinations = list(itertools.product(*value_lists))
        return [[float(v) for v in c] for c in combinations]
    except Exception as e:
        print(f"❌ Fehler beim Parsen von Liste: {e}. Verwende Default.")
        return default_val
    

def input_range(prompt, default_start=None, default_end=None, default_step=None, default_val=None):
    
    user_input = input(
        f"{prompt}  [Default: {default_val}]: "
    ).strip()

    if user_input in {"-", "!", "n", "N"}:
        return "KEEP_DEFAULT"


    # Mehrere Bereiche als Liste kombinieren (z. B. "Liste: 0,1,0.5 ; =30")
    if user_input.lower().startswith("liste:"):
       
        return parse_input_range_list_mode(user_input, default_val)
    

    # Sonderfall: explizit als Tupel-Liste gekennzeichnet
    if user_input.lower().startswith("tupel:"):
        raw_data = user_input[6:].strip()
        try:
            parsed = ast.literal_eval(raw_data)
            return parsed
        except Exception as e:
            print(f"❌ Fehler beim Parsen von Tupel-Eingabe: {e}. Verwende Default.")
            return default_val

    # Textmodus (z. B. "lm", "trf", "interp")
    if isinstance(default_val, str):
        if user_input.startswith("="):
            return user_input[1:].strip()
        elif "," in user_input:
            return [v.strip() for v in user_input.split(",") if v.strip()]
        elif user_input:
            return [user_input.strip()]
        else:
            return default_val
        
    if user_input.lower().startswith("b:"):
        bools = []
        for v in user_input[2:].split(","):
            val = v.strip().lower()
            if val == "true":
                bools.append(True)
            elif val == "false":
                bools.append(False)
            else:
                print(f"❌ Ungültiger Eintrag: {v}. Ignoriere.")
        return bools if bools else default_val
    
    # Fixwert-Modus (für einzelne Zahlen)
    if user_input.startswith("="):
        try:
            return float(user_input[1:].strip())
        except ValueError:
            print("❌ Ungültiger fixer Wert. Verwende Default.")
            return default_val

    if user_input.lower().startswith("liste-s:"):
        value=parse_liste_s_input(user_input)
        print(f"Liste zu {value} formatiert")
        return value
    # Bereichsmodus mit Zahlen (start,end,step)
    try:
        return parse_number_range(user_input)
    except ValueError:
        print("❌ Ungültiger Bereich. Verwende Default.")
        if None not in (default_start, default_end, default_step):
            return np.arange(default_start, default_end + default_step, default_step)
        else:
            return default_val
        

def generate_filter_parameter_grid(filter_configs):
    """
    Erstellt ein Parameter-Grid für die Optimierung aktiver Filter aus einer Liste von Konfigurationen.

    Parameters:
    - filter_configs: list of dict  Konfigurationen mit Filterparametern

    Returns:
    - parameter_grid: dict Mapping von Parametername zu Liste möglicher Werte
    """
    parameter_grid = {}
    # Falls kein filter angewendet wird, dann Default VAriante auswählen
    default_filter_config = filter_configs[0]
    # Schwellenwert für zu viele Kombinationen
    MAX_COMBINATIONS_FOR_VERBOSE_OUTPUT = 100

    for filter_config in filter_configs:
        for key, value in filter_config.items():
            
            if key.startswith("apply_") and value is True:
                filter_name = key.replace("apply_", "")
                print(f"\n🎛 Aktiver Filter: {filter_name}")

                param_keys = [k for k in filter_config.keys() if k.startswith(filter_name) and k != key]

                for param in param_keys:
                    if param in {"curvefit_model", "savgol_cut_signal_sequenz"}:
                        continue

                    

                    param_label = param.replace(f"{filter_name}_", "")

                    default_val = filter_config[param]
                    if isinstance(default_val, list) and len(default_val) == 1:
                        default_val = default_val[0] if default_val else 1.0
                        



                    if isinstance(default_val, (int, float)):
                        default_start = float(default_val) * 0.8
                        default_end = float(default_val) * 1.2
                        default_step = max(float(default_val) * 0.1, 0.1)
                    else:
                        default_start = default_end = default_step = None


                    print("Eingabe Optionen: (start,end,step | =val für konstant | - zum Beibehalten | \nText/Tupel: z. B. Tupel:[([40,0],[80,1])] oder Bereich) | " \
                        "Liste: z.B. Liste: 1,10,1 ; =2 -> [[1,2][2,2],... [10,2]]\n" \
                        "Boolean z.B. b: True,False\n" \
                        "Für curvefit_others: String Liste 'Liste-s:-> trennen durch komma und wenn mehrere ptionen als Kombination mit + zusammen führen\n(Listes: guess_p0+useoff -> ['guess_p0,useoff'] (1 Option))\n")
                    
                    if param =='notch_odd_sequence':
                        print("notch_odd_sequence entspricht der Auswahl jeder Zweiten Sequenz" \
                        "Muss nicht ungerade sein, kann auch gerade sein (je nach Einstellung)\n")


                    if param == "curvefit_others":
                        print("Bei REALEN DATEN wird das Verwenden von useoff STARK empfohlen\n" \
                        "Auswählen durch - Eingabe - Liste-s: useoff")

                    if param == "notch_odd_sequence":
                        print("Für Eingabe b: True eingeben, um nur True auszuwählen \n")


                    values = input_range(
                        f"  Bereich für '{param}' (alias {param_label})",
                        default_start=default_start,
                        default_end=default_end,
                        default_step=default_step,
                        default_val=default_val
                    )
                    print(f"values {values}")

                    if isinstance(values, str) and values == "KEEP_DEFAULT":
                        print(f"  ⏸ '{param}' bleibt konstant bei Defaultwert {default_val}")
                        continue
                    elif isinstance(values, (float, int)):
                        parameter_grid[param] = [values]
                        print(f"  ✅ '{param}' wird fest auf {values} gesetzt (nicht variiert)")
                    else:
                        parameter_grid[param] = values
                        total_combinations = estimate_total_combinations(parameter_grid)

                        if total_combinations > MAX_COMBINATIONS_FOR_VERBOSE_OUTPUT:
                            print(f"  🔁 '{param}' wird variiert – zu viele Kombinationen für detaillierte Anzeige ({total_combinations} gesamt)")
                        else:
                            print(f"  🔁 '{param}' wird variiert im Bereich {values}")
                apply_choice = input("Soll auch die Anwendung ohne Filter verwendet werden (j/n)? [n]: ").strip().lower()
                if apply_choice == "j":
                    parameter_grid[key] = [True, False]

    if not parameter_grid:
        print("⚠️ Keine aktiven Filter mit variablen Parametern gefunden.")
        return {
            key: [value] for key, value in default_filter_config.items()
            if not callable(value) and not key.startswith("curvefit_model")  # Funktionen ggf. ausschließen
        }

    return parameter_grid










# Datenverarbeitung

def generate_edit_data_parameter_grid(T_total, filter_configs, real_data=False):
    from copy import deepcopy

    column_filtering_parameter_grid=None
    column_filter_configs=[]

    def get_bool_grid_input(prompt, default=False):
        print("j,n -> Ture,False  - j -> True - n -> False")
        val = input(prompt).strip().lower()
        if val == "j,n" or val =="n,j":
            print("Variiert zwischen True und False")
            return [True, False]
        elif val == "n":
            print("auf False gesetzt")
            return [False]
        elif val =="j":
            print("auf True gesetzt")
            return [True]
        else:
            return [default]

    parameter_grid = {}

    edit_choice = input("Soll der Datensatz bearbeitet werden (j/n)? [n]: ").strip().lower() or "n"
    edit_data = edit_choice == "j"
    parameter_grid["edit_Data"] = [edit_data]

    if edit_data:
        
        parameter_grid["cut_Data"] = get_bool_grid_input("Soll Datensatz zugeschnitten werden (j/n)? [n]: ")

        if True in parameter_grid["cut_Data"]:
            try:
                start_input = input("Startpunkt (start,end,step) [0.0]: ").strip()
                end_input = input(f"Endpunkt (start,end,step) [{T_total}]: ").strip()
                if start_input and end_input:
                    if start_input.startswith("="):
                        parameter_grid["start_cut"] = [float(start_input[1:])]
                    else:
                        parameter_grid["start_cut"] = parse_number_range(start_input)

                    if end_input.startswith("="):
                        parameter_grid["end_cut"] = [float(end_input[1:])]
                    else:
                        parameter_grid["end_cut"] = parse_number_range(end_input)
                else:
                    parameter_grid["start_cut"] = [0.0]
                    parameter_grid["end_cut"] = [T_total]
            except:
                print("❌ Ungültiger Bereich – Standardwerte werden verwendet.")
                parameter_grid["start_cut"] = [0.0]
                parameter_grid["end_cut"] = [T_total]
        

        parameter_grid["pre_avg_sequenzen"] = get_bool_grid_input("Zwei benachbarte Sequenzen VOR Filterung mitteln? (j/n) [n]: ", default=False)
        parameter_grid["avg_sequenzen"] = get_bool_grid_input("Zwei benachbarte Sequenzen NACH Filterung mitteln? (j/n) [n]: ", default=False)

        if real_data:
            parameter_grid["colum_filtering"] = get_bool_grid_input("Spalten davor filtern? (j/n) [j]: ", default=False)
            

            # Falls eine Spaltenfilterung gewünscht ist, dann soll hier die paremert dazu defniert wqerden
            if True in parameter_grid.get("colum_filtering", []):
                
                column_filter_configs.append(ask_filter_settings())
                column_filtering_parameter_grid=generate_filter_parameter_grid(column_filter_configs)
            
            parameter_grid["cut_last_datapoint"] = get_bool_grid_input("Letzten Datenpunkt abschneiden? (j/n) [j]: ", default=True)
            parameter_grid["invert"] = get_bool_grid_input("Gegebene Daten invertieren? (j/n) [j]: ", default=True)
            parameter_grid["apply_remove_500Hz"] = get_bool_grid_input("500 Hz entfernen? (j/n) [j]: ", default=True)
            parameter_grid["remove_empty_measurement"] = get_bool_grid_input("Leermessung der MRX-Aparatur abziehen? (j/n) [n]: ", default=False)
        else:
            
            # Sicherstellen, dass alle Felder vorhanden sind
            parameter_grid.update({
                "colum_filtering": [False],
                "cut_last_datapoint": [True],
                "invert": [False],
                "apply_remove_500Hz": [True],
                "remove_empty_measurement": [False]
            })

    else:
        # Kein Edit → Fallbackwerte
        parameter_grid.update({
            "cut_Data": [False],
            "pre_avg_sequenzen": [False],
            "avg_sequenzen": [False],
            "colum_filtering": [False],
            "cut_last_datapoint": [True],
            "invert": [False],
            "apply_remove_500Hz": [True],
            "remove_empty_measurement": [False]
        })
        

    return parameter_grid,column_filtering_parameter_grid,column_filter_configs
