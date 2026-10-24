"""Calculate direct and minor-state-mediated transition rates per TR."""

from submission_config import RESULTS_DIR, BN_DIR
from pathlib import Path
import os
import pandas as pd
import glob

OUTPUT_DIR = RESULTS_DIR / 'state_transitions'

def analyze_brain_state_transitions(folder_path, output_filename="transition_analysis_results.csv"):
    state_mapping = {
        1: 'Minor_State_1',
        2: 'Major_State_1',
        3: 'Major_State_2',
        4: 'Major_State_1',
        5: 'Major_State_2',
        6: 'Minor_State_2'
    }

    major_states = {'Major_State_1', 'Major_State_2'}
    minor_states = {'Minor_State_1', 'Minor_State_2'}

    results_list = []

    if not os.path.isdir(folder_path):
        raise FileNotFoundError("State-sequence directory not found.")

    file_paths = sorted(glob.glob(os.path.join(folder_path, '*BN.csv')))

    if not file_paths:
        raise FileNotFoundError("No *BN.csv state sequences found.")

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        try:

            df = pd.read_csv(file_path, header=None)
            state_sequence = df.iloc[:, 0].tolist()

            brain_states = [state_mapping.get(state) for state in state_sequence]

            direct_transitions = 0
            indirect_transitions = 0
            total_trs = len(brain_states)

            if total_trs < 2:
                continue

            i = 0
            while i < total_trs - 1:
                current_state = brain_states[i]
                next_state = brain_states[i+1]

                if current_state in major_states and next_state in major_states and current_state != next_state:
                    direct_transitions += 1
                    i += 1

                elif current_state in major_states and next_state in minor_states:

                    j = i + 2
                    while j < total_trs:

                        if brain_states[j] in major_states:

                            if brain_states[j] != current_state:
                                indirect_transitions += 1

                            i = j
                            break

                        if brain_states[j] in minor_states:
                            j += 1
                        else:

                            i = j
                            break

                    else:
                        i = j

                else:

                    i += 1

            direct_freq = (direct_transitions / total_trs) if total_trs > 0 else 0
            indirect_freq = (indirect_transitions / total_trs) if total_trs > 0 else 0

            results_list.append({
                'filename': filename,
                'total_trs': total_trs,
                'direct_transitions_count': direct_transitions,
                'indirect_transitions_count': indirect_transitions,
                'direct_transition_rate_per_TR': direct_freq,
                'indirect_transition_rate_per_TR': indirect_freq
            })

        except Exception as exc:
            raise RuntimeError(f"Cannot process state sequence: {Path(file_path).name}") from exc

    if results_list:
        results_df = pd.DataFrame(results_list)
        output_path = str(OUTPUT_DIR / output_filename)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    else:
        raise ValueError("No valid transition sequences were available.")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BN_DIR.is_dir():
        raise FileNotFoundError(f"State-sequence directory not found: {BN_DIR}")
    analyze_brain_state_transitions(str(BN_DIR))
    print(f"Results saved to {OUTPUT_DIR}")
