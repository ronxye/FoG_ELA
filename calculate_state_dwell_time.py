"""Calculate dwell times in TR units."""

from submission_config import RESULTS_DIR, BN_DIR
from pathlib import Path
import pandas as pd
import os
import numpy as np
import glob

OUTPUT_DIR = RESULTS_DIR / 'state_dwell_time'

def calculate_dwell_time(states_series, target_state_value):
    if states_series.empty:
        return 0.0

    dwell_times = []
    current_run_length = 0

    for state in states_series:
        if state == target_state_value:
            current_run_length += 1
        else:
            if current_run_length > 0:
                dwell_times.append(current_run_length)
            current_run_length = 0

    if current_run_length > 0:
        dwell_times.append(current_run_length)

    if not dwell_times:
        return 0.0

    return np.mean(dwell_times)

def process_bn_files_for_dwell_time(input_directory, output_filename='state_dwell_times.csv'):
    results = []

    state_mapping = {
        1: 'Minor_State_1',
        2: 'Major_State_1',
        3: 'Major_State_2',
        4: 'Major_State_1',
        5: 'Major_State_2',
        6: 'Minor_State_2'
    }

    bn_file_paths = sorted(glob.glob(os.path.join(input_directory, '*_BN.csv')))
    if not bn_file_paths:
        raise FileNotFoundError("No *_BN.csv state sequences found.")

    for file_path in bn_file_paths:
        try:

            df_states = pd.read_csv(file_path, header=None)
            states = df_states.iloc[:, 0]

            if states.empty:
                continue

            dwell_b1 = calculate_dwell_time(states, 1)
            dwell_b2 = calculate_dwell_time(states, 2)
            dwell_b3 = calculate_dwell_time(states, 3)
            dwell_b4 = calculate_dwell_time(states, 4)
            dwell_b5 = calculate_dwell_time(states, 5)
            dwell_b6 = calculate_dwell_time(states, 6)

            combined_states = states.map(state_mapping).fillna('Other')

            dwell_major_state_1 = calculate_dwell_time(combined_states, 'Major_State_1')
            dwell_major_state_2 = calculate_dwell_time(combined_states, 'Major_State_2')
            dwell_minor_state_1 = calculate_dwell_time(combined_states, 'Minor_State_1')
            dwell_minor_state_2 = calculate_dwell_time(combined_states, 'Minor_State_2')

            subject_id = os.path.basename(file_path).replace('_BN.csv', '')

            results.append({
                'Subject_ID': subject_id,
                'Dwell_Time_B1': dwell_b1,
                'Dwell_Time_B2': dwell_b2,
                'Dwell_Time_B3': dwell_b3,
                'Dwell_Time_B4': dwell_b4,
                'Dwell_Time_B5': dwell_b5,
                'Dwell_Time_B6': dwell_b6,
                'Dwell_Time_Major_State_1': dwell_major_state_1,
                'Dwell_Time_Major_State_2': dwell_major_state_2,
                'Dwell_Time_Minor_State_1': dwell_minor_state_1,
                'Dwell_Time_Minor_State_2': dwell_minor_state_2
            })
        except Exception as exc:
            raise RuntimeError(f"Cannot process state sequence: {Path(file_path).name}") from exc

    results_df = pd.DataFrame(results)

    output_csv_path = str(OUTPUT_DIR / output_filename)
    results_df.to_csv(output_csv_path, index=False)

    return results_df


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BN_DIR.is_dir():
        raise FileNotFoundError(f"State-sequence directory not found: {BN_DIR}")
    process_bn_files_for_dwell_time(str(BN_DIR))
    print(f"Results saved to {OUTPUT_DIR}")
