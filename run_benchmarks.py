from moabb.datasets.utils import find_intersecting_channels
from utils import print_results_summary, run_moabb_benchmark, get_all_datasets, get_brainbot_datasets
from brainbot_dataset import loaded_events_id

import moabb
import mne
import pandas as pd

moabb.set_log_level('INFO')
mne.set_log_level('INFO')

SUBJECTS = 10
MAX_TRIALS = 4
INTERVALS = [[0, 4.0], [0, 4.5], [0, 5.0]]

# --- Event configurations ---
events_5cls = loaded_events_id  # left_hand, right_hand, feet, hands, rest
events_no_hands = {k: v for k, v in loaded_events_id.items() if k != "hands"}  # 4 klasy bez "both hands"
events_no_feet  = {k: v for k, v in loaded_events_id.items() if k != "feet"}   # 4 klasy bez "both feet"

print("5 classes:", list(events_5cls.keys()))
print("4 classes (no hands):", list(events_no_hands.keys()))
print("4 classes (no feet):", list(events_no_feet.keys()))

# --- Datasets ---
datasets_5cls      = get_brainbot_datasets(subjects=SUBJECTS, max_trials=MAX_TRIALS, brainbot_intervals=INTERVALS)
datasets_no_hands  = get_brainbot_datasets(subjects=SUBJECTS, max_trials=MAX_TRIALS, brainbot_intervals=INTERVALS,
                                           events=events_no_hands, code_suffix="-4cls_no_hands")
datasets_no_feet   = get_brainbot_datasets(subjects=SUBJECTS, max_trials=MAX_TRIALS, brainbot_intervals=INTERVALS,
                                           events=events_no_feet,  code_suffix="-4cls_no_feet")

# --- Run benchmarks ---
print("\n=== 5 klas (baseline) ===")
results_5cls = run_moabb_benchmark("pipelines_MI", datasets_list=datasets_5cls,
                                    base_dir="./benchmarks", overwrite=True)
print_results_summary(results_5cls)

print("\n=== 4 klasy - bez 'both hands' ===")
results_no_hands = run_moabb_benchmark("pipelines_MI", datasets_list=datasets_no_hands,
                                        base_dir="./benchmarks-4cls-no_hands", overwrite=True)
print_results_summary(results_no_hands)

print("\n=== 4 klasy - bez 'both feet' ===")
results_no_feet = run_moabb_benchmark("pipelines_MI", datasets_list=datasets_no_feet,
                                       base_dir="./benchmarks-4cls-no_feet", overwrite=True)
print_results_summary(results_no_feet)

# --- Porównanie zbiorcze ---
def compare_results(results_dict: dict):
    """Zestawia wyniki z różnych konfiguracji klas."""
    frames = []
    for label, df in results_dict.items():
        mean_df = df.groupby(['pipeline', 'dataset'])['score'].mean().reset_index()
        mean_df['config'] = label
        frames.append(mean_df)
    combined = pd.concat(frames, ignore_index=True)
    pivot = combined.pivot_table(index=['pipeline', 'dataset'], columns='config', values='score').round(3)
    return pivot

print("\n=== Porównanie konfiguracji klas ===")
comparison = compare_results({
    "5cls": results_5cls,
    "4cls_no_hands": results_no_hands,
    "4cls_no_feet": results_no_feet,
})
print(comparison.to_string())