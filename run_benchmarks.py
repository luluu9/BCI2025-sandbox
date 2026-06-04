from moabb.datasets.utils import find_intersecting_channels
from utils import print_results_summary, run_moabb_benchmark, get_all_datasets, get_brainbot_datasets

import moabb
import mne

moabb.set_log_level('INFO')
mne.set_log_level('INFO')

SUBJECTS = 5
MAX_TRIALS = 1
INTERVALS = [[0, 3.5]]

datasets = get_brainbot_datasets(subjects=SUBJECTS, max_trials=MAX_TRIALS, brainbot_intervals=INTERVALS)
dataset_results = {}
dataset_events = ["left_hand", "right_hand", "feet", "rest"]
sampling = 256 

electrodes, datasets = find_intersecting_channels(datasets)
print("Datasets used:", [type(d).__name__ for d in datasets])
print("Used electrodes:", electrodes)

brainbot_datasets = get_brainbot_datasets(subjects=SUBJECTS, max_trials=MAX_TRIALS, brainbot_intervals=INTERVALS)

results = run_moabb_benchmark("pipelines_MI", datasets_list=brainbot_datasets, overwrite=True)
print_results_summary(results)

results = run_moabb_benchmark("pipelines_MI_tensorflow", datasets_list=brainbot_datasets, overwrite=True)
print_results_summary(results)