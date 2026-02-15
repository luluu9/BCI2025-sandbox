from moabb.datasets.utils import find_intersecting_channels
from datasets import PhysionetMI16, Weibo2014_16, Weibo2014_64_5_classes
from moabb.datasets import PhysionetMI
from utils import print_results_summary, run_moabb_benchmark

import moabb
import mne

moabb.set_log_level('INFO')
mne.set_log_level('INFO')

SUBJECTS = 10
MAX_TRIALS = 4


physionet_dataset = PhysionetMI()
physionet_dataset.subject_list = physionet_dataset.subject_list[:SUBJECTS]

physionetMI_16_dataset = PhysionetMI16()
physionetMI_16_dataset.subject_list = physionetMI_16_dataset.subject_list[:SUBJECTS]

weibo2014_16_dataset = Weibo2014_16()
weibo2014_16_dataset.subject_list = weibo2014_16_dataset.subject_list[:SUBJECTS]

weibo2014_dataset = Weibo2014_64_5_classes()
weibo2014_dataset.subject_list = weibo2014_dataset.subject_list[:SUBJECTS]

datasets = [weibo2014_16_dataset, physionetMI_16_dataset, weibo2014_dataset, physionet_dataset]
dataset_results = {}
dataset_events = ["left_hand", "right_hand", "feet", "hands", "rest"]
sampling = 160 # based on Physionet sampling rate 

electrodes, datasets = find_intersecting_channels(datasets)
print("Datasets used:", [type(d).__name__ for d in datasets])
print("Used electrodes:", electrodes)

results = run_moabb_benchmark("pipelines_MI")
print_results_summary(results)
