from moabb.datasets.utils import find_intersecting_channels
from datasets import PhysionetMI16, Weibo2014_16, Weibo2014_64_5_classes
from moabb.datasets import PhysionetMI
from utils import print_results_summary

import moabb
import mne
from moabb import benchmark
import os

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

def run_moabb_benchmark(pipelines_dir, base_dir="./benchmarks", datasets_list=datasets, n_jobs=1):
    pipelines_path = os.path.join(os.getcwd(), pipelines_dir)
    print(pipelines_path)

    cache_config = dict(
        use=True,
        save_raw=True,
        save_epochs=True,
        save_array=True,
        overwrite_raw=False,
        overwrite_epochs=False,
        overwrite_array=False,
    )

    print("Using cache dir:", mne.get_config('MNE_DATA'))

    return benchmark(
        pipelines=pipelines_path,
        evaluations=["WithinSession"],
        paradigms=["MotorImagery"],
        include_datasets=datasets_list,
        results=os.path.join(base_dir, f"results-{pipelines_dir}"),
        overwrite=False,
        plot=False,
        output=os.path.join(base_dir, f"output-{pipelines_dir}"),
        n_jobs=n_jobs,
        cache_config=cache_config
    )

results = run_moabb_benchmark("pipelines_MI")
print_results_summary(results)
