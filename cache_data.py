from moabb.paradigms import MotorImagery
from utils import get_all_datasets
import mne

datasets = get_all_datasets()
dataset_events = ["left_hand", "right_hand", "feet", "hands", "rest"]
sampling = 160 # based on Physionet sampling rate

print("Datasets used:", [type(d).__name__ for d in datasets])

paradigm = MotorImagery(n_classes=len(dataset_events), events=dataset_events, resample=sampling)
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
for dataset in datasets:
    _ = paradigm.get_data(dataset, dataset.subject_list, cache_config=cache_config)
    print(f"Cached data for dataset: {type(dataset).__name__}")