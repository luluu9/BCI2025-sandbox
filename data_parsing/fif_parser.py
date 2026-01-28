import mne
from pathlib import Path
import numpy as np
import tools

mapping = {
    'A1': 'Cz',
    'A2': 'FCz',
    'A3': 'CP1',
    'A4': 'FC1',
    'A5': 'C1',
    'A6': 'CP3',
    'A7': 'C3',
    'A8': 'FC3',
    'A9': 'C4',
    'A10': 'FC4',
    'A11': 'Pz',
    'A12': 'CP2',
    'A13': 'CP4',
    'A14': 'C2',    
    'A15': 'CPz',
    'A16': 'FC2'
}

USE_ONLY_REAL_EVENTS = True # if False, use all event types (including classification result)

events_real = {"rest": 1, "left_hand": 2, "right_hand": 3, "hands": 4, "feet": 5}
events_predicted = {"rest_predicted": 11, "left_hand_predicted": 12, "right_hand_predicted": 13, "hands_predicted": 14, "feet_predicted": 15}
classification_result = {"correct": 20, "incorrect": 21}
all_possible_events_id = {**events_real, **events_predicted, **classification_result}

def split_annotated_into_segments(file_paths, segment_length_s, step_s, output_dir):
    for data_path in file_paths:
        recording_name = Path(data_path).stem

        filepath = str(data_path)
        raw = mne.io.read_raw_fif(filepath, preload=True)
        eeg_channels = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]
        raw.pick(picks=eeg_channels)
        raw.resample(sfreq=256)
        raw.filter(l_freq=8.0, h_freq=32.0, fir_design='firwin')
        raw.notch_filter(freqs=[50.0])
           
        raw.rename_channels(mapping)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        # to make sure that event ids are consistent across recordings
        description_code_to_consistent_id = {str(v): v for v in all_possible_events_id.values()}
        all_events, all_events_id = mne.events_from_annotations(raw, event_id=description_code_to_consistent_id)

        # filter only events that are really in data (all_events_id)
        # all_events_id has string keys, so we need to convert, and event values is continously increasing
        all_events_id_renamed = {k: all_events_id[str(v)] for k, v in all_possible_events_id.items() if str(v) in all_events_id.keys()}

        # events cant happen concurrently, so remove one of them (currently drop exact classification and store only result (correct/incorrect))
        # probably we can merge it in MNE fashion ("[status]/[classification_result]"), but for now just drop
        for events_pred in events_predicted.keys():
            all_events_id_renamed.pop(events_pred, None)
        all_events = np.array([e for e in all_events if e[2] in all_events_id_renamed.values()])
        
        if USE_ONLY_REAL_EVENTS:
            for events_pred in classification_result.keys():
                all_events_id_renamed.pop(events_pred, None)
            all_events = np.array([e for e in all_events if e[2] in events_real.values()])

        reject_criteria = dict(
            eeg=80e-6,  # 80 µV
        ) 

        task_margin = 1.0 # event is when cue is shown
        task_duration = 5.0
        task_end = task_margin + task_duration
        epochs = mne.Epochs(
            raw=raw,
            events=all_events,
            event_id=all_events_id_renamed,
            baseline=None,
            tmin=task_margin,
            tmax=task_end,
            preload=True,
            reject=reject_criteria
        )

        all_epochs = tools.split_epochs_into_segments(epochs, segment_length_s, step_s)
        all_epochs_filename = f"{recording_name}_epochs_splitted_segment={segment_length_s}-step={step_s}-8-32Hz.epo.fif"
        all_epochs.save(f"{output_dir}/{all_epochs_filename}", overwrite=True)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "brainbot_data/recordings"
    print(f"Searching .fif files in: {data_dir}")

    files = sorted(data_dir.rglob("*.fif"))
    if not files:
        print(f"No .fif files found in {data_dir}.")
        print("You can place .fif files in the brainbot_data/recordings/ directory and rerun the script.")
        print("These files are too large to be included in the repository.")
    else:
        segment_length_default = 2.0
        step_default = 1.0
        output_dir_default = base_dir / "brainbot_data/processed"
        
        s = input(f"Enter segment length in seconds [default {segment_length_default}]: ").strip()
        segment_length = float(s) if s else segment_length_default
        s = input(f"Enter step size in seconds [default {step_default}]: ").strip()
        step = float(s) if s else step_default
        s = input(f"Enter output directory [default brainbot_data/processed]: ").strip()
        output_dir = Path(s) if s else output_dir_default
        s = input(f"Proceed with segment_length={segment_length}, step={step}, output_dir={output_dir}? (y/n) [default y]: ").strip().lower()
        output_dir.mkdir(parents=True, exist_ok=True)
        split_annotated_into_segments(files, segment_length_s=segment_length, step_s=step, output_dir=output_dir)