from unittest import case
import mne
from pathlib import Path
import pandas as pd
import numpy as np
import tools
from enum import Enum

class Subject(Enum):
    MATI = "mati"
    KONRAD = "konrad"
    HANIA = "hania"

class RecordingModes(Enum):
    MANUAL = 1
    LSL = 2

mappings = {
    Subject.MATI: { 
        'A1': 'Fp1',
        'A2': 'Fp2',
        'A3': 'F4',
        'A4': 'Fz',
        'A5': 'F3',
        'A6': 'T7',
        'A7': 'C3',
        'A8': 'Cz',
        'A9': 'C4',
        'A10': 'T8',
        'A11': 'P4',
        'A12': 'Pz',
        'A13': 'P3',
        'A14': 'O1',
        'A15': 'Oz',
        'A16': 'O2'},
    Subject.KONRAD: {
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
        },
    Subject.HANIA: {
        'A1': 'Fp1',
        'A2': 'Fp2',
        'A3': 'F4',
        'A4': 'Fz',
        'A5': 'F3',
        'A6': 'CP1', # modified
        'A7': 'C3', 
        'A8': 'Cz',
        'A9': 'C4',
        'A10': 'CP2', # modified
        'A11': 'P4',
        'A12': 'Pz',
        'A13': 'P3',
        'A14': 'FC1', # modified
        'A15': 'CPz', # modified // not sure if this is exactly correct
        'A16': 'FC2' # modified
    }
}

def annotate_bdf_files(file_paths):
    for path in file_paths:
        if "annotated" in path.name:
            print(f"Skipping already annotated file: {path}")
            continue
        print(f"Processing: {path}")
        raw = mne.io.read_raw_bdf(str(path), preload=False,
                                  stim_channel="Status", verbose=False)
        events = mne.find_events(raw, stim_channel="Status",
                                 shortest_event=1, consecutive=True)

        sfreq = raw.info.get("sfreq")
        df = pd.DataFrame(events, columns=["sample", "previous", "event_id"])
        df["onset_s"] = df["sample"] / sfreq
        mask = ((df["event_id"].to_numpy(np.uint32) & 0xFF00) >> 8).astype(int)
        map_key = {1: "R", 2: "LH", 4: "RH", 8: "BH", 16: "BF"}
        df["key"] = [map_key.get(m) for m in mask]
        df = df[df["key"].notna()][["onset_s", "key"]]
        df["duration_s"] = df["onset_s"].diff().shift(-1)
        print(df.head(10))

        annotations = mne.Annotations(onset=df["onset_s"].to_list(
        ), description=df["key"].to_list(), duration=df["duration_s"].to_list())
        raw.set_annotations(annotations)

        out_file = path.with_suffix("")
        out_file = out_file.parent / f"{out_file.name}_annotated.fif"
        raw.save(str(out_file), overwrite=True)
        print(f"Saved: {out_file}")

def split_annotated_into_segments(file_paths, segment_length_s=2.0, step_s=1.0, mode=RecordingModes.MANUAL):
    for data_path in file_paths:
        if ".bdf" in data_path.name:
            print(f"Skipping not annotated file: {data_path}")
            continue
        recording_name = Path(data_path).stem
        is_real_movement = "real" in data_path.name.lower()

        name_lower = data_path.name.lower()
        subject = next((s for s in Subject if s.value in name_lower))

        filepath = str(data_path)
        raw = mne.io.read_raw_fif(filepath, preload=True)
        eeg_channels = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]
        raw.pick(picks=eeg_channels)
        raw.resample(sfreq=250)
        raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
        raw.notch_filter(freqs=[50.0])
           
        raw.rename_channels(mappings[subject])
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        all_events, all_events_id = mne.events_from_annotations(raw)
        if mode == RecordingModes.LSL:
            all_events_id = {"relax": 1, "left_hand": 2, "right_hand": 3, "both_hands": 4, "both_feets": 5}

            reject_criteria = dict(
                eeg=80e-6,  # 80 µV
            ) 

            task_margin = 1.0 # event is when cue is shown
            task_duration = 5.0
            task_end = task_margin + task_duration
            epochs = mne.Epochs(
                raw=raw,
                events=all_events,
                event_id=all_events_id,
                baseline=None,
                tmin=task_margin,
                tmax=task_end,
                preload=True,
                reject=reject_criteria
            )

            all_epochs = tools.split_epochs_into_segments(epochs, segment_length_s, step_s)
        else:
            all_events_id = {'both_feets': 1, 'both_hands': 2, 'left_hand': 3, 'relax': 4, 'right_hand': 5}
            if 'relax' in all_events_id:
                relax_event_id = {'relax': all_events_id['relax']}
                del all_events_id['relax']
            
            task_margin = 1.0 # seconds, margin to be sure that we are focused on the task
            task_end = 9.0 # seconds, to have same length for all epochs
            epochs = mne.Epochs(
                raw=raw,
                events=all_events,
                event_id=all_events_id,
                baseline=None,
                tmin=task_margin,
                tmax=task_end,
                preload=True
            )

            task_end = 4.0
            relax_epochs = mne.Epochs(
                raw=raw,
                events=all_events,
                event_id=relax_event_id,
                baseline=None,
                tmin=task_margin,
                tmax=task_end,
                preload=True
            )

            splitted_epochs = tools.split_epochs_into_segments(epochs, segment_length_s, step_s)
            splitted_relax_epochs = tools.split_epochs_into_segments(relax_epochs, segment_length_s, step_s)

            all_epochs = tools.merge_epochs(splitted_epochs, splitted_relax_epochs)

        all_epochs_filename = f"{recording_name}_epochs_splitted_segment={segment_length_s}-step={step_s}-epo.fif"
        all_epochs.save(f"data/processed/{all_epochs_filename}", overwrite=True)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data/konrad-real_movement"
    print(f"Searching .bdf files in: {data_dir}")

    files = sorted(
        {p for pattern in ("*.bdf", "*.fif") for p in data_dir.rglob(pattern)},
        key=str
    )
    if not files:
        print(f"No .bdf/.fif files found in {data_dir}. Exiting.")
    else:
        options_string = (
            f"1. Annotate all .bdf files\n"
            f"2. Split epochs into segments (works on annotated files)\n"
            f"3. Exit\n"
            f"Select an option:"
        )
        mode = input(options_string).strip()
        if mode == "1":
            annotate_bdf_files(files)
        elif mode == "2":
            segment_length_default = 2.0
            step_default = 1.0
            mode_default = RecordingModes.MANUAL.value
            
            s = input("Enter segment length in seconds [default 2.0]: ").strip()
            segment_length = float(s) if s else segment_length_default
            s = input("Enter step size in seconds [default 1.0]: ").strip()
            step = float(s) if s else step_default
            s = input("Enter mode (1 - manual recording, 2 - LSL recording) [default 1]: ").strip()
            mode = RecordingModes(int(s)) if s else RecordingModes.MANUAL
            split_annotated_into_segments(files, segment_length_s=segment_length, step_s=step, mode=mode)
        else:
            print("Exiting.")
