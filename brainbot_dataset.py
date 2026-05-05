import os
import numpy as np
import mne
import moabb
import re

moabb.set_log_level('ERROR')
mne.set_log_level('ERROR')

class BrainBotDataset(moabb.datasets.base.BaseDataset):
    def __init__(self, data_dir, subjects, events, interval, data_names, sessions_per_subject):
        """
        Parameters
        ----------
        data_dir : str
            Path to the directory containing the data files.
        subjects : list
            List of subject identifiers.
        events : dict
            Dictionary mapping event names to event IDs.
        interval : list
            Time interval for epochs.
        data_names : dict
            Dictionary mapping subject identifiers to their corresponding data file names (can be multiple runs).
        sessions_per_subject : int
            Number of sessions per subject.
        """
        duration_ms = int((interval[1] - interval[0]) * 1000)
        dataset_code = f"BrainBot{duration_ms}ms"

        super().__init__(
            subjects=subjects,
            sessions_per_subject=sessions_per_subject,
            events=events,
            code=dataset_code,
            interval=interval,
            paradigm="imagery",
        )
        self.data_dir = data_dir
        self.data_names = data_names

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        base = self.data_dir
        filenames = self.data_names[subject]
        epochs_files = []
        for i, session in enumerate(filenames):
            if i >= self.n_sessions:
                break
            session_paths = [os.path.join(base, fname) for fname in session]
            epochs_files.append(session_paths)
        return epochs_files

    def _get_single_subject_data(self, subject):
        paths = self.data_path(subject)
        sessions = {"0": {}} # sessions->runs->raw
        for session_id, session in enumerate(paths):
            for i, path in enumerate(session):
                epochs = mne.read_epochs(path, preload=True)
                data = epochs.get_data()
                n_epochs, n_ch, n_times = data.shape
                data_flat = data.transpose(1, 0, 2).reshape(n_ch, n_epochs * n_times)

                # create stim channel for moabb compatibility
                stim = np.zeros(n_epochs * n_times, dtype=int)
                for j, ev in enumerate(epochs.events):
                    sample = j * n_times
                    stim[sample] = int(ev[2])
                data_with_stim = np.vstack([data_flat, stim])

                sfreq = epochs.info['sfreq']
                ch_names = epochs.ch_names + ['stim']
                ch_types = ['eeg'] * n_ch + ['stim']
                info = mne.create_info(ch_names, sfreq, ch_types)
                raw = mne.io.RawArray(data_with_stim, info)
                if str(session_id) not in sessions:
                    sessions[str(session_id)] = {}
                sessions[str(session_id)][str(i)] = raw
        
        return sessions

def get_brainbot_dataset(interval=[0.0, 2.5]):
    # moabb suggests to have sessions_per_subjects as min number of sessions across subjects
    # so this may lead to some issues, but for now assume max sessions to use all data
    # (currently no drawbacks of this approach seen)
    sessions_per_subject = len(max(files.values(), key=len))
    subjects_sorted = sorted(list(files.keys()))
    dataset = BrainBotDataset(
        data_dir=data_dir,
        subjects=subjects_sorted,
        events=loaded_events_id,
        interval=interval,
        data_names=files,
        sessions_per_subject=sessions_per_subject)
    return dataset


def generate_file_structure(path):
    # generates moabb compatible file structure for a given subject
    # format:
    # {
    #  subject_id: [
    #      [session1_run1_file, session1_run2_file, ...],
    #      [session2_run1_file, session2_run2_file, ...],
    #      ...
    #   ]
    # }
    file_struct = {}
    if not os.path.exists(path):
        return None

    # matches: subject3_ses1_run1_...
    pattern = re.compile(rf"subject(\d+)_ses(\d+)_run(\d+)_.*\.epo\.fif")
    
    sessions_map = {}
    
    for filename in os.listdir(path):
        match = pattern.match(filename)
        if match:
            subject_idx = int(match.group(1))
            ses_idx = int(match.group(2))
            run_idx = int(match.group(3))
            
            if subject_idx not in sessions_map:
                sessions_map[subject_idx] = {}
            if ses_idx not in sessions_map[subject_idx]:
                sessions_map[subject_idx][ses_idx] = []
            sessions_map[subject_idx][ses_idx].append((run_idx, filename))
    
    for subject_idx in sessions_map:
        # sort by session index, then by run index
        sorted_sessions = []
        for ses_idx in sorted(sessions_map[subject_idx].keys()):
            runs = sorted(sessions_map[subject_idx][ses_idx], key=lambda x: x[0])
            sorted_sessions.append([fname for _, fname in runs])
            
        file_struct[subject_idx] = sorted_sessions
    
    return file_struct


data_dir = r"brainbot_data/5s-interval-nofiltering/"
files = generate_file_structure(data_dir)

# assert that event ids are consistent across all subject files
previous_events_id = None
for subject_id in files:
    subject_sessions = files[subject_id]
    for session in subject_sessions:
        for file in session:
            loaded_events_id = mne.read_events(data_dir+file, return_event_id=True)[1]
            #print(loaded_events_id)
            if previous_events_id is not None:
                assert loaded_events_id == previous_events_id, f"Event IDs do not match in file {file}"
            previous_events_id = loaded_events_id

if __name__ == "__main__":
    dataset = get_brainbot_dataset()
    subject_data = dataset._get_single_subject_data(1)
    print(subject_data)
    for subject_id, subject_data in dataset.get_data().items():
        for session_id, session in subject_data.items():
            for run_id, raw in session.items():
                print(f"Subject {subject_id}, Session {session_id}, Run {run_id}")