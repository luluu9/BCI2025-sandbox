import numpy as np
import mne

def split_epochs_into_segments(epochs, seg_length_s, step_s=None):
    sfreq = int(epochs.info['sfreq'])
    seg_samples = int(round(seg_length_s * sfreq))
    step_samples = seg_samples if step_s is None else int(round(step_s * sfreq))

    segments = []
    segment_labels = []
    classes = list(epochs.event_id.keys())

    for cls in classes:
        sub = epochs[cls]  # Epochs for this class
        data = sub.get_data()  # shape (n_epochs, n_ch, n_times)
        for ep_idx in range(data.shape[0]):
            n_times = data.shape[2]
            for start in range(0, n_times - seg_samples + 1, step_samples):
                seg = data[ep_idx, :, start:start + seg_samples]
                segments.append(seg)
                segment_labels.append(cls)

    if len(segments) == 0:
        raise ValueError("No segments produced: check seg_length_s and epoch length.")

    data_new = np.stack(segments)  # (n_new, n_ch, seg_samples)
    info = epochs.info.copy()

    # https://mne.tools/stable/documentation/glossary.html#term-events
    event_id_map = epochs.event_id
    print(event_id_map)
    events = np.c_[np.arange(len(data_new)), np.zeros(len(data_new), int),
                   np.array([event_id_map[l] for l in segment_labels], int)]
    new_epochs = mne.EpochsArray(data_new, info, events=events, event_id=event_id_map, tmin=0.0)
    
    # preserve montage if present
    montage = epochs.get_montage()
    if montage is not None:
        new_epochs.set_montage(montage)

    return new_epochs

def merge_epochs(*epochs):
    if not epochs:
        raise ValueError("no epochs provided")
    parts = []
    for e in epochs:
        if isinstance(e, (list, tuple)):
            parts.extend(e)
        else:
            parts.append(e)
    if len(parts) == 1:
        return parts[0].copy()
    return mne.concatenate_epochs(parts)