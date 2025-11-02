import mne
from pathlib import Path
import pandas as pd
import numpy as np

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
print(f"Searching .bdf files in: {data_dir}")

files = list(data_dir.rglob("*.bdf"))
if not files:
    print("No .bdf files found. Exiting.")
else:
    for path in files:
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
