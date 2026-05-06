"""
Wpływ długości okna czasowego na accuracy TSLR.
Datasety: Zhou2016 (3 klasy) i BNCI2014_001 (4 klasy).

Oś X = długość okna [s] (TMIN_START..tmax_max co TMAX_STEP)
Filtracja: 8–32 Hz IIR wewnątrz CV.
Wynik: window_length_effect.png

Layout:
  Wiersze = datasety (Zhou2016, BNCI2014_001)
  Kolumny = subjecty
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
from pathlib import Path

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent / "moabb"))
from moabb.datasets import Zhou2016, BNCI2014_001
from moabb.paradigms import MotorImagery

mne.set_log_level('ERROR')

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
TMIN_START = 1.0
TMAX_STEP  = 0.5
N_FOLDS    = 5
N_JOBS     = 32

# (dataset_instance, name, events, subjects, tmax_max)
DATASETS = [
    (Zhou2016(),     "Zhou2016",
     ['left_hand', 'right_hand', 'feet'],
     [1, 2, 3, 4], 5.0),
    (BNCI2014_001(), "BNCI2014_001",
     ['left_hand', 'right_hand', 'feet', 'tongue'],
     list(range(1, 10)), 4.0),
]

# ---------------------------------------------------------------------------
# EpochBandpassFilter — 8–32 Hz IIR wewnątrz CV
# ---------------------------------------------------------------------------
class EpochBandpassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=256.0, l_freq=8.0, h_freq=32.0, method='iir'):
        self.sfreq  = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = np.empty_like(X)
        for i, epoch in enumerate(X):
            out[i] = mne.filter.filter_data(
                epoch.astype(np.float64),
                sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq,
                method=self.method, verbose=False,
            )
        return out


def make_tslr(sfreq=256.0):
    return Pipeline([
        ("bp",  EpochBandpassFilter(sfreq=sfreq)),
        ("cov",  Covariances(estimator='oas')),
        ("ts",   TangentSpace(metric='riemann')),
        ("lr",   LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')),
    ])


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Ładowanie subjecta przez MOABB (surowe — filtrujemy w CV)
# ---------------------------------------------------------------------------
def load_subject(dataset, events, subject, tmax_max):
    paradigm = MotorImagery(
        events=events,
        n_classes=len(events),
        fmin=1.0, fmax=45.0,
        tmin=0.0, tmax=tmax_max,
        resample=None,
    )
    try:
        epochs, labels, _ = paradigm.get_data(
            dataset, subjects=[subject], return_epochs=True
        )
    except Exception as e:
        print(f"  BŁĄD subject {subject}: {e}")
        return None, None, None

    sfreq = epochs.info['sfreq']
    X = epochs.get_data()
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return X, y, sfreq


# ---------------------------------------------------------------------------
# Krzywa accuracy vs. tmax
# ---------------------------------------------------------------------------
def compute_window_curve(X_full, y, sfreq, tmax_max):
    tmax_values = np.arange(TMIN_START, tmax_max + 0.01, TMAX_STEP)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    means, stds = [], []
    for tmax in tmax_values:
        n_samples = int(np.round(tmax * sfreq)) + 1
        X_crop = X_full[:, :, :n_samples]
        scores = cross_val_score(make_tslr(sfreq), X_crop, y,
                                 cv=cv, scoring='accuracy', n_jobs=N_JOBS)
        means.append(scores.mean())
        stds.append(scores.std())
        print(f"    tmax={tmax:.1f}s  acc={scores.mean():.3f} ± {scores.std():.3f}")
    return tmax_values, np.array(means), np.array(stds)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rows_data = []

    for dataset, ds_name, events, subjects, tmax_max in DATASETS:
        chance = 1.0 / len(events)
        print(f"\n{'='*60}")
        print(f"{ds_name}  |  {len(events)} klas: {events}  |  chance={chance:.2f}")
        print(f"{'='*60}")

        entries = []
        for subj in subjects:
            print(f"\n  Subject {subj}:")
            X_full, y, sfreq = load_subject(dataset, events, subj, tmax_max)
            if X_full is None:
                continue
            print(f"  {len(X_full)} epok, sfreq={sfreq} Hz")
            tv, m, s = compute_window_curve(X_full, y, sfreq, tmax_max)
            entries.append((f"Sub {subj}", tv, m, s))

        if entries:
            rows_data.append((ds_name, chance, entries))

    if not rows_data:
        print("Brak danych.")
        exit()

    n_rows = len(rows_data)
    n_cols = max(len(r[2]) for r in rows_data)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.8 * n_rows),
        squeeze=False,
        gridspec_kw=dict(hspace=0.55, wspace=0.35),
    )

    for row_idx, (ds_name, chance, entries) in enumerate(rows_data):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(entries):
                ax.set_visible(False)
                continue

            subj_label, tv, means, stds = entries[col_idx]
            ax.plot(tv, means, color="#1f77b4", linewidth=2,
                    marker='o', markersize=4)
            ax.fill_between(tv, means - stds, means + stds,
                            alpha=0.2, color="#1f77b4")
            ax.axhline(chance, color='gray', linestyle='--', linewidth=1,
                       label=f"chance={chance:.2f}")
            ax.set_title(f"{ds_name}\n{subj_label}", fontsize=9)
            ax.set_xlabel("Długość okna [s]")
            ax.set_ylabel("Test accuracy")
            ax.set_ylim(0, 1.05)
            ax.set_xticks(tv)
            ax.tick_params(axis='x', labelrotation=45, labelsize=7)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Wpływ długości okna — TSLR  |  10–30 Hz + notch 50 Hz (filtr w CV)\n"
        "Zhou2016: left_hand, right_hand, feet  "
        "|  BNCI2014_001: left_hand, right_hand, feet, tongue  "
        "|  8–32 Hz IIR",
        fontsize=11, y=1.01,
    )

    out_path = Path(__file__).parent / "window_length_effect.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWykres zapisany: {out_path}")
