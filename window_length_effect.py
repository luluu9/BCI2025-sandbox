"""
Wpływ długości okna czasowego na accuracy klasyfikatora TSLR.

Okna testowane: TMIN_START .. dataset_tmax co TMAX_STEP sekund.
Dla każdego okna liczymy cross-validated test accuracy (N_FOLDS-fold StratifiedKFold).
Filtracja 8-32 Hz wewnątrz CV (bez przecieku).

Layout wykresu:
  Wiersze  = datasety (BrainBot, Weibo2014, PhysionetMI)
  Kolumny  = sesje (BrainBot) lub subjecty (MOABB)
  Oś X     = długość okna [s]
  Oś Y     = test accuracy (mean ± std po foldach)

Wynik: window_length_effect.png
"""

import os
import re
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
from moabb.datasets import Weibo2014, PhysionetMI
from moabb.paradigms import MotorImagery

mne.set_log_level('ERROR')

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
TARGET_EVENTS   = ['left_hand', 'right_hand', 'feet', 'rest']
BRAINBOT_EVENTS = {'rest': 1, 'left_hand': 2, 'right_hand': 3, 'feet': 5}

TMIN_START = 1.0    # pierwsze okno: od 0 do 1 s
TMAX_STEP  = 0.5    # krok
N_FOLDS    = 5
N_JOBS     = 32

BRAINBOT_SUBJECT  = 3
BRAINBOT_DATA_DIR = Path(__file__).parent / "brainbot_data/5s-interval-nofiltering"
BRAINBOT_TMAX_MAX = 5.0

# MOABB datasety — (klasa, subjects, tmax_max)
# tmax_max = maksymalne okno sensowne dla danego datasetu
MOABB_DATASETS = [
    # (dataset_instance, name, subjects, tmax_max)
    (Weibo2014(),   "Weibo2014",   None,             4.0),   # okno [3,7] → 4 s
    (PhysionetMI(), "PhysionetMI", list(range(1, 11)), 3.0),  # okno [0,3] → 3 s
]

# ---------------------------------------------------------------------------
# Pipeline TSLR (Tangent Space + Logistic Regression)
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
        ("cov", Covariances(estimator='oas')),
        ("ts",  TangentSpace(metric='riemann')),
        ("lr",  LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')),
    ])


# ---------------------------------------------------------------------------
# BrainBot — ładowanie
# ---------------------------------------------------------------------------
def load_brainbot_session(subject_id, session_id):
    pattern = re.compile(
        rf"subject{subject_id}_ses{session_id}_run\d+_.*\.epo\.fif"
    )
    files = sorted(f for f in os.listdir(BRAINBOT_DATA_DIR) if pattern.match(f))
    if not files:
        return None, None

    epoch_list = [
        mne.read_epochs(str(BRAINBOT_DATA_DIR / f), preload=True, verbose=False)
        for f in files
    ]
    ref_t = epoch_list[0].info['dev_head_t']
    for ep in epoch_list[1:]:
        ep.info['dev_head_t'] = ref_t
    epochs = mne.concatenate_epochs(epoch_list, verbose=False)
    return epochs, epochs.info['sfreq']


def get_brainbot_sessions(subject_id):
    """Zwraca posortowaną listę ID sesji dostępnych dla danego subjecta."""
    pattern = re.compile(rf"subject{subject_id}_ses(\d+)_run\d+_.*\.epo\.fif")
    return sorted(set(
        int(m.group(1))
        for f in os.listdir(BRAINBOT_DATA_DIR)
        if (m := pattern.match(f))
    ))


def brainbot_get_xy(epochs, tmax):
    """Zwraca (X, y) dla 4 klas BrainBot, przycięte do tmax."""
    wanted = set(BRAINBOT_EVENTS.values())
    mask = np.isin(epochs.events[:, 2], list(wanted))
    cropped = epochs.copy().crop(tmin=0.0, tmax=tmax, verbose=False)
    X = cropped.get_data()[mask]
    inv = {v: k for k, v in BRAINBOT_EVENTS.items()}
    y_str = np.array([inv[i] for i in epochs.events[mask, 2]])
    le = LabelEncoder()
    return X, le.fit_transform(y_str)


# ---------------------------------------------------------------------------
# MOABB — ładowanie (surowe, filtracja w CV)
# ---------------------------------------------------------------------------
def load_moabb_subject(dataset, subject, tmax_max):
    """
    Zwraca (X_full, y, sfreq) — pełne epoki (do tmax_max).
    Używamy szerokiego pasma (1-45 Hz) żeby nie filtrować przed CV.
    """
    paradigm = MotorImagery(
        events=TARGET_EVENTS,
        n_classes=len(TARGET_EVENTS),
        fmin=1.0,
        fmax=45.0,
        tmin=0.0,
        tmax=tmax_max,
        resample=None,
    )
    try:
        epochs_obj, labels, _ = paradigm.get_data(
            dataset, subjects=[subject], return_epochs=True
        )
    except Exception as e:
        print(f"BŁĄD subject {subject}: {e}")
        return None, None, None

    sfreq = epochs_obj.info['sfreq']
    X = epochs_obj.get_data()
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return X, y, sfreq


def moabb_crop_x(X, sfreq, tmax):
    """Przycina X (n_epochs, n_ch, n_times) do tmax sekund."""
    n_samples = int(np.round(tmax * sfreq)) + 1
    return X[:, :, :n_samples]


# ---------------------------------------------------------------------------
# Obliczanie accuracy vs. tmax
# ---------------------------------------------------------------------------
def compute_window_curve(X_full, y, sfreq, tmax_values):
    """
    Dla każdego tmax w tmax_values liczy cross-val test accuracy.
    X_full: epoki pełnej długości (będą przycinane do każdego tmax).
    Zwraca (means, stds) — tablice tej samej długości co tmax_values.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    means, stds = [], []

    for tmax in tmax_values:
        X_crop = moabb_crop_x(X_full, sfreq, tmax)
        estimator = make_tslr(sfreq=sfreq)
        scores = cross_val_score(estimator, X_crop, y,
                                 cv=cv, scoring='accuracy', n_jobs=N_JOBS)
        means.append(scores.mean())
        stds.append(scores.std())
        print(f"    tmax={tmax:.1f}s  acc={scores.mean():.3f} ± {scores.std():.3f}")

    return np.array(means), np.array(stds)


# ---------------------------------------------------------------------------
# Zbieranie danych per-dataset
# ---------------------------------------------------------------------------
def collect_brainbot():
    """
    Zwraca list of (label, tmax_values, means, stds) — jedna pozycja na sesję.
    """
    sessions = get_brainbot_sessions(BRAINBOT_SUBJECT)
    tmax_values = np.arange(TMIN_START, BRAINBOT_TMAX_MAX + 0.01, TMAX_STEP)
    entries = []
    for ses_id in sessions:
        print(f"\n  [BrainBot sub{BRAINBOT_SUBJECT}] sesja {ses_id}")
        epochs, sfreq = load_brainbot_session(BRAINBOT_SUBJECT, ses_id)
        if epochs is None:
            print("    brak plików")
            continue
        X_full, y = brainbot_get_xy(epochs, BRAINBOT_TMAX_MAX)
        means, stds = compute_window_curve(X_full, y, sfreq, tmax_values)
        entries.append((f"Sesja {ses_id}", tmax_values, means, stds))
    return entries


def collect_moabb(dataset, dataset_name, subjects, tmax_max):
    """
    Zwraca list of (label, tmax_values, means, stds) — jedna pozycja na subject.
    """
    subjects_list = subjects if subjects is not None else dataset.subject_list
    tmax_values = np.arange(TMIN_START, tmax_max + 0.01, TMAX_STEP)
    entries = []
    for subj in subjects_list:
        print(f"\n  [{dataset_name}] subject {subj}")
        X_full, y, sfreq = load_moabb_subject(dataset, subj, tmax_max)
        if X_full is None:
            continue
        means, stds = compute_window_curve(X_full, y, sfreq, tmax_values)
        entries.append((f"Sub {subj}", tmax_values, means, stds))
    return entries


# ---------------------------------------------------------------------------
# Rysowanie
# ---------------------------------------------------------------------------
def plot_all(rows_data):
    """
    rows_data: list of (row_title, chance, entries)
      entries: list of (col_label, tmax_values, means, stds)
    """
    n_rows = len(rows_data)
    n_cols  = max(len(r[2]) for r in rows_data)  # max kolumn

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
        gridspec_kw=dict(hspace=0.55, wspace=0.35),
    )

    # ukryj nadmiarowe subploty
    for row_idx, (row_title, chance, entries) in enumerate(rows_data):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(entries):
                ax.set_visible(False)
                continue

            col_label, tmax_values, means, stds = entries[col_idx]
            ax.plot(tmax_values, means, color="#1f77b4", linewidth=2, marker='o',
                    markersize=4)
            ax.fill_between(tmax_values, means - stds, means + stds,
                            alpha=0.2, color="#1f77b4")
            ax.axhline(chance, color='gray', linestyle='--', linewidth=1,
                       label=f"chance={chance:.2f}")
            ax.set_title(f"{row_title}\n{col_label}", fontsize=9)
            ax.set_xlabel("Długość okna [s]")
            ax.set_ylabel("Test accuracy")
            ax.set_ylim(0, 1.05)
            ax.set_xticks(tmax_values)
            ax.tick_params(axis='x', labelrotation=45, labelsize=7)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Wpływ długości okna na accuracy — TSLR (8–32 Hz filtr w CV)\n"
        f"4 klasy: {', '.join(TARGET_EVENTS)}  |  {N_FOLDS}-fold CV",
        fontsize=12, y=1.01,
    )

    out_path = Path(__file__).parent / "window_length_effect.png"
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWykres zapisany: {out_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    chance_4cls = 1.0 / 4

    rows_data = []

    # --- BrainBot ---
    print("\n" + "="*60)
    print(f"BrainBot subject {BRAINBOT_SUBJECT}")
    print("="*60)
    bb_entries = collect_brainbot()
    if bb_entries:
        rows_data.append((f"BrainBot sub{BRAINBOT_SUBJECT}", chance_4cls, bb_entries))

    # --- MOABB ---
    for dataset, ds_name, subjects, tmax_max in MOABB_DATASETS:
        print("\n" + "="*60)
        print(f"Dataset: {ds_name}")
        print("="*60)
        entries = collect_moabb(dataset, ds_name, subjects, tmax_max)
        if entries:
            rows_data.append((ds_name, chance_4cls, entries))

    if rows_data:
        plot_all(rows_data)
    else:
        print("Brak danych do wykreślenia.")
