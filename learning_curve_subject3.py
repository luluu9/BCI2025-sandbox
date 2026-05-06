"""
Krzywe uczenia (learning curves) dla subject3 — każda sesja osobno.
Pokazuje jak accuracy TS-SVM zmienia się w zależności od liczby próbek treningowych.
Cel: wyznaczyć punkt stabilizacji.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne
from pathlib import Path

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline

mne.set_log_level('ERROR')

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
DATA_DIR   = Path("brainbot_data/5s-interval-nofiltering")
SUBJECT_ID = 3
TMAX       = 5.0       # interwał czasowy epoki
N_FOLDS    = 5
N_JOBS     = 16         # sklearn learning_curve

ALL_EVENTS = {'rest': 1, 'left_hand': 2, 'right_hand': 3, 'hands': 4, 'feet': 5}

CONFIGS = {
    "5cls":          ALL_EVENTS,
    "4cls_no_hands": {k: v for k, v in ALL_EVENTS.items() if k != 'hands'},
    "4cls_no_feet":  {k: v for k, v in ALL_EVENTS.items() if k != 'feet'},
}

CONFIG_COLORS = {
    "5cls":          "#1f77b4",   # niebieski
    "4cls_no_hands": "#2ca02c",   # zielony
    "4cls_no_feet":  "#ff7f0e",   # pomarańczowy
}

# ---------------------------------------------------------------------------
# EpochBandpassFilter (filtracja wewnątrz CV)
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


def make_ts_svm(sfreq=256.0):
    return Pipeline([
        ("bp",  EpochBandpassFilter(sfreq=sfreq)),
        ("cov", Covariances(estimator='lwf')),
        ("ts",  TangentSpace(metric='riemann')),
        ("svm", SVC(kernel='rbf', C=1.0, gamma='scale')),
    ])


# ---------------------------------------------------------------------------
# Wczytywanie — jedna sesja
# ---------------------------------------------------------------------------
def load_session(subject_id: int, session_id: int):
    pattern = re.compile(
        rf"subject{subject_id}_ses{session_id}_run(\d+)_.*\.epo\.fif"
    )
    files = sorted([f for f in os.listdir(DATA_DIR) if pattern.match(f)])
    if not files:
        return None, None
    epoch_list = [mne.read_epochs(str(DATA_DIR / f), preload=True, verbose=False)
                  for f in files]
    ref_t = epoch_list[0].info['dev_head_t']
    for ep in epoch_list[1:]:
        ep.info['dev_head_t'] = ref_t
    epochs = mne.concatenate_epochs(epoch_list, verbose=False)
    sfreq  = epochs.info['sfreq']
    return epochs, sfreq


def get_xy(epochs, events_filter, tmax):
    wanted = set(events_filter.values())
    mask   = np.isin(epochs.events[:, 2], list(wanted))
    cropped = epochs.copy().crop(tmin=0.0, tmax=tmax, verbose=False)
    X = cropped.get_data()[mask]
    inv = {v: k for k, v in events_filter.items()}
    y_str = np.array([inv[i] for i in epochs.events[mask, 2]])
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    return X, y, le.classes_


# ---------------------------------------------------------------------------
# Oblicz krzywe uczenia dla jednej sesji / jednej konfiguracji klas
# ---------------------------------------------------------------------------
def compute_learning_curve(X, y, sfreq, n_folds=N_FOLDS):
    n = len(X)
    min_cls_count = np.min(np.bincount(y))
    # min train size: tyle żeby każda klasa miała ≥2 próbki w CV
    min_train = max(n_folds * len(np.unique(y)), 10)
    if n < min_train + 5:
        return None

    # train_sizes jako bezwzględne liczby próbek
    max_train = int(n * (n_folds - 1) / n_folds)
    step = max(1, (max_train - min_train) // 15)
    train_sizes_abs = np.arange(min_train, max_train + 1, step)
    if len(train_sizes_abs) < 3:
        return None

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(
        make_ts_svm(sfreq=sfreq),
        X, y,
        train_sizes=train_sizes_abs,
        cv=cv,
        scoring='accuracy',
        n_jobs=N_JOBS,
        error_score='raise',
    )
    return {
        'train_sizes':  train_sizes,
        'test_mean':    test_scores.mean(axis=1),
        'test_std':     test_scores.std(axis=1),
        'train_mean':   train_scores.mean(axis=1),
        'train_std':    train_scores.std(axis=1),
    }


# ---------------------------------------------------------------------------
# Główna pętla
# ---------------------------------------------------------------------------
def run():
    # wykryj dostępne sesje subject3
    pattern  = re.compile(rf"subject{SUBJECT_ID}_ses(\d+)_run\d+_.*\.epo\.fif")
    session_ids = sorted(set(
        int(m.group(1)) for f in os.listdir(DATA_DIR)
        if (m := pattern.match(f))
    ))
    print(f"Subject {SUBJECT_ID} — sesje: {session_ids}")

    # --- zbierz wyniki ---
    all_results = {}   # all_results[ses_id][cfg_name] = curve_dict | None

    for ses_id in session_ids:
        print(f"\n--- Sesja {ses_id} ---")
        epochs, sfreq = load_session(SUBJECT_ID, ses_id)
        if epochs is None:
            print("  brak danych")
            continue
        all_results[ses_id] = {}

        for cfg_name, events_filter in CONFIGS.items():
            X, y, classes = get_xy(epochs, events_filter, TMAX)
            n_cls = len(np.unique(y))
            print(f"  {cfg_name}: {len(X)} epok, {n_cls} klas", end="  →  ")
            result = compute_learning_curve(X, y, sfreq)
            if result is None:
                print("za mało próbek")
            else:
                best = result['test_mean'].max()
                print(f"max acc={best:.3f}  (n_train_max={result['train_sizes'][-1]})")
            all_results[ses_id][cfg_name] = result

    # --- wykresy ---
    n_ses = len(all_results)
    fig = plt.figure(figsize=(6 * n_ses, 10))
    gs  = gridspec.GridSpec(2, n_ses, hspace=0.45, wspace=0.3)

    for col, ses_id in enumerate(sorted(all_results)):
        ax_test  = fig.add_subplot(gs[0, col])
        ax_train = fig.add_subplot(gs[1, col])

        n_total_epochs = {}   # do tytułu
        _, sfreq_tmp = load_session(SUBJECT_ID, ses_id)

        for cfg_name, result in all_results[ses_id].items():
            if result is None:
                continue
            color = CONFIG_COLORS[cfg_name]
            n_cls = len(CONFIGS[cfg_name])
            chance = 1 / n_cls
            xs = result['train_sizes']

            # test accuracy
            ax_test.plot(xs, result['test_mean'], color=color, label=cfg_name, linewidth=2)
            ax_test.fill_between(xs,
                result['test_mean'] - result['test_std'],
                result['test_mean'] + result['test_std'],
                alpha=0.15, color=color)
            # linia chance
            ax_test.axhline(chance, color=color, linestyle=':', linewidth=1, alpha=0.6)

            # train accuracy
            ax_train.plot(xs, result['train_mean'], color=color, label=cfg_name, linewidth=2)
            ax_train.fill_between(xs,
                result['train_mean'] - result['train_std'],
                result['train_mean'] + result['train_std'],
                alpha=0.15, color=color)

        # wspólna linia chance dla 5cls (dla referencji)
        ax_test.set_title(f"Sesja {ses_id} — TEST accuracy\n(linie przerywane = chance)", fontsize=10)
        ax_test.set_xlabel("Liczba próbek treningowych")
        ax_test.set_ylabel("Accuracy")
        ax_test.legend(fontsize=8)
        ax_test.set_ylim(0, 1.05)
        ax_test.grid(True, alpha=0.3)

        ax_train.set_title(f"Sesja {ses_id} — TRAIN accuracy", fontsize=10)
        ax_train.set_xlabel("Liczba próbek treningowych")
        ax_train.set_ylabel("Accuracy")
        ax_train.legend(fontsize=8)
        ax_train.set_ylim(0, 1.05)
        ax_train.grid(True, alpha=0.3)

    fig.suptitle(
        f"Subject {SUBJECT_ID} — krzywe uczenia TS-SVM (8–32 Hz, filtr w CV)\n"
        f"t=[0, {TMAX}]s, {N_FOLDS}-fold StratifiedKFold",
        fontsize=13, y=1.01
    )

    out_path = Path(__file__).parent / f"learning_curve_subject{SUBJECT_ID}.png"
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\nWykres zapisany: {out_path}")


if __name__ == "__main__":
    run()
