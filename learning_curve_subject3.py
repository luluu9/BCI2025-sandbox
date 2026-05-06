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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

mne.set_log_level('ERROR')

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
DATA_DIR   = Path("brainbot_data/5s-interval-nofiltering")
SUBJECT_ID = 3
TMAX       = 5.0       # interwał czasowy epoki
N_FOLDS    = 5
N_REPEATS  = 10    # ile razy losujemy podzbiór treningowy dla każdego rozmiaru
TEST_FRAC  = 0.20  # stały holdout testowy

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
#
# Schemat:
#   1. Stały holdout: 20% danych → test set (nigdy nie trenujemy na nim)
#   2. Pozostałe 80% → pula treningowa
#   3. Dla każdego train_size losujemy N_REPEATS razy podzbiór z puli
#      i oceniamy na stałym test set → mean ± std
# ---------------------------------------------------------------------------
def compute_learning_curve(X, y, sfreq):
    n = len(X)
    n_cls = len(np.unique(y))

    # stały podział train_pool / test (stratyfikowany)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=0)
    train_pool_idx, test_idx = next(sss.split(X, y))

    X_pool, y_pool = X[train_pool_idx], y[train_pool_idx]
    X_test, y_test = X[test_idx],       y[test_idx]

    n_pool = len(X_pool)
    # rozmiary treningu: od min_train do n_pool (15 kroków)
    min_train = max(n_cls * 4, 10)
    if n_pool < min_train + 3:
        return None

    step = max(1, (n_pool - min_train) // 14)
    train_sizes = np.arange(min_train, n_pool, step)  # n_pool excluded (sklearn wymaga < n_pool)
    if len(train_sizes) < 3:
        return None

    test_means, test_stds   = [], []
    train_means, train_stds = [], []

    for ts in train_sizes:
        t_scores, tr_scores = [], []
        rng = np.random.default_rng(42)
        for _ in range(N_REPEATS):
            sub_idx = rng.choice(n_pool, size=int(ts), replace=False)
            Xtr, ytr = X_pool[sub_idx], y_pool[sub_idx]
            clf = make_ts_svm(sfreq=sfreq)
            clf.fit(Xtr, ytr)
            t_scores.append(clf.score(X_test, y_test))
            tr_scores.append(clf.score(Xtr, ytr))

        test_means.append(np.mean(t_scores))
        test_stds.append(np.std(t_scores))
        train_means.append(np.mean(tr_scores))
        train_stds.append(np.std(tr_scores))

    return {
        'train_sizes':  train_sizes,
        'n_test':       len(test_idx),
        'test_mean':    np.array(test_means),
        'test_std':     np.array(test_stds),
        'train_mean':   np.array(train_means),
        'train_std':    np.array(train_stds),
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
                best = float('nan')
                n_test = '?'
            else:
                best = result['test_mean'].max()
                n_test = result['n_test']
            print(f"max acc={best:.3f}  (n_train_max={result['train_sizes'][-1] if result else '?'}, n_test={n_test})")
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

        # --- tytuł z info o test set ---
        n_test_info = ""
        for cfg_name, result in all_results[ses_id].items():
            if result is not None:
                n_test_info = f"  (test={result['n_test']} próbek, stały)"
                break

        ax_test.set_title(f"Sesja {ses_id} — TEST accuracy{n_test_info}\n(linie przerywane = chance)", fontsize=10)
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
        f"t=[0, {TMAX}]s  |  stały holdout test={int(TEST_FRAC*100)}%  |  {N_REPEATS} losowań/punkt",
        fontsize=13, y=1.01
    )

    out_path = Path(__file__).parent / f"learning_curve_subject{SUBJECT_ID}.png"
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\nWykres zapisany: {out_path}")


if __name__ == "__main__":
    run()
