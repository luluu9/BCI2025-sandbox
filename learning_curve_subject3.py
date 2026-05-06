"""
Krzywe uczenia (learning curves) dla subject3 — każda sesja osobno.
4 klasy: left_hand, right_hand, feet, rest.
Filtracja 8-32 Hz wewnątrz CV (bez przecieku między foldami).
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, learning_curve, GridSearchCV
from sklearn.pipeline import Pipeline

mne.set_log_level('ERROR')

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
DATA_DIR   = Path(__file__).parent / "brainbot_data/5s-interval-nofiltering"
SUBJECT_ID = 3
TMAX       = 5.0
N_FOLDS    = 5
N_JOBS     = 32

EVENTS = {'rest': 1, 'left_hand': 2, 'right_hand': 3, 'feet': 5}

PIPELINE_COLORS = {
    "TSLR":        "#1f77b4",  # niebieski
    "TSSVM_grid":  "#2ca02c",  # zielony
    "EN_grid":     "#d62728",  # czerwony
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


def make_tslr(sfreq=256.0):
    """Tangent Space + Logistic Regression (bez grid search)."""
    return Pipeline([
        ("bp",  EpochBandpassFilter(sfreq=sfreq)),
        ("cov", Covariances(estimator='oas')),
        ("ts",  TangentSpace(metric='riemann')),
        ("lr",  LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')),
    ])


def make_tssvm_grid(sfreq=256.0):
    """Tangent Space + SVM z nested grid search (C, kernel)."""
    pipe = Pipeline([
        ("bp",  EpochBandpassFilter(sfreq=sfreq)),
        ("cov", Covariances(estimator='oas')),
        ("ts",  TangentSpace(metric='riemann')),
        ("svc", SVC()),
    ])
    param_grid = {
        'svc__C':      [0.5, 1.0, 1.5],
        'svc__kernel': ['rbf', 'linear'],
    }
    return GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy',
                        n_jobs=1, refit=True)


def make_en_grid(sfreq=256.0):
    """Tangent Space + ElasticNet Logistic Regression z nested grid search."""
    pipe = Pipeline([
        ("bp",  EpochBandpassFilter(sfreq=sfreq)),
        ("cov", Covariances(estimator='oas')),
        ("ts",  TangentSpace(metric='riemann')),
        ("lr",  LogisticRegression(penalty='elasticnet', solver='saga',
                                   intercept_scaling=1000.0, max_iter=1000,
                                   l1_ratio=0.70)),
    ])
    param_grid = {
        'lr__l1_ratio': [0.20, 0.30, 0.45, 0.65, 0.75],
    }
    return GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy',
                        n_jobs=1, refit=True)


PIPELINES = {
    "TSLR":       make_tslr,
    "TSSVM_grid": make_tssvm_grid,
    "EN_grid":    make_en_grid,
}


# ---------------------------------------------------------------------------
# Wczytywanie — jedna sesja
# ---------------------------------------------------------------------------
def load_session(subject_id: int, session_id: int):
    pattern = re.compile(rf"subject{subject_id}_ses{session_id}_run(\d+)_.*\.epo\.fif")
    files = sorted([f for f in os.listdir(DATA_DIR) if pattern.match(f)])
    if not files:
        return None, None
    epoch_list = [mne.read_epochs(str(DATA_DIR / f), preload=True, verbose=False)
                  for f in files]
    ref_t = epoch_list[0].info['dev_head_t']
    for ep in epoch_list[1:]:
        ep.info['dev_head_t'] = ref_t
    epochs = mne.concatenate_epochs(epoch_list, verbose=False)
    return epochs, epochs.info['sfreq']


def get_xy(epochs, tmax):
    wanted = set(EVENTS.values())
    mask = np.isin(epochs.events[:, 2], list(wanted))
    cropped = epochs.copy().crop(tmin=0.0, tmax=tmax, verbose=False)
    X = cropped.get_data()[mask]
    inv = {v: k for k, v in EVENTS.items()}
    y_str = np.array([inv[i] for i in epochs.events[mask, 2]])
    le = LabelEncoder()
    return X, le.fit_transform(y_str)


# ---------------------------------------------------------------------------
# Krzywa uczenia (StratifiedKFold CV, nested dla grid search)
# ---------------------------------------------------------------------------
def compute_learning_curve(estimator, X, y):
    n_cls = len(np.unique(y))
    min_train = max(N_FOLDS * n_cls, 10)
    max_train = int(len(X) * (N_FOLDS - 1) / N_FOLDS)
    if max_train < min_train + 5:
        return None

    step = max(1, (max_train - min_train) // 15)
    train_sizes_abs = np.arange(min_train, max_train + 1, step)
    if len(train_sizes_abs) < 3:
        return None

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,
        train_sizes=train_sizes_abs,
        cv=cv, scoring='accuracy',
        n_jobs=N_JOBS, error_score='raise',
    )
    return {
        'train_sizes': train_sizes,
        'test_mean':   test_scores.mean(axis=1),
        'test_std':    test_scores.std(axis=1),
        'train_mean':  train_scores.mean(axis=1),
        'train_std':   train_scores.std(axis=1),
    }


# ---------------------------------------------------------------------------
# Główna pętla
# ---------------------------------------------------------------------------
def run():
    pattern = re.compile(rf"subject{SUBJECT_ID}_ses(\d+)_run\d+_.*\.epo\.fif")
    session_ids = sorted(set(
        int(m.group(1)) for f in os.listdir(DATA_DIR)
        if (m := pattern.match(f))
    ))
    chance = 1 / len(EVENTS)
    print(f"Subject {SUBJECT_ID} — sesje: {session_ids}")
    print(f"Klasy: {list(EVENTS.keys())}  chance={chance:.3f}")
    print(f"Pipeline'y: {list(PIPELINES.keys())}\n")

    # all_results[ses_id][pipeline_name] = result_dict | None
    all_results = {}

    for ses_id in session_ids:
        print(f"--- Sesja {ses_id} ---")
        epochs, sfreq = load_session(SUBJECT_ID, ses_id)
        if epochs is None:
            print("  brak danych")
            continue

        X, y = get_xy(epochs, TMAX)
        print(f"  {len(X)} epok, {len(np.unique(y))} klas")
        all_results[ses_id] = {}

        for pipe_name, pipe_factory in PIPELINES.items():
            estimator = pipe_factory(sfreq=sfreq)
            print(f"  [{pipe_name}] ", end="", flush=True)
            result = compute_learning_curve(estimator, X, y)
            if result is None:
                print("za mało próbek")
            else:
                print(f"max acc={result['test_mean'].max():.3f}  "
                      f"(n_train_max={result['train_sizes'][-1]})")
            all_results[ses_id][pipe_name] = result

    # --- wykresy: 2 wiersze (test / train), kolumny = sesje ---
    valid_sessions = [s for s, r in all_results.items()
                      if any(v is not None for v in r.values())]
    n_ses = len(valid_sessions)
    if n_ses == 0:
        print("Brak wyników do wykreślenia.")
        return

    fig, axes = plt.subplots(2, n_ses, figsize=(5 * n_ses, 8),
                              squeeze=False,
                              gridspec_kw=dict(hspace=0.45, wspace=0.3))

    for col, ses_id in enumerate(sorted(valid_sessions)):
        ax_test  = axes[0][col]
        ax_train = axes[1][col]

        for pipe_name, result in all_results[ses_id].items():
            if result is None:
                continue
            color = PIPELINE_COLORS[pipe_name]
            xs = result['train_sizes']

            for ax, key_m, key_s in [
                (ax_test,  'test_mean',  'test_std'),
                (ax_train, 'train_mean', 'train_std'),
            ]:
                m, s = result[key_m], result[key_s]
                ax.plot(xs, m, color=color, linewidth=2, label=pipe_name)
                ax.fill_between(xs, m - s, m + s, alpha=0.15, color=color)

        for ax, title in [
            (ax_test,  f"Sesja {ses_id} — TEST"),
            (ax_train, f"Sesja {ses_id} — TRAIN"),
        ]:
            ax.axhline(chance, color='gray', linestyle='--', linewidth=1,
                       label=f"chance={chance:.2f}")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Liczba próbek treningowych")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Subject {SUBJECT_ID} — krzywe uczenia (8–32 Hz, filtr w CV, nested GS)\n"
        f"4 klasy: {', '.join(EVENTS.keys())}  |  t=[0, {TMAX}]s  |  {N_FOLDS}-fold CV",
        fontsize=12, y=1.01
    )

    out_path = Path(__file__).parent / f"learning_curve_subject{SUBJECT_ID}.png"
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\nWykres zapisany: {out_path}")


if __name__ == "__main__":
    run()
