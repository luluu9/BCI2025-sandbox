"""
Hipoteza A: czy mała liczba kanałów (16 jak w BrainBot) powoduje większą dewiację?

Dla Weibo2014 i PhysionetMI porównujemy dwa warunki:
  - FULL: wszystkie kanały datasetu (60 / 64)
  - REDUCED: tylko 16 kanałów pokrywających się z BrainBot

Dla każdego subjectu i warunku liczymy accuracy vs. długość okna (TSLR, 5-fold CV).
Wynik: channel_count_effect.png

Layout:
  Wiersz 0: Weibo2014 – FULL (60 ch)
  Wiersz 1: Weibo2014 – REDUCED (16 ch)
  Wiersz 2: PhysionetMI – FULL (64 ch)
  Wiersz 3: PhysionetMI – REDUCED (16 ch)
  Kolumny: subjecty
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
from moabb.datasets import Weibo2014, PhysionetMI
from moabb.paradigms import MotorImagery

mne.set_log_level('ERROR')

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
TARGET_EVENTS = ['left_hand', 'right_hand', 'feet', 'rest']

# 16 kanałów BrainBot — centralne/centroparietalne
BRAINBOT_CHANNELS = [
    'Cz', 'FCz', 'CP1', 'FC1', 'C1', 'CP3', 'C3', 'FC3',
    'C4', 'FC4', 'Pz', 'CP2', 'CP4', 'C2', 'CPz', 'FC2',
]

TMIN_START = 1.0
TMAX_STEP  = 0.5
N_FOLDS    = 5
N_JOBS     = 32

# (dataset_instance, name, subjects, tmax_max)
MOABB_DATASETS = [
    (Weibo2014(),   "Weibo2014",   list(range(1, 6)), 4.0),
    (PhysionetMI(), "PhysionetMI", list(range(1, 6)), 3.0),
]

# Kolory warunków
COLOR_FULL    = "#1f77b4"   # niebieski — pełna liczba kanałów
COLOR_REDUCED = "#d62728"   # czerwony  — 16 kanałów (jak BrainBot)

# ---------------------------------------------------------------------------
# Pipeline TSLR
# ---------------------------------------------------------------------------
class EpochBandpassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=256.0, l_freq=8.0, h_freq=32.0, method='iir'):
        self.sfreq = sfreq; self.l_freq = l_freq
        self.h_freq = h_freq; self.method = method

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
# Ładowanie jednego subjecta z MOABB
# ---------------------------------------------------------------------------
def load_subject(dataset, subject, tmax_max):
    """Zwraca (epochs_mne, sfreq) — pełne epoki do tmax_max, filtr 1-45 Hz."""
    paradigm = MotorImagery(
        events=TARGET_EVENTS,
        n_classes=len(TARGET_EVENTS),
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

    le = LabelEncoder()
    y = le.fit_transform(labels)
    return epochs, y, epochs.info['sfreq']


# ---------------------------------------------------------------------------
# Obliczanie krzywej accuracy vs. tmax dla danego X_full
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
        print(f"      tmax={tmax:.1f}s  acc={scores.mean():.3f} ± {scores.std():.3f}")
    return tmax_values, np.array(means), np.array(stds)


# ---------------------------------------------------------------------------
# Główna pętla per-dataset
# ---------------------------------------------------------------------------
def collect_dataset(dataset, dataset_name, subjects, tmax_max):
    """
    Zwraca (n_ch_full, n_ch_reduced, results_full, results_reduced)
    gdzie results_* = list of (subj_label, tmax_values, means, stds)
    """
    results_full    = []
    results_reduced = []
    n_ch_full = n_ch_reduced = None

    for subj in subjects:
        print(f"\n  [{dataset_name}] subject {subj}")
        epochs, y, sfreq = load_subject(dataset, subj, tmax_max)
        if epochs is None:
            continue

        # --- FULL ---
        print(f"    FULL ({len(epochs.ch_names)} ch):")
        n_ch_full = len(epochs.ch_names)
        X_full = epochs.get_data()
        tv, m, s = compute_window_curve(X_full, y, sfreq, tmax_max)
        results_full.append((f"Sub {subj}", tv, m, s))

        # --- REDUCED: przecięcie z kanałami BrainBot ---
        available = [ch for ch in BRAINBOT_CHANNELS if ch in epochs.ch_names]
        n_ch_reduced = len(available)
        print(f"    REDUCED ({n_ch_reduced} ch z {len(BRAINBOT_CHANNELS)}):")
        epochs_r = epochs.copy().pick_channels(available, ordered=False)
        X_reduced = epochs_r.get_data()
        tv, m, s = compute_window_curve(X_reduced, y, sfreq, tmax_max)
        results_reduced.append((f"Sub {subj}", tv, m, s))

    return n_ch_full, n_ch_reduced, results_full, results_reduced


# ---------------------------------------------------------------------------
# Rysowanie
# ---------------------------------------------------------------------------
def plot_dataset(dataset_name, n_ch_full, n_ch_reduced, results_full, results_reduced,
                 chance, ax_row_full, ax_row_reduced):
    """Wypełnia dwa rzędy osi: full i reduced."""
    for col_idx, (subj_label, tv, means, stds) in enumerate(results_full):
        ax = ax_row_full[col_idx]
        ax.plot(tv, means, color=COLOR_FULL, linewidth=2, marker='o', markersize=4)
        ax.fill_between(tv, means - stds, means + stds, alpha=0.2, color=COLOR_FULL)
        ax.axhline(chance, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f"{dataset_name} – {subj_label}\n"
                     f"FULL ({n_ch_full} ch)", fontsize=8)
        _style_ax(ax, tv)

    for col_idx, (subj_label, tv, means, stds) in enumerate(results_reduced):
        ax = ax_row_reduced[col_idx]
        ax.plot(tv, means, color=COLOR_REDUCED, linewidth=2, marker='o', markersize=4)
        ax.fill_between(tv, means - stds, means + stds, alpha=0.2, color=COLOR_REDUCED)
        ax.axhline(chance, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f"{dataset_name} – {subj_label}\n"
                     f"REDUCED ({n_ch_reduced} ch)", fontsize=8)
        _style_ax(ax, tv)

    # ukryj niewykorzystane kolumny
    n_subj = len(results_full)
    for ax_row in (ax_row_full, ax_row_reduced):
        for col_idx in range(n_subj, len(ax_row)):
            ax_row[col_idx].set_visible(False)


def _style_ax(ax, tmax_values):
    ax.set_xlabel("Okno [s]", fontsize=7)
    ax.set_ylabel("Test acc", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(tmax_values)
    ax.tick_params(axis='x', labelrotation=45, labelsize=6)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    chance = 1.0 / len(TARGET_EVENTS)

    # Zbierz dane
    dataset_results = []
    for dataset, ds_name, subjects, tmax_max in MOABB_DATASETS:
        print(f"\n{'='*60}\n{ds_name}\n{'='*60}")
        n_full, n_red, r_full, r_red = collect_dataset(
            dataset, ds_name, subjects, tmax_max
        )
        dataset_results.append((ds_name, n_full, n_red, r_full, r_red, tmax_max))

    if not dataset_results:
        print("Brak danych.")
        exit()

    # Oblicz wymiary wykresu
    n_cols = max(len(r[3]) for r in dataset_results)   # max subjects
    n_rows = 2 * len(dataset_results)                   # 2 rzędy per dataset (full+reduced)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.2 * n_rows),
        squeeze=False,
        gridspec_kw=dict(hspace=0.65, wspace=0.35),
    )

    for ds_idx, (ds_name, n_full, n_red, r_full, r_red, _) in enumerate(dataset_results):
        row_full    = axes[2 * ds_idx]
        row_reduced = axes[2 * ds_idx + 1]
        plot_dataset(ds_name, n_full, n_red, r_full, r_red,
                     chance, row_full, row_reduced)

    fig.suptitle(
        "Hipoteza A: wpływ liczby kanałów na accuracy i dewiację — TSLR\n"
        f"Niebieski = pełna liczba ch  |  Czerwony = 16 ch (jak BrainBot: centralne)\n"
        f"4 klasy: {', '.join(TARGET_EVENTS)}  |  {N_FOLDS}-fold CV  |  8–32 Hz filtr w CV",
        fontsize=11, y=1.01,
    )

    out_path = Path(__file__).parent / "channel_count_effect.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWykres zapisany: {out_path}")
