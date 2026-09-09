"""
Krzywe uczenia potoku TS+LR — dokładność klasyfikacji w funkcji liczby próbek
treningowych, zagregowana po uczestnikach/sesjach każdego datasetu.

Datasety (każdy z własnym, natywnym zestawem klas):
  - BrainBot (subject 3, wszystkie sesje) — lewa/prawa ręka, obie nogi, spoczynek
  - Weibo2014                              — lewa/prawa ręka, obie nogi, spoczynek
  - PhysionetMI                            — lewa/prawa ręka, obie nogi, spoczynek
  - Zhou2016                               — lewa/prawa ręka, obie nogi (bez spoczynku)
  - BNCI2014_001                           — lewa/prawa ręka, obie nogi, język (bez spoczynku)

Filtracja 8–32 Hz IIR wewnątrz CV (bez przecieku danych).
Wyniki cache'owane per (dataset, subject/sesja) w learning_curve_cache/,
dzięki czemu można przegenerować sam wykres bez ponownego treningu.

Wynik: ../paper/img/5_liczba_probek.png
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
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent / "moabb"))
from moabb.datasets import Weibo2014, PhysionetMI, Zhou2016, BNCI2014_001
from moabb.paradigms import MotorImagery

mne.set_log_level('ERROR')

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
N_FOLDS  = 5
N_JOBS   = 32
N_POINTS = 15  # liczba punktów na krzywej uczenia

# Katalog z zapisanymi wynikami per (dataset, subject/sesja) — pozwala
# przegenerować sam wykres bez ponownego trenowania modeli.
CACHE_DIR = Path(__file__).parent / "learning_curve_cache"

BRAINBOT_SUBJECT  = 3
BRAINBOT_DATA_DIR = Path(__file__).parent / "brainbot_data/5s-interval-nofiltering"
BRAINBOT_TMAX     = 5.0
BRAINBOT_EVENTS   = {'rest': 1, 'left_hand': 2, 'right_hand': 3, 'feet': 5}

# (dataset_instance, name, events, subjects, tmax_max)
MOABB_DATASETS = [
    (Weibo2014(),    "Weibo2014",
     ['left_hand', 'right_hand', 'feet', 'rest'],
     None, 4.0),
    (PhysionetMI(),  "PhysionetMI",
     ['left_hand', 'right_hand', 'feet', 'rest'],
     list(range(1, 11)), 3.0),
    (Zhou2016(),     "Zhou2016",
     ['left_hand', 'right_hand', 'feet'],
     [1, 2, 3, 4], 5.0),
    (BNCI2014_001(), "BNCI2014_001",
     ['left_hand', 'right_hand', 'feet', 'tongue'],
     list(range(1, 10)), 4.0),
]


# ---------------------------------------------------------------------------
# Pipeline TS+LR — filtracja 8–32 Hz IIR wewnątrz CV (bez przecieku)
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
def load_brainbot_session(session_id):
    pattern = re.compile(
        rf"subject{BRAINBOT_SUBJECT}_ses{session_id}_run\d+_.*\.epo\.fif"
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


def get_brainbot_sessions():
    """Zwraca posortowaną listę ID sesji dostępnych dla BRAINBOT_SUBJECT."""
    pattern = re.compile(rf"subject{BRAINBOT_SUBJECT}_ses(\d+)_run\d+_.*\.epo\.fif")
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
def load_moabb_subject(dataset, events, subject, tmax_max):
    """Zwraca (X, y, sfreq); szerokie pasmo (1-45 Hz), filtrujemy w CV."""
    paradigm = MotorImagery(
        events=events,
        n_classes=len(events),
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
        print(f"  BŁĄD subject {subject}: {e}")
        return None, None, None

    sfreq = epochs_obj.info['sfreq']
    X = epochs_obj.get_data()
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return X, y, sfreq


# ---------------------------------------------------------------------------
# Cache wyników per (dataset, subject/sesja) na dysku
# ---------------------------------------------------------------------------
def cache_path(ds_name, unit_id):
    return CACHE_DIR / f"{ds_name}_{unit_id}.npz"


def load_cached_curve(ds_name, unit_id):
    path = cache_path(ds_name, unit_id)
    if not path.exists():
        return None
    data = np.load(path)
    return data["train_sizes"], data["test_mean"], data["test_std"]


def save_cached_curve(ds_name, unit_id, train_sizes, test_mean, test_std):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path(ds_name, unit_id),
             train_sizes=train_sizes, test_mean=test_mean, test_std=test_std)


# ---------------------------------------------------------------------------
# Krzywa uczenia (StratifiedKFold CV)
# ---------------------------------------------------------------------------
def compute_learning_curve(X, y, sfreq):
    n_cls = len(np.unique(y))
    min_train = max(N_FOLDS * n_cls, 10)
    max_train = int(len(X) * (N_FOLDS - 1) / N_FOLDS)
    if max_train < min_train + 5:
        return None

    step = max(1, (max_train - min_train) // N_POINTS)
    train_sizes_abs = np.arange(min_train, max_train + 1, step)
    if len(train_sizes_abs) < 3:
        return None

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    train_sizes, _, test_scores = learning_curve(
        make_tslr(sfreq=sfreq), X, y,
        train_sizes=train_sizes_abs,
        cv=cv, scoring='accuracy',
        n_jobs=N_JOBS, error_score='raise',
    )
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    for n, m, s in zip(train_sizes, test_mean, test_std):
        print(f"    n_train={n:4d}  acc={m:.3f} ± {s:.3f}")
    return train_sizes, test_mean, test_std


# ---------------------------------------------------------------------------
# Zbieranie danych per-dataset (z użyciem cache)
# ---------------------------------------------------------------------------
def collect_brainbot():
    """Zwraca listę (label, train_sizes, test_mean, test_std) — jedna pozycja na sesję."""
    sessions = get_brainbot_sessions()
    entries = []
    for ses_id in sessions:
        unit_id = f"ses{ses_id}"
        cached = load_cached_curve("BrainBot", unit_id)
        if cached is not None:
            tv, m, s = cached
            print(f"  [BrainBot] sesja {ses_id}: wczytano z cache")
            entries.append((f"Sesja {ses_id}", tv, m, s))
            continue

        print(f"  [BrainBot] sesja {ses_id}: trening...")
        epochs, sfreq = load_brainbot_session(ses_id)
        if epochs is None:
            print("    brak plików")
            continue
        X, y = brainbot_get_xy(epochs, BRAINBOT_TMAX)
        result = compute_learning_curve(X, y, sfreq)
        if result is None:
            print("    za mało próbek")
            continue
        tv, m, s = result
        save_cached_curve("BrainBot", unit_id, tv, m, s)
        entries.append((f"Sesja {ses_id}", tv, m, s))
    return entries


def collect_moabb(dataset, ds_name, events, subjects, tmax_max):
    """Zwraca listę (label, train_sizes, test_mean, test_std) — jedna pozycja na subject."""
    subjects_list = subjects if subjects is not None else dataset.subject_list
    entries = []
    for subj in subjects_list:
        cached = load_cached_curve(ds_name, subj)
        if cached is not None:
            tv, m, s = cached
            print(f"  [{ds_name}] subject {subj}: wczytano z cache")
            entries.append((f"Sub {subj}", tv, m, s))
            continue

        print(f"  [{ds_name}] subject {subj}: trening...")
        X, y, sfreq = load_moabb_subject(dataset, events, subj, tmax_max)
        if X is None:
            continue
        print(f"    {len(X)} epok, sfreq={sfreq} Hz")
        result = compute_learning_curve(X, y, sfreq)
        if result is None:
            print("    za mało próbek")
            continue
        tv, m, s = result
        save_cached_curve(ds_name, subj, tv, m, s)
        entries.append((f"Sub {subj}", tv, m, s))
    return entries


# ---------------------------------------------------------------------------
# Agregacja krzywych o różnej długości (interpolacja na wspólną siatkę)
# ---------------------------------------------------------------------------
def aggregate_entries(entries):
    common_min = max(tv[0] for _, tv, _, _ in entries)
    common_max = min(tv[-1] for _, tv, _, _ in entries)
    if common_max <= common_min:
        common_min = min(tv[0] for _, tv, _, _ in entries)
        common_max = max(tv[-1] for _, tv, _, _ in entries)

    grid = np.linspace(common_min, common_max, N_POINTS)
    curves = np.vstack([np.interp(grid, tv, m) for _, tv, m, _ in entries])
    agg_mean = curves.mean(axis=0)
    agg_std = curves.std(axis=0, ddof=1) if len(entries) > 1 else np.zeros_like(agg_mean)
    return grid, agg_mean, agg_std


# ---------------------------------------------------------------------------
# Rysowanie — agregacja uczestników/sesji w jeden wykres na dataset
# ---------------------------------------------------------------------------
def plot_all(rows_data):
    """
    rows_data: list of (ds_name, chance, entries)
      entries: list of (label, train_sizes, test_mean, test_std)
    """
    n_datasets = len(rows_data)
    n_cols = 2
    n_rows = int(np.ceil(n_datasets / n_cols))
    is_last_row_alone = n_datasets % n_cols == 1

    fig = plt.figure(figsize=(6.0 * n_cols, 5.0 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols * 2)

    axes_list = []
    for i in range(n_datasets):
        row, col = divmod(i, n_cols)
        if is_last_row_alone and row == n_rows - 1:
            ax = fig.add_subplot(gs[row, 1:3])
        else:
            ax = fig.add_subplot(gs[row, col * 2:col * 2 + 2])
        axes_list.append(ax)

    for ax, (ds_name, chance, entries) in zip(axes_list, rows_data):
        for _, tv, m, _ in entries:
            ax.plot(tv, m, color="gray", linewidth=1, alpha=0.55, zorder=1)

        grid, agg_mean, agg_std = aggregate_entries(entries)
        ax.plot(grid, agg_mean, color="#1f77b4", linewidth=2.5,
                marker='o', markersize=5, zorder=3, label="Średnia")
        ax.fill_between(grid, agg_mean - agg_std, agg_mean + agg_std,
                        alpha=0.25, color="#1f77b4", zorder=2,
                        label="Odch. std.")
        ax.axhline(chance, color='black', linestyle='--', linewidth=1,
                   label=f"Poziom losowy ({chance:.2f})")

        ax.set_title(f"{ds_name}\nn={len(entries)}", fontsize=10)
        ax.set_xlabel("Liczba próbek treningowych")
        ax.set_ylabel("Dokładność klasyfikacji")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        gray_line = plt.Line2D([0], [0], color="gray", linewidth=1, alpha=0.6)
        ax.legend(handles + [gray_line], labels + ["Pojedynczy uczestnik/sesja"],
                  fontsize=7, loc="lower right")

    fig.suptitle(
        "Krzywe uczenia potoku TS+LR w funkcji liczby próbek treningowych",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path(__file__).parent.parent / "paper" / "img" / "5_liczba_probek.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWykres zapisany: {out_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rows_data = []

    # --- BrainBot ---
    print("\n" + "="*60)
    print(f"BrainBot subject {BRAINBOT_SUBJECT}")
    print("="*60)
    bb_entries = collect_brainbot()
    if bb_entries:
        rows_data.append(("BrainBot", 1.0 / len(BRAINBOT_EVENTS), bb_entries))

    # --- MOABB ---
    for dataset, ds_name, events, subjects, tmax_max in MOABB_DATASETS:
        print("\n" + "="*60)
        print(f"Dataset: {ds_name}")
        print("="*60)
        entries = collect_moabb(dataset, ds_name, events, subjects, tmax_max)
        if entries:
            rows_data.append((ds_name, 1.0 / len(events), entries))

    if rows_data:
        plot_all(rows_data)
    else:
        print("Brak danych do wykreślenia.")
