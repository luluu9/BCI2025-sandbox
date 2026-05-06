"""
Krzywe uczenia na datasetach MOABB: Weibo2014 i PhysionetMI.
4 klasy: left_hand, right_hand, feet, rest.
Filtracja 8-32 Hz wewnątrz CV (bez przecieku między foldami).

Każdy subject traktowany jest jako osobna kolumna wykresu.
Wykresy zapisywane do:
  learning_curve_Weibo2014.png
  learning_curve_PhysionetMI.png
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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, learning_curve, GridSearchCV
from sklearn.pipeline import Pipeline

# MOABB
sys.path.insert(0, str(Path(__file__).parent.parent / "moabb"))
from moabb.datasets import Weibo2014, PhysionetMI
from moabb.paradigms import MotorImagery

mne.set_log_level('ERROR')

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

# Wspólne 4 klasy (nazwy zgodne z MOABB event_id)
TARGET_EVENTS = ['left_hand', 'right_hand', 'feet', 'rest']

N_FOLDS = 5
N_JOBS  = 32

PIPELINE_COLORS = {
    "TSLR":        "#1f77b4",
    "TSSVM_grid":  "#2ca02c",
    "EN_grid":     "#d62728",
}

# Dataset-specific subject lists (None = wszystkie)
SUBJECTS_WEIBO    = None # list(range(1, 2))   # 10 subjects
SUBJECTS_PHYSIONET = list(range(1, 11))   # pierwsze 10, można rozszerzyć

# ---------------------------------------------------------------------------
# EpochBandpassFilter (filtracja wewnątrz CV — brak przecieku)
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


# ---------------------------------------------------------------------------
# Fabryki pipeline'ów
# ---------------------------------------------------------------------------
def make_tslr(sfreq=256.0):
    return Pipeline([
        ("bp",  EpochBandpassFilter(sfreq=sfreq)),
        ("cov", Covariances(estimator='oas')),
        ("ts",  TangentSpace(metric='riemann')),
        ("lr",  LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')),
    ])


def make_tssvm_grid(sfreq=256.0):
    pipe = Pipeline([
        ("bp",  EpochBandpassFilter(sfreq=sfreq)),
        ("cov", Covariances(estimator='oas')),
        ("ts",  TangentSpace(metric='riemann')),
        ("svc", SVC()),
    ])
    return GridSearchCV(pipe,
                        {'svc__C': [0.5, 1.0, 1.5], 'svc__kernel': ['rbf', 'linear']},
                        cv=3, scoring='accuracy', n_jobs=1, refit=True)


def make_en_grid(sfreq=256.0):
    pipe = Pipeline([
        ("bp",  EpochBandpassFilter(sfreq=sfreq)),
        ("cov", Covariances(estimator='oas')),
        ("ts",  TangentSpace(metric='riemann')),
        ("lr",  LogisticRegression(penalty='elasticnet', solver='saga',
                                   intercept_scaling=1000.0, max_iter=1000,
                                   l1_ratio=0.70)),
    ])
    return GridSearchCV(pipe,
                        {'lr__l1_ratio': [0.20, 0.30, 0.45, 0.65, 0.75]},
                        cv=3, scoring='accuracy', n_jobs=1, refit=True)


PIPELINES = {
    "TSLR":       make_tslr,
    # "TSSVM_grid": make_tssvm_grid,
    # "EN_grid":    make_en_grid,
}

# ---------------------------------------------------------------------------
# Ładowanie danych przez MOABB (bez filtrowania — filtrujemy w CV)
# ---------------------------------------------------------------------------
def load_moabb_dataset(dataset, subjects):
    """
    Zwraca dict: subject_id -> (X, y, sfreq)
    Używamy paradigm z fmin=None/fmax=None odpowiadającym brak filtrowania
    i return_epochs=True, by dostać surowe epoki.
    """
    # Paradigm bez filtrowania: fmin/fmax ustawiamy bardzo szeroko, żeby MOABB
    # nie robił własnego filtrowania; właściwą filtrację robimy w CV.
    paradigm = MotorImagery(
        events=TARGET_EVENTS,
        n_classes=len(TARGET_EVENTS),
        fmin=1.0,
        fmax=45.0,
        tmin=0.0,
        tmax=None,           # użyj całego okna zdefiniowanego przez dataset
        resample=None,
    )

    result = {}
    subjects_to_load = subjects if subjects is not None else dataset.subject_list

    for subj in subjects_to_load:
        print(f"  Ładowanie subject {subj}...", end=" ", flush=True)
        try:
            epochs_obj, labels, _ = paradigm.get_data(dataset, subjects=[subj],
                                                       return_epochs=True)
        except Exception as e:
            print(f"BŁĄD: {e}")
            continue

        sfreq = epochs_obj.info['sfreq']
        X = epochs_obj.get_data()
        le = LabelEncoder()
        y = le.fit_transform(labels)
        print(f"{len(X)} epok, sfreq={sfreq} Hz")
        result[subj] = (X, y, sfreq)

    return result


# ---------------------------------------------------------------------------
# Krzywa uczenia
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
# Rysowanie
# ---------------------------------------------------------------------------
def plot_and_save(all_results, dataset_name, n_classes, out_path):
    """
    all_results: dict subject_id -> dict pipe_name -> result | None
    """
    valid_subjects = sorted(
        s for s, r in all_results.items()
        if any(v is not None for v in r.values())
    )
    n_subj = len(valid_subjects)
    if n_subj == 0:
        print(f"[{dataset_name}] Brak wyników do wykreślenia.")
        return

    chance = 1.0 / n_classes
    fig, axes = plt.subplots(2, n_subj, figsize=(4 * n_subj, 8),
                              squeeze=False,
                              gridspec_kw=dict(hspace=0.50, wspace=0.30))

    for col, subj in enumerate(valid_subjects):
        ax_test  = axes[0][col]
        ax_train = axes[1][col]

        for pipe_name, result in all_results[subj].items():
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

        for ax, row_label in [
            (ax_test,  f"Sub {subj} — TEST"),
            (ax_train, f"Sub {subj} — TRAIN"),
        ]:
            ax.axhline(chance, color='gray', linestyle='--', linewidth=1,
                       label=f"chance={chance:.2f}")
            ax.set_title(row_label, fontsize=9)
            ax.set_xlabel("Próbki treningowe")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{dataset_name} — krzywe uczenia (8–32 Hz filtr w CV, nested GS)\n"
        f"4 klasy: {', '.join(TARGET_EVENTS)}  |  {N_FOLDS}-fold CV",
        fontsize=11, y=1.01,
    )
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Wykres zapisany: {out_path}")


# ---------------------------------------------------------------------------
# Główna funkcja per-dataset
# ---------------------------------------------------------------------------
def run_dataset(dataset, dataset_name, subjects):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}  (subjects: {subjects})")
    print(f"{'='*60}")

    subject_data = load_moabb_dataset(dataset, subjects)
    if not subject_data:
        print("Brak danych — pomijam.")
        return

    all_results = {}
    for subj, (X, y, sfreq) in subject_data.items():
        print(f"\n  Subject {subj}  ({len(X)} epok, {len(np.unique(y))} klas)")
        all_results[subj] = {}
        for pipe_name, pipe_factory in PIPELINES.items():
            estimator = pipe_factory(sfreq=sfreq)
            print(f"    [{pipe_name}] ", end="", flush=True)
            result = compute_learning_curve(estimator, X, y)
            if result is None:
                print("za mało próbek")
            else:
                print(f"max_acc={result['test_mean'].max():.3f}  "
                      f"n_train_max={result['train_sizes'][-1]}")
            all_results[subj][pipe_name] = result

    out_path = Path(__file__).parent / f"learning_curve_{dataset_name}.png"
    plot_and_save(all_results, dataset_name, n_classes=len(TARGET_EVENTS), out_path=out_path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_dataset(Weibo2014(),    "Weibo2014",    SUBJECTS_WEIBO)
    run_dataset(PhysionetMI(),  "PhysionetMI",  SUBJECTS_PHYSIONET)
