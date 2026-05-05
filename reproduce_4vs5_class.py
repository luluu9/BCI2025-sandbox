"""
Minimal reproducer: Tangent Space SVM, 5 klas vs 4 klasy (bez hands / bez feet)
Bez użycia MOABB - bezpośrednie wczytywanie .epo.fif + pyriemann + sklearn.

Cel: sprawdzić czy drop jakości w 4-klasowym wariancie to realny wynik
     czy artefakt pipelinów MOABB.
"""

import os
import re
import numpy as np
import mne
from pathlib import Path

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

mne.set_log_level('ERROR')


class EpochBandpassFilter(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer: filtruje każdą epokę niezależnie (unika przecieku
    przez odpowiedź impulsową filtra między fold-ami train/test).

    Każda epoka (n_ch, n_times) jest filtrowana osobno - to właściwy sposób:
    nie ma możliwości 'krwawienia' epoki treningowej do testowej.
    """

    def __init__(self, sfreq: float = 256.0, l_freq: float = 8.0,
                 h_freq: float = 32.0, method: str = 'iir'):
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.method = method

    def fit(self, X, y=None):
        # parametry filtru są stałe - nic do nauczenia
        return self

    def transform(self, X):
        # X: (n_epochs, n_channels, n_times)
        out = np.empty_like(X)
        for i, epoch in enumerate(X):
            out[i] = mne.filter.filter_data(
                epoch.astype(np.float64),
                sfreq=self.sfreq,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                method=self.method,
                verbose=False,
            )
        return out

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
# Dane surowe (niefiltrowane) - filtracja odbywa się wewnątrz pipeline CV
DATA_DIR = Path("brainbot_data/5s-interval-nofiltering")
INTERVALS = [4.0, 4.5, 5.0]   # sekundy (odpowiada BrainBot4000ms, 4500ms, 5000ms)
N_FOLDS   = 5                  # StratifiedKFold wewnątrz każdego subjecta

# Pełne mapowanie event_id (z pliku)
ALL_EVENTS = {'rest': 1, 'left_hand': 2, 'right_hand': 3, 'hands': 4, 'feet': 5}

CONFIGS = {
    "5cls":          ALL_EVENTS,
    "4cls_no_hands": {k: v for k, v in ALL_EVENTS.items() if k != 'hands'},
    "4cls_no_feet":  {k: v for k, v in ALL_EVENTS.items() if k != 'feet'},
}


# ---------------------------------------------------------------------------
# Wczytywanie danych
# ---------------------------------------------------------------------------
def load_subject_epochs(subject_id: int) -> mne.Epochs:
    """Wczytaj i połącz wszystkie pliki .epo.fif dla danego subjecta."""
    pattern = re.compile(rf"subject{subject_id}_ses(\d+)_run(\d+)_.*\.epo\.fif")
    files = sorted([f for f in os.listdir(DATA_DIR) if pattern.match(f)])
    if not files:
        return None
    epoch_list = [mne.read_epochs(str(DATA_DIR / f), preload=True, verbose=False)
                  for f in files]
    print(f"  Pliki ({len(files)}): " + ", ".join(f"{f.split('_ses')[1][:6]}" for f in files))
    # EEG: dev_head_t różni się między sesjami, możemy zignorować (nie MEG)
    ref_t = epoch_list[0].info['dev_head_t']
    for ep in epoch_list[1:]:
        ep.info['dev_head_t'] = ref_t
    all_epochs = mne.concatenate_epochs(epoch_list, verbose=False)
    return all_epochs


def get_xy(epochs: mne.Epochs, events_filter: dict, tmax: float):
    """
    Zwraca (X, y) dla zadanego zestawu klas i interwału.
    X: ndarray (n_epochs, n_channels, n_times)
    y: ndarray (n_epochs,) – string labels
    """
    # wybierz tylko interesujące klasy
    wanted_ids = set(events_filter.values())
    mask = np.isin(epochs.events[:, 2], list(wanted_ids))

    # crop do interwału [0, tmax]
    ep_cropped = epochs.copy().crop(tmin=0.0, tmax=tmax, verbose=False)

    X = ep_cropped.get_data()[mask]  # (n, ch, time)
    y_ids = epochs.events[mask, 2]

    # zamień ID na nazwy klas
    inv_map = {v: k for k, v in events_filter.items()}
    y = np.array([inv_map[i] for i in y_ids])
    return X, y


# ---------------------------------------------------------------------------
# Pipeline: TS-SVM
# ---------------------------------------------------------------------------
def make_ts_svm(sfreq: float = 256.0, filter_inside_cv: bool = True):
    """
    filter_inside_cv=True  → filtracja 8-32 Hz wewnątrz CV (train i test osobno)
    filter_inside_cv=False → zakłada wstępnie przefiltrowane dane
    """
    steps = []
    if filter_inside_cv:
        steps.append(("bandpass", EpochBandpassFilter(sfreq=sfreq, l_freq=8.0, h_freq=32.0)))
    steps += [
        ("cov", Covariances(estimator='lwf')),
        ("ts",  TangentSpace(metric='riemann')),
        ("svm", SVC(kernel='rbf', C=1.0, gamma='scale')),
    ]
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Główna pętla
# ---------------------------------------------------------------------------
def run():
    # Znajdź dostępnych subjectów
    all_files = os.listdir(DATA_DIR)
    subject_ids = sorted(set(
        int(m.group(1)) for f in all_files
        if (m := re.match(r"subject(\d+)_", f))
    ))
    print(f"Znalezione subjecty: {subject_ids}\n")

    # results[config] = list of rows; each row = [score_4.0, score_4.5, score_5.0]
    results = {cfg: [] for cfg in CONFIGS}

    for subject_id in subject_ids:
        print(f"--- Subject {subject_id} ---")
        epochs = load_subject_epochs(subject_id)
        if epochs is None:
            print(f"  Brak plików, pomijam.")
            continue
        sfreq = epochs.info['sfreq']  # przekazujemy do filtru wewnątrz pipeline

        for cfg_name, events_filter in CONFIGS.items():
            row = []
            for tmax in INTERVALS:
                X, y = get_xy(epochs, events_filter, tmax)
                if len(np.unique(y)) < 2 or len(X) < N_FOLDS:
                    row.append(float('nan'))
                    continue

                le = LabelEncoder()
                y_enc = le.fit_transform(y)

                cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
                scores = cross_val_score(
                    make_ts_svm(sfreq=sfreq, filter_inside_cv=True),
                    X, y_enc, cv=cv, scoring='accuracy', n_jobs=1
                )
                row.append(scores.mean())
            results[cfg_name].append(row)  # row = [score_4.0, score_4.5, score_5.0]
            print(f"  {cfg_name:18s}  "
                  + "  ".join(f"t={t:.1f}s: {v:.3f}" if not np.isnan(v) else f"t={t:.1f}s:  NaN "
                               for t, v in zip(INTERVALS, row)))

    # ---------------------------------------------------------------------------
    # Podsumowanie
    # ---------------------------------------------------------------------------
    n_subj = len(results[list(CONFIGS)[0]])
    print("\n" + "="*70)
    print(f"PODSUMOWANIE - średnia po {n_subj} subjectach")
    print("="*70)
    header = f"{'config':18s}  " + "  ".join(f"t={t:.1f}s" for t in INTERVALS) + "  | chance"
    print(header)
    print("-"*len(header))
    for cfg_name in CONFIGS:
        means = []
        for idx in range(len(INTERVALS)):
            vals = [row[idx] for row in results[cfg_name] if not np.isnan(row[idx])]
            means.append(np.mean(vals) if vals else float('nan'))
        n_cls = len(CONFIGS[cfg_name])
        chance = 1 / n_cls
        print(f"{cfg_name:18s}  " + "  ".join(f"{m:.3f}" if not np.isnan(m) else " NaN " for m in means)
              + f"  | {chance:.3f} ({n_cls} klas)")

    print()
    print(f"Filtracja: 8-32 Hz IIR, aplikowana niezależnie na każdy fold train/test")
    print(f"Dane: {DATA_DIR} (surowe, niefiltrowane)")

    # ---------------------------------------------------------------------------
    # Diagnoza: sprawdź rozkład klas i epok per config
    # ---------------------------------------------------------------------------
    print("\n" + "="*70)
    print("DIAGNOZA - liczba epok na klasę (subject 1, t=4.0s)")
    print("="*70)
    epochs1 = load_subject_epochs(1)
    if epochs1 is not None:
        for cfg_name, events_filter in CONFIGS.items():
            X, y = get_xy(epochs1, events_filter, 4.0)
            vals, counts = np.unique(y, return_counts=True)
            print(f"  {cfg_name}: {len(X)} epok → " + ", ".join(f"{v}:{c}" for v,c in zip(vals, counts)))


if __name__ == "__main__":
    run()
