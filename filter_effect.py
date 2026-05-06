"""
Hipoteza B: czy zmiana filtrowania zmniejsza dewiację na danych BrainBot?

Testowane warunki filtrowania:
  1. 8–32 Hz IIR        (baseline, obecny setup)
  2. 4–40 Hz IIR        (szersze pasmo)
  3. 8–32 Hz FIR        (inny typ filtra)
  4. 8–32 Hz + notch 50 Hz  (usunięcie szumu sieciowego)
  5. 8–12 Hz IIR        (tylko pasmo mu)
  6. 13–30 Hz IIR       (tylko pasmo beta)

Layout:
  Wiersze = sesje subjecta 3
  Linie (kolory) = warunki filtrowania
  Oś X = długość okna [s], Oś Y = test accuracy ± std po foldach

Wynik: filter_effect.png
"""

import os
import re
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

mne.set_log_level('ERROR')

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
BRAINBOT_EVENTS   = {'rest': 1, 'left_hand': 2, 'right_hand': 3, 'feet': 5}
BRAINBOT_DATA_DIR = Path(__file__).parent / "brainbot_data/5s-interval-nofiltering"
SUBJECT_ID        = 3
TMAX_MAX          = 5.0
TMIN_START        = 1.0
TMAX_STEP         = 0.5
N_FOLDS           = 5
N_JOBS            = 32

# Warunki filtrowania: (nazwa, l_freq, h_freq, method, notch_freq)
FILTER_CONDITIONS = [
    ("8–32 Hz IIR",          8.0,  32.0, 'iir', None),
    ("4–40 Hz IIR",          4.0,  40.0, 'iir', None),
    ("8–32 Hz FIR",          8.0,  32.0, 'fir', None),
    ("8–32 Hz + notch 50",   8.0,  32.0, 'iir', 50.0),
    ("8–12 Hz (mu)",         8.0,  12.0, 'iir', None),
    ("13–30 Hz (beta)",     13.0,  30.0, 'iir', None),
]

CONDITION_COLORS = [
    "#1f77b4",  # niebieski   — 8-32 IIR (baseline)
    "#ff7f0e",  # pomarańczowy — 4-40 IIR
    "#2ca02c",  # zielony     — 8-32 FIR
    "#9467bd",  # fioletowy   — 8-32 + notch
    "#8c564b",  # brązowy     — mu
    "#e377c2",  # różowy      — beta
]

# ---------------------------------------------------------------------------
# Transformer filtrujący (obsługuje notch + bandpass)
# ---------------------------------------------------------------------------
class EpochFilter(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=256.0, l_freq=8.0, h_freq=32.0,
                 method='iir', notch_freq=None):
        self.sfreq      = sfreq
        self.l_freq     = l_freq
        self.h_freq     = h_freq
        self.method     = method
        self.notch_freq = notch_freq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = np.empty_like(X)
        for i, epoch in enumerate(X):
            data = epoch.astype(np.float64)
            if self.notch_freq is not None:
                data = mne.filter.notch_filter(
                    data, Fs=self.sfreq, freqs=self.notch_freq,
                    method='iir', verbose=False,
                )
            data = mne.filter.filter_data(
                data, sfreq=self.sfreq,
                l_freq=self.l_freq, h_freq=self.h_freq,
                method=self.method, verbose=False,
            )
            out[i] = data
        return out


def make_pipeline(sfreq, l_freq, h_freq, method, notch_freq):
    return Pipeline([
        ("filt", EpochFilter(sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
                             method=method, notch_freq=notch_freq)),
        ("cov",  Covariances(estimator='oas')),
        ("ts",   TangentSpace(metric='riemann')),
        ("lr",   LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')),
    ])


# ---------------------------------------------------------------------------
# BrainBot — ładowanie
# ---------------------------------------------------------------------------
def load_session(subject_id, session_id):
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


def get_xy(epochs, tmax):
    wanted = set(BRAINBOT_EVENTS.values())
    mask = np.isin(epochs.events[:, 2], list(wanted))
    cropped = epochs.copy().crop(tmin=0.0, tmax=tmax, verbose=False)
    X = cropped.get_data()[mask]
    inv = {v: k for k, v in BRAINBOT_EVENTS.items()}
    y_str = np.array([inv[i] for i in epochs.events[mask, 2]])
    le = LabelEncoder()
    return X, le.fit_transform(y_str)


# ---------------------------------------------------------------------------
# Obliczanie krzywej per-warunek
# ---------------------------------------------------------------------------
def compute_window_curve(X_full, y, sfreq, cond_name, l_freq, h_freq, method, notch_freq):
    tmax_values = np.arange(TMIN_START, TMAX_MAX + 0.01, TMAX_STEP)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    means, stds = [], []

    for tmax in tmax_values:
        n_samples = int(np.round(tmax * sfreq)) + 1
        X_crop = X_full[:, :, :n_samples]
        pipe = make_pipeline(sfreq, l_freq, h_freq, method, notch_freq)
        scores = cross_val_score(pipe, X_crop, y,
                                 cv=cv, scoring='accuracy', n_jobs=N_JOBS)
        means.append(scores.mean())
        stds.append(scores.std())

    print(f"    [{cond_name}] max_acc={max(means):.3f}  "
          f"mean_std={np.mean(stds):.3f}")
    return tmax_values, np.array(means), np.array(stds)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Zbierz ID sesji
    pattern = re.compile(rf"subject{SUBJECT_ID}_ses(\d+)_run\d+_.*\.epo\.fif")
    session_ids = sorted(set(
        int(m.group(1))
        for f in os.listdir(BRAINBOT_DATA_DIR)
        if (m := pattern.match(f))
    ))

    # Wyniki: ses_id -> list of (cond_name, tmax_values, means, stds)
    session_results = {}

    for ses_id in session_ids:
        print(f"\n{'='*50}")
        print(f"Sesja {ses_id}")
        print(f"{'='*50}")
        epochs, sfreq = load_session(SUBJECT_ID, ses_id)
        if epochs is None:
            print("  brak danych")
            continue

        X_full, y = get_xy(epochs, TMAX_MAX)
        print(f"  {len(X_full)} epok, sfreq={sfreq} Hz")

        cond_results = []
        for cond_name, l_freq, h_freq, method, notch_freq in FILTER_CONDITIONS:
            tv, m, s = compute_window_curve(
                X_full, y, sfreq, cond_name,
                l_freq, h_freq, method, notch_freq,
            )
            cond_results.append((cond_name, tv, m, s))

        session_results[ses_id] = cond_results

    valid_sessions = [s for s in session_results if session_results[s]]
    if not valid_sessions:
        print("Brak wyników.")
        exit()

    # ---------------------------------------------------------------------------
    # Wykres: dwa rzędy — accuracy (góra) i std/dewiacja (dół), kolumny = sesje
    # ---------------------------------------------------------------------------
    n_ses = len(valid_sessions)
    fig, axes = plt.subplots(
        2, n_ses,
        figsize=(5 * n_ses, 9),
        squeeze=False,
        gridspec_kw=dict(hspace=0.55, wspace=0.35),
    )

    for col, ses_id in enumerate(sorted(valid_sessions)):
        ax_acc = axes[0][col]
        ax_std = axes[1][col]
        cond_results = session_results[ses_id]

        chance = 1.0 / len(BRAINBOT_EVENTS)

        for (cond_name, tv, means, stds), color in zip(cond_results, CONDITION_COLORS):
            ax_acc.plot(tv, means, color=color, linewidth=2, marker='o',
                        markersize=4, label=cond_name)
            ax_acc.fill_between(tv, means - stds, means + stds,
                                alpha=0.12, color=color)
            ax_std.plot(tv, stds, color=color, linewidth=2, marker='o',
                        markersize=4, label=cond_name)

        ax_acc.axhline(chance, color='gray', linestyle='--', linewidth=1,
                       label=f"chance={chance:.2f}")
        ax_acc.set_title(f"Sesja {ses_id} — accuracy", fontsize=10)
        ax_acc.set_ylabel("Test accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.legend(fontsize=6.5)
        ax_acc.grid(True, alpha=0.3)

        ax_std.set_title(f"Sesja {ses_id} — dewiacja (std po foldach)", fontsize=10)
        ax_std.set_ylabel("Std accuracy")
        ax_std.set_ylim(0, 0.35)
        ax_std.legend(fontsize=6.5)
        ax_std.grid(True, alpha=0.3)

        for ax in (ax_acc, ax_std):
            ax.set_xlabel("Długość okna [s]")
            tmax_values = cond_results[0][1]
            ax.set_xticks(tmax_values)
            ax.tick_params(axis='x', labelrotation=45, labelsize=7)

    fig.suptitle(
        f"Hipoteza B: wpływ filtrowania na accuracy i dewiację — BrainBot sub{SUBJECT_ID}\n"
        f"4 klasy: {', '.join(BRAINBOT_EVENTS.keys())}  |  {N_FOLDS}-fold CV  |  TSLR",
        fontsize=12, y=1.01,
    )

    out_path = Path(__file__).parent / "filter_effect.png"
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWykres zapisany: {out_path}")
