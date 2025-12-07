import os
import json
import mne
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from mne.decoding import CSP

# 1. Configuration
data_dir = "../data/processed/"
filenames = [
    'kasia1_run1_20251206_185544_raw_epochs_splitted_segment=2.0-step=1.0-epo.fif',
    'kasia2_run1_20251206_191125_raw_epochs_splitted_segment=2.0-step=1.0-epo.fif',
]
# Events to include in the model (matching your MOABB paradigm)
target_events = ["relax", "left_hand", "right_hand", "both_hands", "both_feets"]

# 2. Load and concatenate data
epochs_list = []
for fname in filenames:
    fpath = os.path.join(data_dir, fname)
    print(f"Loading: {fpath}")
    # preload=True is required for get_data()
    epochs = mne.read_epochs(fpath, preload=True, verbose=False)
    epochs_list.append(epochs)

# Combine runs
all_epochs = mne.concatenate_epochs(epochs_list)
# it's already preprocessed and segmented as needed
# 3. Filter for specific classes
# This selects only the epochs corresponding to the target events
epochs_selected = all_epochs[target_events]
print(f"Total epochs for training: {len(epochs_selected)}")
print(f"Classes: {epochs_selected.event_id}")

# 4. Prepare Training Data
X = epochs_selected.get_data(copy=True)  # Shape: (n_epochs, n_channels, n_times)
y = epochs_selected.events[:, -1]        # Shape: (n_epochs,)

# Split into train and validation (10%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# 5. Define Pipeline (CSP + SVM)
# Using the exact parameters from your notebook
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
svm = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
clf = make_pipeline(csp, svm)

# 6. Train
print("Fitting classifier on training set...")
clf.fit(X_train, y_train)

# 7. Evaluate
print("\nEvaluating on validation set...")
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc:.4f}")
print("\nClassification Report:")
# Get target names from event_id
# event_id is like {'relax': 1, 'left_hand': 2, ...}
# We need to map class IDs to names
id_to_name = {v: k for k, v in epochs_selected.event_id.items()}
# Ensure we only include classes that are present in y (though stratify should keep them all)
unique_labels = sorted(list(set(y)))
target_names = [id_to_name[label] for label in unique_labels]

print(classification_report(y_val, y_pred, target_names=target_names))

# 8. Save Model and Metadata
model_filename = "csp_svm_kasia_model.pkl"
mapping_filename = "csp_svm_kasia_mapping.json"

# Save the sklearn pipeline
joblib.dump(clf, model_filename)
print(f"Model saved to: {model_filename}")

# Save the event mapping (Label -> ID) so you know what the predictions mean later
with open(mapping_filename, 'w') as f:
    json.dump(epochs_selected.event_id, f, indent=4)
print(f"Event mapping saved to: {mapping_filename}")

# Try to load and predict to verify
loaded_clf = joblib.load(model_filename)
y_pred = loaded_clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Loaded Validation Accuracy: {acc:.4f}")