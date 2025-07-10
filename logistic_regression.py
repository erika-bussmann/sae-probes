import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Tuple

def train_and_evaluate_probe(
    phrase_activations: Dict[str, torch.Tensor],
    phrase_labels: Dict[str, int], # Assuming binary (0 or 1) labels for simplicity
    test_size: float = 0.2,
    random_state: int = 42,
    probe_type: str = 'last_token', # 'last_token', 'mean_pool'
    log_reg_C: float = 0.1, # Inverse of regularization strength
    log_reg_max_iter: int = 1000
) -> None:
    """
    Trains a Logistic Regression probe on extracted activations and evaluates its performance.

    Args:
        phrase_activations: A dictionary mapping input phrases to their raw activation tensors
                            (shape: (sequence_length, hidden_size)).
        phrase_labels: A dictionary mapping input phrases to their corresponding ground truth labels (int).
                       Labels should be 0 or 1 for binary classification.
        test_size: The proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility of data splitting and probe training.
        probe_type: Strategy to get a single vector from activations:
                    - 'last_token': Uses the activation of the last token in the sequence.
                    - 'mean_pool': Computes the mean across the sequence length.
        log_reg_C: Inverse of regularization strength for Logistic Regression.
        log_reg_max_iter: Maximum number of iterations for the Logistic Regression solver.
    """
    if not phrase_activations or not phrase_labels:
        print("Error: Empty activations or labels provided. Cannot train probe.")
        return

    # 1. Prepare Data for Probe
    X_list = [] # Features (activations)
    y_list = [] # Labels

    # Ensure consistent order and alignment between activations and labels
    phrases_in_order = list(phrase_activations.keys())

    for phrase in phrases_in_order:
        if phrase not in phrase_labels:
            print(f"Warning: Label not found for phrase: '{phrase}'. Skipping.")
            continue

        activation_tensor = phrase_activations[phrase]
        
        # Ensure activations are 2D (sequence_length, hidden_size)
        if activation_tensor.ndim == 3 and activation_tensor.shape[0] == 1:
            activation_tensor = activation_tensor.squeeze(0) # Remove batch dim if present

        if activation_tensor.ndim != 2:
            print(f"Warning: Activation for '{phrase}' has unexpected dimensions: {activation_tensor.shape}. Skipping.")
            continue

        if probe_type == 'last_token':
            # Take the activation of the last token
            feature_vector = activation_tensor[-1, :].numpy()
        elif probe_type == 'mean_pool':
            # Mean pool across the sequence length
            feature_vector = activation_tensor.mean(dim=0).numpy()
        else:
            print(f"Error: Invalid probe_type '{probe_type}'. Choose 'last_token' or 'mean_pool'.")
            return

        X_list.append(feature_vector)
        y_list.append(phrase_labels[phrase])

    if not X_list:
        print("No valid data points to train the probe. Check input activations and labels.")
        return

    X = np.array(X_list)
    y = np.array(y_list)

    # Check for binary classification:
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        print(f"Warning: Probe expects binary labels (0 or 1), but found {len(unique_labels)} unique labels: {unique_labels}.")
        if len(unique_labels) < 2:
            print("Cannot train a classifier with fewer than 2 classes.")
            return
        # For multi-class, you might need to adjust f1_score `average` parameter
        # or use `classification_report` for more detailed metrics.
        # For this example, we'll proceed but be aware.


    # 2. Split Data into Training and Test Sets
    # Use stratification if dealing with imbalanced classes, but for simplicity, default split here.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y # Stratify is good practice for classification
    )

    print(f"\n--- Probe Training & Evaluation ---")
    print(f"Dataset size: {len(X)} samples")
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Probe type (activation aggregation): '{probe_type}'")

    # 3. Initialize and Train Logistic Regression Probe
    probe = LogisticRegression(random_state=random_state, C=log_reg_C, max_iter=log_reg_max_iter)
    probe.fit(X_train, y_train)
    print("Logistic Regression probe trained.")

    # 4. Evaluate the Probe
    y_pred = probe.predict(X_test)

    # For binary classification (our assumed case), 'binary' average is default for f1_score
    # For multi-class, consider 'macro', 'micro', or 'weighted'
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary') # 'binary' is suitable for 2 classes, pos_label=1

    print(f"\nResults on Test Set (using {test_size*100}% of data):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Optional: print classification report for more detail
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))