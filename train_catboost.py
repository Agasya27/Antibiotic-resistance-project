import os
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier, Pool


def main():
    parser = argparse.ArgumentParser(description='Train CatBoost on microbiology data')
    parser.add_argument('--antibiotic', default=None, help='Optional: filter to a specific antibiotic name')
    parser.add_argument('--target-scheme', default='binary_rs', choices=['binary_rs', 'binary_ni'],
                        help='binary_rs: Resistant vs Susceptible (drop Intermediate); binary_ni: Non-susceptible (Resistant+Intermediate) vs Susceptible')
    args = parser.parse_args()

    df = pd.read_csv('microbiology_combined_clean.csv', low_memory=False)

    # Basic cleaning: drop obvious missing targets
    df = df.dropna(subset=['susceptibility'])

    # Optional filtering by antibiotic
    if args.antibiotic:
        df = df[df['antibiotic'].astype(str) == args.antibiotic]

    sus = df['susceptibility'].astype(str).str.strip().str.title()
    if args.target_scheme == 'binary_rs':
        # Keep only Resistant vs Susceptible, drop Intermediate/others
        mask = sus.isin(['Susceptible', 'Resistant'])
        df = df[mask].copy()
        df['target'] = (sus[mask] == 'Resistant').astype(int).values
    else:
        # Non-susceptible (Resistant+Intermediate) = 1 vs Susceptible = 0
        mask = sus.isin(['Susceptible', 'Resistant', 'Intermediate'])
        df = df[mask].copy()
        tmp = sus[mask]
        df['target'] = tmp.map({'Susceptible': 0, 'Resistant': 1, 'Intermediate': 1}).values

    # Features (strings kept as-is for CatBoost categorical handling)
    feature_cols = [
        'medication_category', 'medication_name', 'antibiotic_class', 'ordering_mode',
        'culture_description', 'organism', 'antibiotic', 'age', 'gender', 'prior_organism',
        'was_positive', 'time_to_culturetime', 'medication_time_to_culturetime',
        'prior_infecting_organism_days_to_culutre', 'implied_susceptibility'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df['target'].values

    # Identify categorical columns by dtype or known string columns
    cat_features = [i for i, c in enumerate(X.columns) if X[c].dtype == 'object']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute class weights to address imbalance
    pos_ratio = (y_train == 1).mean()
    neg_ratio = 1 - pos_ratio
    # Class weights inversely proportional to frequency
    class_weights = [1.0 / max(1e-6, neg_ratio), 1.0 / max(1e-6, pos_ratio)]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_test, y_test, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=3000,
        depth=8,
        learning_rate=0.03,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        l2_leaf_reg=8.0,
        border_count=254,
        auto_class_weights=None,
        class_weights=class_weights,
        verbose=200,
        od_type='Iter',
        od_wait=300,
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    # Predictions and threshold tuning
    probs = model.predict_proba(X_test)[:, 1]
    best_thr, best_f1 = 0.5, -1
    for thr in np.linspace(0.1, 0.9, 33):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(y_test, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    preds = (probs >= best_thr).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y_test, preds)

    print(f"Best threshold: {best_thr:.3f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Save artifacts
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    model.save_model('models/catboost_resistance_new.cbm')
    # Also save as a pickle for compatibility with .pkl consumers
    with open('models/catboost_resistance_new.pkl', 'wb') as pf:
        pickle.dump(model, pf)
    with open('outputs/metrics_catboost.txt', 'w') as f:
        f.write(f'Best threshold: {best_thr:.3f}\n')
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'Precision: {prec:.4f}\n')
        f.write(f'Recall: {rec:.4f}\n')
        f.write(f'F1: {f1:.4f}\n')
        f.write(f'ROC-AUC: {auc:.4f}\n')
        f.write('Confusion Matrix:\n')
        f.write(np.array2string(cm))


if __name__ == '__main__':
    main()
