import os
import argparse
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import Pool


def build_targets(df: pd.DataFrame, target_scheme: str) -> pd.DataFrame:
    df = df.dropna(subset=['susceptibility']).copy()
    sus = df['susceptibility'].astype(str).str.strip().str.title()
    if target_scheme == 'binary_rs':
        mask = sus.isin(['Susceptible', 'Resistant'])
        df = df[mask].copy()
        df['target'] = (sus[mask] == 'Resistant').astype(int).values
        df['target_label'] = np.where(df['target'] == 1, 'Resistant', 'Susceptible')
    else:
        mask = sus.isin(['Susceptible', 'Resistant', 'Intermediate'])
        df = df[mask].copy()
        tmp = sus[mask]
        df['target'] = tmp.map({'Susceptible': 0, 'Resistant': 1, 'Intermediate': 1}).values
        df['target_label'] = np.where(df['target'] == 1, 'Non-susceptible', 'Susceptible')
    return df


def get_feature_cols(df: pd.DataFrame):
    feature_cols = [
        'medication_category', 'medication_name', 'antibiotic_class', 'ordering_mode',
        'culture_description', 'organism', 'antibiotic', 'age', 'gender', 'prior_organism',
        'was_positive', 'time_to_culturetime', 'medication_time_to_culturetime',
        'prior_infecting_organism_days_to_culutre', 'implied_susceptibility'
    ]
    return [c for c in feature_cols if c in df.columns]


def main():
    parser = argparse.ArgumentParser(description='Visualize CatBoost model predictions and importances')
    parser.add_argument('--model-path', default='models/catboost_resistance_new.pkl', help='Path to pickled CatBoost model')
    parser.add_argument('--csv', default='microbiology_combined_clean.csv', help='Input CSV used for training')
    parser.add_argument('--antibiotic', default=None, help='Optional: filter to a specific antibiotic name')
    parser.add_argument('--target-scheme', default='binary_rs', choices=['binary_rs', 'binary_ni'], help='Target definition to use for grouping')
    parser.add_argument('--outdir', default='outputs/visualizations', help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)
    if args.antibiotic:
        df = df[df['antibiotic'].astype(str) == args.antibiotic]
    df = build_targets(df, args.target_scheme)

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].copy()
    cat_features = [i for i, c in enumerate(X.columns) if X[c].dtype == 'object']
    pool = Pool(X, cat_features=cat_features)

    # Load model (.pkl preferred, fallback to .cbm)
    model = None
    if os.path.exists(args.model_path):
        with open(args.model_path, 'rb') as pf:
            model = pickle.load(pf)
    else:
        cbm_fallback = 'models/catboost_resistance_new.cbm'
        if os.path.exists(cbm_fallback):
            # Avoid dependency on CatBoost JSON when pickle unavailable; CatBoost can load .cbm via load_model
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(cbm_fallback)
        else:
            raise FileNotFoundError('No model file found (.pkl or .cbm)')

    # Predict probabilities for positive class
    probs = model.predict_proba(pool)[:, 1]
    df['prob_positive'] = probs

    # 1) Boxplot: probability by target label
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='target_label', y='prob_positive')
    sns.stripplot(data=df.sample(min(1000, len(df))), x='target_label', y='prob_positive', color='black', alpha=0.3, jitter=0.25)
    plt.title('Predicted Probability by Target')
    plt.xlabel('Target')
    plt.ylabel('Predicted P(positive)')
    plt.tight_layout()
    boxplot_path = os.path.join(args.outdir, 'boxplot_target_probs.png')
    plt.savefig(boxplot_path, dpi=150)
    plt.close()

    # 2) Scatterplot: age vs probability
    if 'age' in df.columns:
        scatter_df = df[['age', 'prob_positive', 'target_label']].dropna()
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=scatter_df.sample(min(5000, len(scatter_df))), x='age', y='prob_positive', hue='target_label', alpha=0.5)
        plt.title('Predicted Probability vs Age')
        plt.xlabel('Age')
        plt.ylabel('Predicted P(positive)')
        plt.legend(title='Target')
        plt.tight_layout()
        scatter_path = os.path.join(args.outdir, 'scatter_age_probs.png')
        plt.savefig(scatter_path, dpi=150)
        plt.close()

    # 3) Bar graph: feature importances
    importances = model.get_feature_importance(pool)
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False).head(25)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df, x='importance', y='feature', orient='h')
    plt.title('Top Feature Importances (CatBoost)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    bar_path = os.path.join(args.outdir, 'bar_feature_importance.png')
    plt.savefig(bar_path, dpi=150)
    plt.close()

    print('Saved:')
    print(f' - {boxplot_path}')
    if 'age' in df.columns:
        print(f' - {scatter_path}')
    print(f' - {bar_path}')


if __name__ == '__main__':
    main()
