import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = "."
OUTPUT_DIR = "outputs/heatmaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_csvs(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in data directory")

    items = []
    for file in csv_files:
        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        items.append((os.path.basename(file), df))

    print(f"Loaded {len(items)} files")
    return items


def pick_join_keys(dfs, user_keys=None):
    """Pick stable composite join keys present across all DataFrames.

    Priority order (left to right):
    - anon_id, pat_enc_csn_id_coded, order_proc_id_coded, order_time_jittered_utc
    - anon_id, pat_enc_csn_id_coded, order_proc_id_coded
    - anon_id
    If user_keys provided, validate they exist in all, else raise.
    """
    if user_keys:
        keys = [k.strip().lower() for k in user_keys]
        missing = {
            k
            for k in keys
            if not all(k in df.columns for df in dfs)
        }
        if missing:
            raise ValueError(f"Join keys not present in all files: {sorted(missing)}")
        return keys

    candidates = [
        ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded", "order_time_jittered_utc"],
        ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"],
        ["anon_id"],
    ]
    for keys in candidates:
        if all(all(k in df.columns for k in keys) for df in dfs):
            return keys
    # Fallback to anon_id if at least available in first df
    if "anon_id" in dfs[0].columns:
        return ["anon_id"]
    raise ValueError("Could not determine join keys shared across CSVs")


def deduplicate_by_keys(df, keys):
    # Collapse many-to-many to one row per key to avoid explosive merges
    # Using first occurrence; suitable for high-level EDA
    existing = [k for k in keys if k in df.columns]
    if not existing:
        return df
    return df.groupby(existing, as_index=False).first()


def merge_dataframes(dfs, on_keys, how="inner"):
    """Merge a list of DataFrames on provided keys using specified join type.

    To reduce memory, each subsequent merge excludes duplicate key columns and
    drops duplicate columns by name after the merge.
    """
    merged = deduplicate_by_keys(dfs[0], on_keys)
    for df in dfs[1:]:
        right = deduplicate_by_keys(df, on_keys)
        # Avoid duplicate key columns when keys appear multiple times due to prior merges
        right = right.loc[:, ~right.columns.duplicated()]
        right_non_keys = [c for c in right.columns if c not in on_keys]
        # Avoid bringing in columns already on the left to prevent suffix clashes
        existing_cols = set(merged.columns)
        right_non_keys = [c for c in right_non_keys if c not in existing_cols]
        if not right_non_keys:
            # Nothing new to add from this frame
            continue
        merged = pd.merge(
            merged,
            right[on_keys + right_non_keys],
            on=on_keys,
            how=how,
        )
        merged = merged.loc[:, ~merged.columns.duplicated()]

    print("Merged shape:", merged.shape)
    return merged


def drop_sparse_columns(df, threshold=0.7):
    missing_ratio = df.isnull().mean()
    df = df.loc[:, missing_ratio < threshold]
    print("After dropping sparse columns:", df.shape)
    return df


def plot_missing_heatmap(df, filename):
    if len(df) == 0:
        print("Skipping missing heatmap: empty DataFrame")
        return
    sample = df.sample(n=min(3000, len(df)), random_state=42)

    plt.figure(figsize=(14, 5))
    sns.heatmap(
        sample.isnull(),
        yticklabels=False,
        cbar=False
    )
    plt.title("Missing Value Heatmap (Sampled)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_correlation_heatmap(df, filename):
    numeric_df = df.select_dtypes(include=["int64", "float64"]).copy()
    if numeric_df.empty:
        print("Skipping correlation heatmap: no numeric columns")
        return

    if numeric_df.shape[1] > 25:
        numeric_df = numeric_df.iloc[:, :25]

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        numeric_df.corr(numeric_only=True),
        cmap="coolwarm",
        center=0
    )
    plt.title("Correlation Heatmap (Top Numeric Features)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def generate_per_file_heatmaps(items):
    for name, df in items:
        safe = os.path.splitext(name)[0]
        # Missingness per file
        plot_missing_heatmap(
            df,
            os.path.join(OUTPUT_DIR, f"{safe}_missing_values_heatmap.png"),
        )
        # Correlation per file
        plot_correlation_heatmap(
            df,
            os.path.join(OUTPUT_DIR, f"{safe}_correlation_heatmap.png"),
        )


def main():
    parser = argparse.ArgumentParser(description="Microbiology EDA and heatmaps")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Folder containing CSV files")
    parser.add_argument(
        "--join-keys",
        default=None,
        help="Comma-separated list of join keys to use (e.g., anon_id,pat_enc_csn_id_coded,order_proc_id_coded)",
    )
    parser.add_argument(
        "--join-type",
        default="inner",
        choices=["inner", "left", "right", "outer"],
        help="Join type when merging tables (default: inner)",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional: sample N rows from merged data before plotting to reduce memory",
    )
    parser.add_argument(
        "--per-file-only",
        action="store_true",
        help="Skip merging and generate heatmaps per CSV file only",
    )
    args = parser.parse_args()

    items = load_csvs(args.data_dir)

    if args.per_file_only:
        print("Generating per-file heatmaps (skipping merge) ...")
        generate_per_file_heatmaps(items)
        print("EDA (per-file) completed successfully")
        return

    # Attempt merged EDA with deduplication
    dfs = [df for _, df in items]
    user_keys = [k.strip() for k in args.join_keys.split(",")] if args.join_keys else None
    join_keys = pick_join_keys(dfs, user_keys)
    print(f"Using join keys: {join_keys}; join type: {args.join_type}")

    merged_df = merge_dataframes(dfs, on_keys=join_keys, how=args.join_type)
    merged_df = drop_sparse_columns(merged_df)

    if args.sample_rows is not None and args.sample_rows > 0 and len(merged_df) > args.sample_rows:
        merged_df = merged_df.sample(n=args.sample_rows, random_state=42)
        print(f"Downsampled merged data to {len(merged_df)} rows for plotting")

    os.makedirs("outputs", exist_ok=True)
    merged_df.to_csv("outputs/microbiology_combined_clean.csv", index=False)

    plot_missing_heatmap(
        merged_df,
        os.path.join(OUTPUT_DIR, "missing_values_heatmap.png")
    )

    plot_correlation_heatmap(
        merged_df,
        os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    )

    print("EDA completed successfully")


if __name__ == "__main__":
    main()
