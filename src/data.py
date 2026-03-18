import os
import random
import csv
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import requests

def download_file(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {url} -> {local_path}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)

def read_variable_length_csv(filepath):
    """Read M4 CSV, skipping header."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            row_id = row[0]
            values = [float(x) for x in row[1:] if x != '']
            rows.append([row_id] + values)
    df = pd.DataFrame(rows)
    df = df.rename(columns={0: 'id'})
    return df

def load_and_sample_data(config):
    """Step 1: Download and sample N series per frequency."""
    raw_dir = config['raw_data_dir']
    proc_dir = config['data_dir']
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    freq_horizon = config['freq_horizon']
    base_url = config['base_url']
    n_series = config['n_series_per_freq']

    for freq, horizon in freq_horizon.items():
        print(f"\nProcessing frequency: {freq}")
        train_file = f"{freq}-train.csv"
        test_file = f"{freq}-test.csv"
        train_path = os.path.join(raw_dir, train_file)
        test_path = os.path.join(raw_dir, test_file)

        download_file(base_url + "Train/" + train_file, train_path)
        download_file(base_url + "Test/" + test_file, test_path)

        train_df = read_variable_length_csv(train_path)
        test_df = read_variable_length_csv(test_path)
        print(f"  Total series: train={len(train_df)}, test={len(test_df)}")

        all_ids = train_df['id'].tolist()
        selected_ids = random.sample(all_ids, min(n_series, len(all_ids)))
        train_sel = train_df[train_df['id'].isin(selected_ids)].copy()
        test_sel = test_df[test_df['id'].isin(selected_ids)].copy()

        train_sel.to_csv(os.path.join(proc_dir, f"{freq}_train_selected.csv"), index=False)
        test_sel.to_csv(os.path.join(proc_dir, f"{freq}_test_selected.csv"), index=False)
        print(f"  Saved {len(train_sel)} series.")

    # Meta info
    horizon_info = []
    for freq, horizon in freq_horizon.items():
        train_df = pd.read_csv(os.path.join(proc_dir, f"{freq}_train_selected.csv"))
        for _, row in train_df.iterrows():
            series_id = row['id']
            series_length = row.iloc[1:].count()
            horizon_info.append({
                'id': series_id,
                'frequency': freq,
                'horizon': horizon,
                'train_length': series_length
            })
    horizon_df = pd.DataFrame(horizon_info)
    horizon_df.to_csv(os.path.join(proc_dir, "series_horizons.csv"), index=False)
    print("\nMeta information saved.")

def create_validation_windows(config):
    """Step 2: Create rolling windows for each series."""
    proc_dir = config['data_dir']
    seasonality = config['seasonality']
    meta = pd.read_csv(os.path.join(proc_dir, "series_horizons.csv"))
    all_windows = []

    for freq in seasonality.keys():
        train_df = pd.read_csv(os.path.join(proc_dir, f"{freq}_train_selected.csv"))
        for _, row in train_df.iterrows():
            series_id = row['id']
            meta_row = meta[(meta['id'] == series_id) & (meta['frequency'] == freq)]
            if len(meta_row) == 0:
                continue
            horizon = int(meta_row.iloc[0]['horizon'])
            train_values = row.iloc[1:].dropna().values.astype(float)
            T = len(train_values)
            max_possible = (T // horizon) - 1
            K = min(3, max_possible)  # at most 3 windows
            if K <= 0:
                continue
            for i in range(K):
                val_start = T - (i+1)*horizon
                val_end = T - i*horizon
                train_end = val_start
                all_windows.append({
                    'series_id': series_id,
                    'freq': freq,
                    'window_id': i,
                    'train': train_values[:train_end].copy(),
                    'val': train_values[val_start:val_end].copy(),
                    'horizon': horizon
                })

    with open(os.path.join(proc_dir, "validation_windows.pkl"), "wb") as f:
        pickle.dump(all_windows, f)
    print(f"Created {len(all_windows)} windows.")
    freq_counts = Counter([w['freq'] for w in all_windows])
    for freq, cnt in freq_counts.items():
        print(f"  {freq}: {cnt}")

def load_windows(config):
    proc_dir = config['data_dir']
    with open(os.path.join(proc_dir, "validation_windows.pkl"), "rb") as f:
        windows = pickle.load(f)
    return windows