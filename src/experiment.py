import os
import sys
import yaml
import pandas as pd
import numpy as np
from IPython.display import display

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_and_sample_data, create_validation_windows, load_windows
from src.metrics import smape, mase
from src.models import (
    naive_forecast, seasonal_naive_forecast, train_ets, train_theta,
    train_catboost_global, prepare_catboost_data, catboost_predict_global,
    train_neural_model, predict_neural_model,
    train_catboost_on_series, catboost_predict,
    compute_series_features  # we need to define this
)
from src.ensemble import stacking_catboost_theta, stacking_catboost_seasonalnaive, stacking_neural

def compute_series_features(series, freq, seasonality):
    from statsmodels.tsa.stattools import acf
    feat = {}
    feat['length'] = len(series)
    feat['mean'] = np.mean(series)
    feat['std'] = np.std(series)
    feat['min'] = np.min(series)
    feat['max'] = np.max(series)
    if len(series) > 1:
        feat['acf1'] = acf(series, nlags=1, fft=False)[1]
    else:
        feat['acf1'] = 0
    feat['seasonality'] = seasonality[freq]
    return feat

def step3_baselines(config):
    """Run baselines on windows and return results DataFrame."""
    windows = load_windows(config)
    results = []
    for win in windows:
        train = np.array(win['train'])
        val = np.array(win['val'])
        horizon = win['horizon']
        season = config['seasonality'][win['freq']]

        # Naive
        pred = naive_forecast(train, horizon)
        results.append(('Naive', win, smape(val, pred), mase(val, pred, train, season)))

        # Seasonal Naive
        pred = seasonal_naive_forecast(train, horizon, season)
        results.append(('SeasonalNaive', win, smape(val, pred), mase(val, pred, train, season)))

        # AutoETS
        pred = train_ets(train, horizon, win['freq'], config['seasonality'])
        if not np.any(np.isnan(pred)):
            results.append(('AutoETS', win, smape(val, pred), mase(val, pred, train, season)))

        # Theta
        pred = train_theta(train, horizon, win['freq'], config['seasonality'])
        if not np.any(np.isnan(pred)):
            results.append(('Theta', win, smape(val, pred), mase(val, pred, train, season)))

    df = pd.DataFrame(results, columns=['method', 'win', 'smape', 'mase'])
    agg = df.groupby('method').agg(
        smape_mean=('smape', 'mean'), smape_std=('smape', 'std'),
        mase_mean=('mase', 'mean'), mase_std=('mase', 'std')
    ).reset_index()
    print("\nBaselines aggregated:")
    display(agg)
    return df

def step4_global_models(config):
    """Train CatBoost and PatchTST on windows and compare."""
    windows = load_windows(config)
    from src.models import FEATURE_KEYS, create_features

    # Group by freq
    windows_by_freq = {f: [] for f in config['seasonality']}
    for w in windows:
        windows_by_freq[w['freq']].append(w)

    # CatBoost
    results_cb = []
    for freq, wins in windows_by_freq.items():
        if not wins:
            continue
        horizon = wins[0]['horizon']
        X, y, meta = [], [], []
        for w in wins:
            train = np.array(w['train'])
            val = np.array(w['val'])
            if len(train) < 6:
                continue
            feats = create_features(train)
            if any(np.isnan(feats)):
                continue
            X.append(feats)
            y.append(val[:horizon])
            meta.append((w['series_id'], w['window_id']))
        if not X:
            continue
        X_arr = np.array(X)
        y_arr = np.array(y)
        model = train_catboost_global(X_arr, y_arr, config['catboost_params'])
        if model is None:
            continue
        preds = model.predict(X_arr)
        for i, (sid, wid) in enumerate(meta):
            true = y_arr[i]
            pred = preds[i]
            train_orig = [w['train'] for w in wins if w['series_id']==sid and w['window_id']==wid][0]
            season = config['seasonality'][freq]
            results_cb.append(('CatBoost', freq, sid, wid, smape(true, pred), mase(true, pred, train_orig, season)))

    df_cb = pd.DataFrame(results_cb, columns=['model','freq','series_id','window_id','smape','mase'])
    print("\nCatBoost on windows:")
    display(df_cb.groupby('model').agg(smape_mean=('smape','mean'), smape_std=('smape','std'),
                                        mase_mean=('mase','mean'), mase_std=('mase','std')).reset_index())

    # PatchTST
    results_pt = []
    for freq, wins in windows_by_freq.items():
        if not wins:
            continue
        horizon = wins[0]['horizon']
        rows = []
        for w in wins:
            uid = f"{w['series_id']}_{w['window_id']}"
            for t, val in enumerate(w['train']):
                rows.append({'unique_id': uid, 'ds': t, 'y': val})
        df_freq = pd.DataFrame(rows)
        nf = NeuralForecast(
            models=[PatchTST(h=horizon, **config['patchtst_params'])],
            freq=1
        )
        nf.fit(df=df_freq)
        preds_df = nf.predict()
        for w in wins:
            uid = f"{w['series_id']}_{w['window_id']}"
            row = preds_df[preds_df['unique_id'] == uid]
            if row.empty:
                continue
            pred = row['PatchTST'].values[:horizon]
            if len(pred) < horizon:
                continue
            true = np.array(w['val'][:horizon])
            train_orig = np.array(w['train'])
            season = config['seasonality'][freq]
            results_pt.append(('PatchTST', freq, w['series_id'], w['window_id'],
                               smape(true, pred), mase(true, pred, train_orig, season)))

    df_pt = pd.DataFrame(results_pt, columns=['model','freq','series_id','window_id','smape','mase'])
    print("\nPatchTST on windows:")
    display(df_pt.groupby('model').agg(smape_mean=('smape','mean'), smape_std=('smape','std'),
                                        mase_mean=('mase','mean'), mase_std=('mase','std')).reset_index())

    # Combined
    all_res = pd.concat([df_cb, df_pt], ignore_index=True)
    comp = all_res.groupby('model').agg(
        smape_mean=('smape','mean'), smape_std=('smape','std'),
        mase_mean=('mase','mean'), mase_std=('mase','std')
    ).reset_index()
    print("\nComparison CatBoost vs PatchTST:")
    display(comp)

def step5_simple_ensembles(config):
    """Run step 5: individual models, simple avg, stacking with CatBoost+SeasonalNaive."""
    from src.models import train_catboost_on_series, catboost_predict, seasonal_naive_forecast
    from src.ensemble import stacking_catboost_seasonalnaive
    meta = pd.read_csv(os.path.join(config['data_dir'], "series_horizons.csv"))
    results = {'model': [], 'freq': [], 'series_id': [], 'smape': [], 'mase': []}

    for freq in ['Yearly', 'Quarterly', 'Monthly']:
        print(f"\nProcessing {freq}")
        train_df, test_df = load_freq_data(config, freq)
        horizon = meta[meta['frequency'] == freq]['horizon'].iloc[0]
        series_ids = train_df['id'].tolist()

        # Prepare data
        train_dict = {}
        test_dict = {}
        full_dict = {}
        for sid in series_ids:
            train_row = train_df[train_df['id']==sid].iloc[0,1:].dropna().values
            test_row = test_df[test_df['id']==sid].iloc[0,1:].dropna().values
            if len(train_row) < 6 + horizon:
                continue
            train_dict[sid] = train_row
            test_dict[sid] = test_row[:horizon]
            full_dict[sid] = train_row

        if not train_dict:
            continue

        # Individual CatBoost models
        pred_cb_test = {}
        for sid, train_series in train_dict.items():
            model = train_catboost_on_series(train_series, horizon, config['catboost_params'])
            if model is not None:
                pred = catboost_predict(model, train_series, horizon)
                if not np.any(np.isnan(pred)):
                    pred_cb_test[sid] = pred

        # Global PatchTST
        rows = []
        for sid, train_series in train_dict.items():
            for t, val in enumerate(train_series):
                rows.append({'unique_id': sid, 'ds': t, 'y': val})
        df_freq = pd.DataFrame(rows)
        nf = NeuralForecast(
            models=[PatchTST(h=horizon, **config['patchtst_params'])],
            freq=1
        )
        nf.fit(df=df_freq)
        preds_df = nf.predict()
        pred_ptst_test = {}
        for sid in train_dict:
            rows = preds_df[preds_df['unique_id']==sid].sort_values('ds')['PatchTST'].values
            if len(rows) >= horizon:
                pred_ptst_test[sid] = rows[:horizon]

        # Collect metrics
        for sid in train_dict:
            true = test_dict[sid]
            train_series = train_dict[sid]
            season = config['seasonality'][freq]

            if sid in pred_cb_test:
                results['model'].append('CatBoost')
                results['freq'].append(freq)
                results['series_id'].append(sid)
                results['smape'].append(smape(true, pred_cb_test[sid]))
                results['mase'].append(mase(true, pred_cb_test[sid], train_series, season))

            if sid in pred_ptst_test:
                results['model'].append('PatchTST')
                results['freq'].append(freq)
                results['series_id'].append(sid)
                results['smape'].append(smape(true, pred_ptst_test[sid]))
                results['mase'].append(mase(true, pred_ptst_test[sid], train_series, season))

            if sid in pred_cb_test and sid in pred_ptst_test:
                avg = (pred_cb_test[sid] + pred_ptst_test[sid]) / 2
                results['model'].append('SimpleAvg')
                results['freq'].append(freq)
                results['series_id'].append(sid)
                results['smape'].append(smape(true, avg))
                results['mase'].append(mase(true, avg, train_series, season))

        # Stacking CatBoost + SeasonalNaive
        # (simplified: use the function from ensemble)
        # We'll just reuse the logic, but we need to collect stacking preds similarly
        # For simplicity, we can call a function that returns stacking_preds and then evaluate
        # Let's implement directly:
        X_stack, y_stack = [], []
        for sid in train_dict:
            train_series = train_dict[sid]
            if len(train_series) < 2*horizon:
                continue
            train_part = train_series[:-horizon]
            val_part = train_series[-horizon:]

            cb_model_val = train_catboost_on_series(train_part, horizon, config['catboost_params'])
            if cb_model_val is None:
                continue
            pred_cb_val = catboost_predict(cb_model_val, train_part, horizon)
            if np.any(np.isnan(pred_cb_val)):
                continue
            pred_sn_val = seasonal_naive_forecast(train_part, horizon, config['seasonality'][freq])
            for t in range(horizon):
                X_stack.append([pred_cb_val[t], pred_sn_val[t]])
                y_stack.append(val_part[t])

        if X_stack:
            X_stack = np.array(X_stack)
            y_stack = np.array(y_stack)
            meta_model = LinearRegression().fit(X_stack, y_stack)

            for sid in train_dict:
                if sid not in pred_cb_test:
                    continue
                pred_sn_test = seasonal_naive_forecast(train_dict[sid], horizon, config['seasonality'][freq])
                pred_cb_i = pred_cb_test[sid]
                if not np.any(np.isnan(pred_cb_i)) and not np.any(np.isnan(pred_sn_test)):
                    stack_pred = np.zeros(horizon)
                    for t in range(horizon):
                        stack_pred[t] = meta_model.predict([[pred_cb_i[t], pred_sn_test[t]]])[0]
                    results['model'].append('Stacking')
                    results['freq'].append(freq)
                    results['series_id'].append(sid)
                    results['smape'].append(smape(true, stack_pred))
                    results['mase'].append(mase(true, stack_pred, train_dict[sid], config['seasonality'][freq]))

    df = pd.DataFrame(results)
    agg = df.groupby('model').agg(
        smape_mean=('smape','mean'), smape_std=('smape','std'),
        mase_mean=('mase','mean'), mase_std=('mase','std')
    ).reset_index()
    print("\n=== Step 5 results ===")
    display(agg)
    return df

def step6_neural_ensemble(config):
    """Run neural ensemble (PatchTST + N-BEATS) with CV stacking."""
    from src.ensemble import stacking_neural
    meta = pd.read_csv(os.path.join(config['data_dir'], "series_horizons.csv"))
    all_results = []

    for freq in ['Yearly', 'Quarterly', 'Monthly']:
        print(f"\nProcessing {freq}")
        train_df, test_df = load_freq_data(config, freq)
        horizon = meta[meta['frequency'] == freq]['horizon'].iloc[0]

        train_dict, test_dict, full_dict = {}, {}, {}
        for sid in train_df['id'].tolist():
            tr = train_df[train_df['id']==sid].iloc[0,1:].dropna().values
            te = test_df[test_df['id']==sid].iloc[0,1:].dropna().values
            if len(tr) < 2*horizon:
                continue
            train_dict[sid] = tr
            test_dict[sid] = te[:horizon]
            full_dict[sid] = tr

        if not train_dict:
            continue

        pt_test, nb_test, stacking, valid = stacking_neural(config, train_dict, test_dict, full_dict, freq, horizon)
        if pt_test is None:
            continue

        for sid in full_dict:
            true = test_dict[sid]
            tr = full_dict[sid]
            seas = config['seasonality'][freq]

            if sid in pt_test and not np.any(np.isnan(pt_test[sid])):
                all_results.append(('PatchTST', freq, sid, smape(true, pt_test[sid]), mase(true, pt_test[sid], tr, seas)))
            if sid in nb_test and not np.any(np.isnan(nb_test[sid])):
                all_results.append(('NBEATS', freq, sid, smape(true, nb_test[sid]), mase(true, nb_test[sid], tr, seas)))
            if sid in pt_test and sid in nb_test and not (np.any(np.isnan(pt_test[sid])) or np.any(np.isnan(nb_test[sid]))):
                avg = (pt_test[sid] + nb_test[sid]) / 2
                all_results.append(('SimpleAvg', freq, sid, smape(true, avg), mase(true, avg, tr, seas)))
            if sid in stacking:
                all_results.append(('Stacking', freq, sid, smape(true, stacking[sid]), mase(true, stacking[sid], tr, seas)))

    if all_results:
        df = pd.DataFrame(all_results, columns=['model','freq','series_id','smape','mase'])
        agg = df.groupby('model').agg(
            smape_mean=('smape','mean'), smape_std=('smape','std'),
            mase_mean=('mase','mean'), mase_std=('mase','std')
        ).reset_index()
        print("\n=== Neural ensemble results (PatchTST + N-BEATS) ===")
        display(agg)
        return df
    else:
        return None

def step7_final_stacking(config):
    """Final stacking: PatchTST + Theta + series features, with CatBoost meta-model."""
    from src.ensemble import stacking_catboost_theta
    meta = pd.read_csv(os.path.join(config['data_dir'], "series_horizons.csv"))
    all_results = []

    for freq in ['Yearly', 'Quarterly', 'Monthly']:
        print(f"\nProcessing {freq}")
        train_df, test_df = load_freq_data(config, freq)
        horizon = meta[meta['frequency'] == freq]['horizon'].iloc[0]

        min_len = {'Yearly':6, 'Quarterly':12, 'Monthly':24}[freq]
        train_dict, test_dict, full_dict, feat_dict = {}, {}, {}, {}
        for sid in train_df['id'].tolist():
            tr = train_df[train_df['id']==sid].iloc[0,1:].dropna().values
            te = test_df[test_df['id']==sid].iloc[0,1:].dropna().values
            if len(tr) < min_len:
                continue
            train_dict[sid] = tr
            test_dict[sid] = te[:horizon]
            full_dict[sid] = tr
            feat_dict[sid] = compute_series_features(tr, freq, config['seasonality'])

        if not train_dict:
            continue

        pt_test, th_test, stacking, valid, scaler = stacking_catboost_theta(
            config, train_dict, test_dict, full_dict, feat_dict, freq, horizon
        )
        if pt_test is None:
            continue

        for sid in full_dict:
            true = test_dict[sid]
            tr = full_dict[sid]
            seas = config['seasonality'][freq]

            if sid in pt_test and not np.any(np.isnan(pt_test[sid])):
                all_results.append(('PatchTST', freq, sid, smape(true, pt_test[sid]), mase(true, pt_test[sid], tr, seas)))
            if sid in th_test and not np.any(np.isnan(th_test[sid])):
                all_results.append(('Theta', freq, sid, smape(true, th_test[sid]), mase(true, th_test[sid], tr, seas)))
            if sid in pt_test and sid in th_test and not (np.any(np.isnan(pt_test[sid])) or np.any(np.isnan(th_test[sid]))):
                avg = (pt_test[sid] + th_test[sid]) / 2
                all_results.append(('SimpleAvg', freq, sid, smape(true, avg), mase(true, avg, tr, seas)))
            if sid in stacking:
                all_results.append(('Stacking', freq, sid, smape(true, stacking[sid]), mase(true, stacking[sid], tr, seas)))

    if all_results:
        df = pd.DataFrame(all_results, columns=['model','freq','series_id','smape','mase'])
        agg = df.groupby('model').agg(
            smape_mean=('smape','mean'), smape_std=('smape','std'),
            mase_mean=('mase','mean'), mase_std=('mase','std')
        ).reset_index()
        print("\n=== FINAL STACKING RESULTS (PatchTST + Theta + features) ===")
        display(agg)
        return df
    else:
        return None

def load_freq_data(config, freq):
    proc_dir = config['data_dir']
    train_path = os.path.join(proc_dir, f"{freq}_train_selected.csv")
    test_path = os.path.join(proc_dir, f"{freq}_test_selected.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def run_experiment(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Step 1
    load_and_sample_data(config)
    # Step 2
    create_validation_windows(config)
    # Step 3
    step3_baselines(config)
    # Step 4
    step4_global_models(config)
    # Step 5
    step5_simple_ensembles(config)
    # Step 6 neural
    step6_neural_ensemble(config)
    # Step 7 final
    step7_final_stacking(config)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python src/experiment.py <config.yaml>")
        sys.exit(1)
    run_experiment(sys.argv[1])