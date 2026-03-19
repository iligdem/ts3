# src/ensemble.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from neuralforecast.models import PatchTST, NBEATS

from src.models import (
    train_patchtst_global, predict_patchtst,
    train_catboost_on_series, catboost_predict,
    seasonal_naive_forecast, train_theta,
    train_catboost_global, catboost_predict_global, prepare_catboost_data,
    train_neural_model, predict_neural_model,
    create_features, FEATURE_KEYS
)
from src.metrics import smape, mase

def stacking_catboost_theta(config, train_dict, test_dict, full_dict, feat_dict, freq, horizon):
    """
    Advanced stacking with CatBoost as meta-model, using PatchTST and Theta as base models.
    Includes series features.
    """
    seasonality = config['seasonality']
    n_folds = config['n_folds_cv']
    patchtst_params = config['patchtst_params']
    meta_params = config['meta_model_params']

    series_list = list(train_dict.keys())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config['seed'])

    oof_pt = {s: None for s in series_list}
    oof_th = {s: None for s in series_list}
    oof_true = {s: test_dict[s] for s in series_list}

    for fold, (tr_idx, val_idx) in enumerate(kf.split(series_list)):
        print(f"  Fold {fold+1}/{n_folds}")
        tr_series = [series_list[i] for i in tr_idx]
        val_series = [series_list[i] for i in val_idx]
        print(f"    Train series: {len(tr_series)}, Val series: {len(val_series)}")

        # PatchTST
        print("    Training PatchTST...")
        nf_pt = train_neural_model({s: train_dict[s] for s in tr_series}, horizon, PatchTST, patchtst_params)
        if nf_pt is None:
            print("      train_neural_model returned None for PatchTST, skipping fold predictions")
            # Но мы всё равно продолжим, так как Theta может работать
        else:
            for s in val_series:
                print(f"      Predicting PatchTST for {s}...")
                pred = predict_neural_model(nf_pt, train_dict[s], s, horizon)
                if not np.any(np.isnan(pred)):
                    oof_pt[s] = pred
                    print(f"        success")
                else:
                    print(f"        failed (NaN)")

        # Theta
        print("    Training Theta...")
        for s in val_series:
            print(f"      Predicting Theta for {s}...")
            pred = train_theta(train_dict[s], horizon, freq, seasonality)
            if not np.any(np.isnan(pred)):
                oof_th[s] = pred
                print(f"        success")
            else:
                print(f"        failed (NaN)")

    valid_series = [s for s in series_list if oof_pt[s] is not None and oof_th[s] is not None]
    if len(valid_series) == 0:
        return None, None, None, None, None

    # Prepare meta features
    X_meta, y_meta = [], []
    for s in valid_series:
        feat = list(feat_dict[s].values())
        for t in range(horizon):
            X_meta.append([oof_pt[s][t], oof_th[s][t]] + feat)
            y_meta.append(oof_true[s][t])

    X_meta = np.array(X_meta)
    y_meta = np.array(y_meta)
    scaler = StandardScaler().fit(X_meta)
    X_scaled = scaler.transform(X_meta)

    meta_model = CatBoostRegressor(**meta_params)
    meta_model.fit(X_scaled, y_meta)

    # Retrain base models on full data
    nf_pt_full = train_neural_model(full_dict, horizon, PatchTST, patchtst_params)
    pt_test = {s: predict_neural_model(nf_pt_full, full_dict[s], s, horizon) for s in full_dict}
    th_test = {s: train_theta(full_dict[s], horizon, freq, seasonality) for s in full_dict}

    # Stacking predictions
    stacking_preds = {}
    for s in full_dict:
        if s not in valid_series:
            continue
        if np.any(np.isnan(pt_test[s])) or np.any(np.isnan(th_test[s])):
            continue
        feat = list(feat_dict[s].values())
        stack = np.zeros(horizon)
        ok = True
        for t in range(horizon):
            X_t = np.array([[pt_test[s][t], th_test[s][t]] + feat])
            X_t_scaled = scaler.transform(X_t)
            stack[t] = meta_model.predict(X_t_scaled)[0]
        if not np.any(np.isnan(stack)):
            stacking_preds[s] = stack

    return pt_test, th_test, stacking_preds, valid_series, scaler

def stacking_catboost_seasonalnaive(config, train_dict, test_dict, full_dict, series_info_list, freq, horizon):
    """
    Simple stacking with CatBoost and SeasonalNaive, using LinearRegression as meta-model.
    """
    from src.models import train_catboost_on_series, catboost_predict, seasonal_naive_forecast
    from sklearn.linear_model import LinearRegression

    X_stack, y_stack = [], []
    for sid, train_series in train_dict.items():
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

    if not X_stack:
        return None, None, None, None

    X_stack = np.array(X_stack)
    y_stack = np.array(y_stack)
    meta_model = LinearRegression().fit(X_stack, y_stack)

    # Retrain on full data
    cb_models_full = {}
    for sid, train_series in full_dict.items():
        model = train_catboost_on_series(train_series, horizon, config['catboost_params'])
        if model is not None:
            cb_models_full[sid] = model

    stacking_preds = {}
    for sid in full_dict:
        if sid not in cb_models_full:
            continue
        pred_cb = catboost_predict(cb_models_full[sid], full_dict[sid], horizon)
        if np.any(np.isnan(pred_cb)):
            continue
        pred_sn = seasonal_naive_forecast(full_dict[sid], horizon, config['seasonality'][freq])
        stack = np.zeros(horizon)
        for t in range(horizon):
            stack[t] = meta_model.predict([[pred_cb[t], pred_sn[t]]])[0]
        stacking_preds[sid] = stack

    return cb_models_full, stacking_preds, None, None  # dummy returns for compatibility

def stacking_neural(config, train_dict, test_dict, full_dict, freq, horizon):
    """
    Stacking with PatchTST and N-BEATS, Ridge meta-model per horizon.
    """
    from src.models import train_neural_model, predict_neural_model
    n_folds = config['n_folds_cv']
    pt_params = config['patchtst_params']
    nb_params = config['nbeats_params']

    series_list = list(train_dict.keys())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config['seed'])

    oof_pt = {s: None for s in series_list}
    oof_nb = {s: None for s in series_list}
    oof_true = {s: test_dict[s] for s in series_list}

    for fold, (tr_idx, val_idx) in enumerate(kf.split(series_list)):
        tr_series = [series_list[i] for i in tr_idx]
        val_series = [series_list[i] for i in val_idx]

        nf_pt = train_neural_model({s: train_dict[s] for s in tr_series}, horizon, PatchTST, pt_params)
        for s in val_series:
            pred = predict_neural_model(nf_pt, train_dict[s], s, horizon)
            if not np.any(np.isnan(pred)):
                oof_pt[s] = pred

        nf_nb = train_neural_model({s: train_dict[s] for s in tr_series}, horizon, NBEATS, nb_params)
        for s in val_series:
            pred = predict_neural_model(nf_nb, train_dict[s], s, horizon)
            if not np.any(np.isnan(pred)):
                oof_nb[s] = pred

    valid = [s for s in series_list if oof_pt[s] is not None and oof_nb[s] is not None]
    if not valid:
        return None, None, None, None

    # Meta models per horizon
    X_stack = [[] for _ in range(horizon)]
    y_stack = [[] for _ in range(horizon)]
    for s in valid:
        for t in range(horizon):
            X_stack[t].append([oof_pt[s][t], oof_nb[s][t]])
            y_stack[t].append(oof_true[s][t])

    meta_models = []
    for t in range(horizon):
        Xt = np.array(X_stack[t])
        yt = np.array(y_stack[t])
        if len(Xt) >= 2:
            meta_models.append(Ridge(alpha=1.0).fit(Xt, yt))
        else:
            meta_models.append(None)

    # Retrain on full
    nf_pt_full = train_neural_model(full_dict, horizon, PatchTST, pt_params)
    nf_nb_full = train_neural_model(full_dict, horizon, NBEATS, nb_params)

    pt_test = {s: predict_neural_model(nf_pt_full, full_dict[s], s, horizon) for s in full_dict}
    nb_test = {s: predict_neural_model(nf_nb_full, full_dict[s], s, horizon) for s in full_dict}

    stacking = {}
    for s in full_dict:
        if s not in valid:
            continue
        if np.any(np.isnan(pt_test[s])) or np.any(np.isnan(nb_test[s])):
            continue
        stack = np.zeros(horizon)
        ok = True
        for t in range(horizon):
            if meta_models[t] is not None:
                stack[t] = meta_models[t].predict([[pt_test[s][t], nb_test[s][t]]])[0]
            else:
                ok = False
                break
        if ok:
            stacking[s] = stack

    return pt_test, nb_test, stacking, valid