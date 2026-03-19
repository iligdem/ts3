import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from catboost import CatBoostRegressor
from sktime.forecasting.theta import ThetaForecaster
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST, NBEATS

# Feature creation for CatBoost
FEATURE_KEYS = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'mean', 'std']

def create_features(series, max_lag=6):
    features = {}
    for lag in range(1, max_lag+1):
        features[f'lag_{lag}'] = series[-lag] if len(series) >= lag else np.nan
    features['mean'] = np.mean(series)
    features['std'] = np.std(series)
    return [features[key] for key in FEATURE_KEYS]

# ----- CatBoost helpers -----
def prepare_catboost_data(series_dict, horizon, max_lag=6):
    X, y, meta = [], [], []
    for sid, train_series in series_dict.items():
        for t in range(max_lag, len(train_series) - horizon + 1):
            feats = create_features(train_series[t-max_lag:t], max_lag=max_lag)
            if np.any(np.isnan(feats)):
                continue
            X.append(feats)
            y.append(train_series[t:t+horizon])
            meta.append((sid, t))
    return np.array(X, dtype=float), np.array(y, dtype=float), meta

def train_catboost_global(X, y, params):
    if len(X) == 0:
        return None
    model = CatBoostRegressor(**params)
    model.fit(X, y)
    return model


def catboost_predict_global(model, train_series, horizon, max_lag=6):
    if len(train_series) < max_lag:
        return np.full(horizon, np.nan)
    feats = create_features(train_series[-max_lag:], max_lag=max_lag)
    if np.any(np.isnan(feats)):
        return np.full(horizon, np.nan)
    X = np.array([feats], dtype=float)
    pred = model.predict(X)[0]
    return pred[:horizon]

def train_catboost_on_series(train_series, horizon, params):
    X, y = [], []
    max_lag = 6
    if len(train_series) < max_lag + horizon:
        return None
    for t in range(max_lag, len(train_series) - horizon + 1):
        X.append(create_features(train_series[t-max_lag:t], max_lag=max_lag))
        y.append(train_series[t:t+horizon])
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    if len(X) == 0 or np.any(np.isnan(X)) or np.any(np.isnan(y)):
        return None
    if np.all(np.std(X, axis=0) == 0):
        return None
    try:
        model = CatBoostRegressor(**params)
        model.fit(X, y)
        return model
    except Exception:
        return None

def catboost_predict(model, train_series, horizon):
    return catboost_predict_global(model, train_series, horizon)

# ----- Neural models helpers -----
def train_neural_model(train_dict, horizon, model_class, params):
    rows = []
    for sid, vals in train_dict.items():
        for t, v in enumerate(vals):
            rows.append({'unique_id': sid, 'ds': t, 'y': v})
    df = pd.DataFrame(rows)
    try:
        nf = NeuralForecast(models=[model_class(h=horizon, **params)], freq=1)
        nf.fit(df=df)
        print(f"      train_neural_model: success, model {model_class.__name__} trained on {len(train_dict)} series")
        return nf
    except Exception as e:
        print(f"      train_neural_model exception: {e}")
        return None
def predict_neural_model(nf, train_series, sid, horizon):
    print(f"        predict_neural_model: sid={sid}, len(train_series)={len(train_series)}, horizon={horizon}")
    y_tr = pd.Series(train_series).astype(float)
    new_df = pd.DataFrame({
        'unique_id': [sid] * len(train_series),
        'ds': range(len(train_series)),
        'y': y_tr
    })
    print(f"        new_df shape: {new_df.shape}, head:\n{new_df.head(2)}")
    try:
        preds = nf.predict(df=new_df)
        print(f"        preds type: {type(preds)}")
        if preds is not None:
            print(f"        preds shape: {preds.shape if hasattr(preds, 'shape') else 'N/A'}")
            print(f"        preds columns: {preds.columns.tolist() if hasattr(preds, 'columns') else 'N/A'}")
        else:
            print(f"        preds is None")
            return np.full(horizon, np.nan)
        model_name = nf.models[0].__class__.__name__
        print(f"        model_name: {model_name}")
        if not preds.empty and model_name in preds.columns:
            vals = preds[model_name].values[:horizon]
            print(f"        vals shape: {vals.shape}, first few: {vals[:3]}")
            if len(vals) == horizon and not np.any(np.isnan(vals)):
                print(f"        success for {sid}")
                return vals
            else:
                print(f"        invalid vals: len={len(vals)}, has NaN={np.any(np.isnan(vals))}")
        else:
            print(f"        empty preds or missing column")
    except Exception as e:
        print(f"        EXCEPTION in predict_neural_model for {sid}: {e}")
        import traceback
        traceback.print_exc()
    return np.full(horizon, np.nan)

# Специализированные обёртки для PatchTST
def train_patchtst_global(train_dict, horizon, params):
    """Обучает глобальную модель PatchTST через train_neural_model."""
    return train_neural_model(train_dict, horizon, PatchTST, params)

def predict_patchtst(nf, train_series, sid, horizon):
    """Предсказание с помощью обученной модели PatchTST через predict_neural_model."""
    return predict_neural_model(nf, train_series, sid, horizon)

# ----- Theta helpers -----
def train_theta(series, horizon, freq, seasonality):
    try:
        sp = seasonality[freq]
        forecaster = ThetaForecaster(sp=sp)
        forecaster.fit(series)
        pred = forecaster.predict(fh=np.arange(1, horizon+1))
        # pred уже numpy array, не вызываем .values
        if np.any(np.isnan(pred)):
            print(f"      train_theta: NaN in prediction for series of length {len(series)}, freq {freq}, using fallback")
            return np.full(horizon, series[-1])
        return pred
    except Exception as e:
        print(f"      train_theta exception: {e}, using fallback")
        return np.full(horizon, series[-1])

# ----- ETS helpers -----
def train_ets(series, horizon, freq, seasonality):
    try:
        if freq == 'Yearly':
            model = ExponentialSmoothing(series, trend=True, seasonal=None,
                                         initialization_method='estimated')
        else:
            model = ExponentialSmoothing(series, trend=True, seasonal='add',
                                         seasonal_periods=seasonality[freq],
                                         initialization_method='estimated')
        fit = model.fit()
        forecast = fit.forecast(horizon)
        return forecast.values
    except Exception:
        return np.full(horizon, series[-1])

# ----- Naive / SeasonalNaive -----
def naive_forecast(train, horizon):
    return np.full(horizon, train[-1])

def seasonal_naive_forecast(train, horizon, seasonality):
    if seasonality == 1:
        return naive_forecast(train, horizon)
    last_season = train[-seasonality:]
    repeats = (horizon // seasonality) + 1
    return np.tile(last_season, repeats)[:horizon]