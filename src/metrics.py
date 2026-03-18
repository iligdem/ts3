import numpy as np

def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        val = np.where(denominator != 0, 200 * diff / denominator, 0)
    return np.mean(val)

def mase(y_true, y_pred, y_train, seasonality):
    if seasonality == 1:
        scale = np.mean(np.abs(np.diff(y_train)))
    else:
        diff = y_train[seasonality:] - y_train[:-seasonality]
        scale = np.mean(np.abs(diff))
    if scale == 0:
        scale = 1.0
    mae = np.mean(np.abs(y_true - y_pred))
    return mae / scale

def msis(y_true, y_lower, y_upper, y_train, seasonality, alpha=0.05):
    if seasonality == 1:
        scale = np.mean(np.abs(np.diff(y_train)))
    else:
        diff = y_train[seasonality:] - y_train[:-seasonality]
        scale = np.mean(np.abs(diff))
    if scale == 0:
        scale = 1.0
    h = len(y_true)
    score = 0
    for t in range(h):
        if y_true[t] < y_lower[t]:
            score += (y_upper[t] - y_lower[t]) + (2/alpha) * (y_lower[t] - y_true[t])
        elif y_true[t] > y_upper[t]:
            score += (y_upper[t] - y_lower[t]) + (2/alpha) * (y_true[t] - y_upper[t])
        else:
            score += (y_upper[t] - y_lower[t])
    return (score / h) / scale