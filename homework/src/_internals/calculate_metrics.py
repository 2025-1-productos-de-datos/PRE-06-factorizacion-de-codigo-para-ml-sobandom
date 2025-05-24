# Metricas de error durante testing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(model, x, y):
    y_pred = model.predict(x)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, mae, r2
