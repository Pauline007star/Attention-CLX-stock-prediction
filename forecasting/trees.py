"""Train a tree on one-step ARIMA errors using lagged inputs only."""

import numpy as np
from xgboost import XGBRegressor


def fit_residual_tree(windows, targets, seed):
    is_train = targets["split"].eq("train")
    is_validation = targets["split"].eq("validation")
    is_test = targets["split"].eq("test")
    actual = targets["actual_close"].to_numpy()
    baseline = targets["arima_fixed"].to_numpy()
    lagged = windows.reshape(len(windows), -1)
    residual = actual - baseline
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=150,
        max_depth=2,
        learning_rate=0.03,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(lagged[is_train], residual[is_train])

    def validation_error(n_trees):
        correction = model.predict(lagged[is_validation], iteration_range=(0, n_trees))
        forecast = baseline[is_validation] + correction
        return float(np.abs(actual[is_validation] - forecast).mean())

    validation_errors = {n_trees: validation_error(n_trees) for n_trees in (25, 50, 100, 150)}
    best = min(validation_errors, key=validation_errors.get)
    correction = model.predict(lagged[is_test], iteration_range=(0, best))
    return baseline[is_test] + correction, {"validation_mae": validation_errors[best], "trees": best}
