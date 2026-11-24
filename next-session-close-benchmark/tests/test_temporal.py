import unittest

import numpy as np
import pandas as pd
import torch

from forecasting.data import build_dataset
from forecasting.models import AttentionBackbone, CloseChangeLSTM, fit_forecast


def sample_market():
    generator = np.random.default_rng(123)
    close = 10 + generator.normal(0, 0.15, 190).cumsum()
    return pd.DataFrame({
        "open": close * 1.002,
        "high": close * 1.03,
        "low": close * 0.97,
        "close": close,
        "vol": generator.uniform(100, 300, len(close)),
        "amount": generator.uniform(5000, 15000, len(close)),
    }, index=pd.bdate_range("2010-01-01", periods=len(close), name="trade_date"))


class TemporalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.market = sample_market()
        self.validation = self.market.index[105]
        self.test = self.market.index[145]

    def test_first_test_window_uses_only_known_history(self):
        data = build_dataset(self.market, self.validation, self.test, warmup=30)
        row = data.targets.index.get_loc(self.test)
        self.assertEqual(data.targets.iloc[row]["as_of_date"], self.market.index[144])
        self.assertEqual(data.targets.iloc[row]["split"], "test")
        np.testing.assert_array_equal(
            data.windows["lstm_market"][row], self.market.iloc[135:145].to_numpy(dtype=np.float32)
        )

    def test_future_changes_cannot_modify_earlier_windows_or_pca_fit(self):
        baseline = build_dataset(self.market, self.validation, self.test, warmup=30)
        altered = self.market.copy()
        altered.loc[self.test:] *= 3
        repeated = build_dataset(altered, self.validation, self.test, warmup=30)
        first_test = baseline.targets.index.get_loc(self.test)
        for name in baseline.windows:
            with self.subTest(name=name):
                np.testing.assert_allclose(
                    baseline.windows[name][:first_test + 1],
                    repeated.windows[name][:first_test + 1],
                    rtol=1e-6,
                    atol=1e-7,
                )
        np.testing.assert_array_equal(baseline.pca_variance_ratio, repeated.pca_variance_ratio)
        self.assertAlmostEqual(
            baseline.targets.loc[self.test, "arima_fixed"],
            repeated.targets.loc[self.test, "arima_fixed"],
        )
        self.assertNotEqual(
            baseline.targets.loc[self.test, "actual_close"],
            repeated.targets.loc[self.test, "actual_close"],
        )

    def test_rejects_test_before_validation(self):
        with self.assertRaisesRegex(ValueError, "validation-then-test"):
            build_dataset(self.market, self.test, self.validation, warmup=30)

    def test_held_out_values_cannot_change_checkpoint_or_first_prediction(self):
        torch.set_num_threads(1)
        baseline = build_dataset(self.market, self.validation, self.test, warmup=30)
        altered = self.market.copy()
        altered.loc[self.test:] *= 3
        repeated = build_dataset(altered, self.validation, self.test, warmup=30)
        original, original_checkpoint = fit_forecast(
            "lstm_pca", baseline.windows["lstm_pca"], baseline.targets, epochs=1, seed=7
        )
        changed, changed_checkpoint = fit_forecast(
            "lstm_pca", repeated.windows["lstm_pca"], repeated.targets, epochs=1, seed=7
        )
        self.assertEqual(original_checkpoint, changed_checkpoint)
        self.assertAlmostEqual(original[0], changed[0])

    def test_recurrent_output_shapes(self):
        torch.set_num_threads(1)
        self.assertEqual(CloseChangeLSTM(6)(torch.zeros(3, 10, 6)).shape, (3, 1))
        prediction = AttentionBackbone(7, 20)(torch.zeros(3, 20, 7))
        self.assertEqual(prediction.shape, (3, 1))
        self.assertTrue(((prediction > 0) & (prediction < 1)).all().item())


if __name__ == "__main__":
    unittest.main()
