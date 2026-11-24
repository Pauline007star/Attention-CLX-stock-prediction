# Next-Session Close Forecasting

**Do Market Features Beat the Last Close?**

After Bank of China closes for the day, should a next-session forecasting model use recent trading activity and ARIMA forecast errors, or simply carry forward the closing price? This project tests both the value of those features within an LSTM and whether the resulting forecast earns its place over the simplest baseline.

**Decision on this benchmark.** Keep the previous close as the reference forecast. It had the lowest MAE and RMSE across all 180 shared test dates in each of three runs. Adding two PCA components lowered the LSTM’s average MAE by **1.49%** relative to the same architecture using market data alone. The PCA model still had **4.76% higher MAE** than carrying forward the last close.

| Forecast | Mean test MAE | Mean test RMSE |
|---|---:|---:|
| Last observed close | **0.01172** | **0.01645** |
| Expanding ARIMA(2,1,0) | 0.01197 | 0.01650 |
| LSTM with market data | 0.01247 | 0.01659 |
| LSTM with market data and six engineered factors | 0.01238 | 0.01655 |
| LSTM with market data and two factor PCs | 0.01228 | 0.01651 |

Neural metrics are averaged across seeds 7, 21 and 42. The full report includes the residual tree, the Attention CNN–BiLSTM backbone and per-run records. [Full results](results/RESULTS.md)

## What the comparison tests

1. **Does the feature set improve the LSTM?** Three variants share a 10-session window and a 24-unit LSTM. All predict the next close change, which is added to the last close for scoring. The input variants are OHLC, volume and trading amount alone, the same market fields with six engineered factors, and the market fields with two PCA components derived from those factors.
2. **Does the resulting forecast beat a cheap alternative?** Each variant is scored against the previous close and an expanding-window ARIMA on the same dates. A separate XGBoost experiment tests whether predicting one-step residuals improves fixed-parameter ARIMA. The PyTorch Attention backbone is also evaluated as a standalone price forecaster.

The six factors summarize daily return, distance from the five-session moving average, 10-session return volatility, intraday range relative to close, log-volume change and the observed ARIMA forecast error. The ablation evaluates the feature bundle. It does not isolate the value of any one factor.

## Evaluation protocol

- **Decision time.** Each forecast is issued after the previous session's OHLC, volume and amount are available. Every output row pairs an as-of date with the next observed trading session. As a new close arrives, it can enter subsequent one-step forecast windows.
- **Chronological split.** The CSV has 3,681 observations for 601988.SH, from 2007-01-04 through 2022-03-17. After warm-up there are 2,875 training targets, 354 validation targets starting 2020-01-02 and 180 test targets starting 2021-06-22. Validation MAE selects recurrent checkpoints and tree count.
- **Training-only transforms.** Feature scalers and PCA are fit on training inputs. The ARIMA residual feature uses parameters fit on the first 252 observations, then one-step predictions from prior closes. The realized residual enters the features only after the actual close is known. The expanding ARIMA baseline refits before each test forecast. [Statsmodels state-space methods](https://www.statsmodels.org/stable/statespace.html)

The scores describe one stock over a previously inspected historical window. The upstream README identifies TuShare as the source, and its adjustment settings are undocumented. The evaluation measures price error in the input CSV’s units. A trading-performance evaluation would require its own decision rule and execution assumptions.

## Run on CPU

From the repository root, using the verified Python 3.9.10 environment or a new Python 3.9 environment:

```sh
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

DATA="legacy/601988.SH.csv"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m forecasting --data "$DATA" \
  --output runs/smoke --epochs 1 --limit 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m forecasting --data "$DATA" \
  --output runs/seed42 --seed 42
```

Add `--baseline-only` for the three statistical baselines. The optional XGBoost run uses `python -m pip install -r requirements-xgboost.txt` and adds `--xgboost` to the run command. On macOS, a missing OpenMP library can be installed with `brew install libomp`. The verified Mac run used the OpenMP library already supplied with its PyTorch installation. [XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html)

Use a new output directory for each run. Every run writes a dated `predictions.csv` and `metrics.json` with the split, checkpoint, input checksum and library versions. To reproduce the reported tree row, use `--xgboost`. To repeat the neural ablation across all reported seeds, run with 7, 21 and 42 in separate directories.

## Project layout

| Location | Responsibility |
|---|---|
| `forecasting/data.py` | Input validation, causal factors and residuals, train-only PCA, date-labelled windows |
| `forecasting/models.py` | LSTM ablations, Attention backbone, validation checkpoint selection |
| `forecasting/trees.py` | Optional ARIMA residual correction |
| `forecasting/__main__.py` | Single command for training, scoring and saving dated results |
| `tests/` | Timing and leakage regression tests, neural output shape checks |
| `results/` | Verified summary and run metadata |
| `docs/` | Migration notes and comparison with the historical experiments |
| `legacy/` | Archived scripts, notebook, data and historical outputs |

[Migration notes](docs/MIGRATION.md) · [Historical README](legacy/README.md)

The benchmark reads the original market CSV from `legacy/`. The archived experiments retain their original dependency and path assumptions.

## Attribution

The historical fork added recurrent training changes, market-input comparisons and PCA experiment artifacts. This revision reorganizes those experiments and reruns them under one evaluation protocol. The Attention architecture is attributed to Shi, Hu, Mo and Wu. [Historical fork](https://github.com/Pauline007star/Attention-CLX-stock-prediction/tree/4948cca73b1dc789925d0e06faf18c6e3d4ef9b7) · [Upstream implementation](https://github.com/zshicode/Attention-CLX-stock-prediction) · [Paper](https://arxiv.org/abs/2204.02623)
