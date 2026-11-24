# Next-Session Close Forecasting

**Do Market Features Beat the Last Close?**

This project tests whether market indicators and recent ARIMA forecast errors improve next-session closing-price forecasts for Bank of China. On 180 shared test dates, adding two PCA components lowered average LSTM MAE by **1.49%** versus the same architecture using market data alone. The previous-close baseline still had the lowest MAE and RMSE in all three runs.

| Forecast | Test MAE | Test RMSE |
|---|---:|---:|
| Last observed close | **0.01172** | **0.01645** |
| LSTM with historical market data | 0.01247 | 0.01659 |
| LSTM with market data and two PCA components | 0.01228 | 0.01651 |

The LSTM metrics are means across three training seeds on one historical test window. The benchmark uses chronological validation and training-only preprocessing.

[Read the experiment and run it on CPU](next-session-close-benchmark/README.md) · [Inspect the full results and run records](next-session-close-benchmark/results/RESULTS.md)

The original fork records changes to recurrent training and adds experiments with market inputs, ARIMA residuals and PCA features. The revised benchmark evaluates a shared next-close target with time-aligned inputs. The Attention-CLX architecture is credited to Shi, Hu, Mo and Wu. [Upstream implementation](https://github.com/zshicode/Attention-CLX-stock-prediction) · [Paper](https://arxiv.org/abs/2204.02623)

<details>
<summary>Original Attention-CLX documentation</summary>

## Attention-based CNN-LSTM and XGBoost hybrid model for stock prediction

### Requirements

The code has been tested running under Python 3.7.4, with the following packages and their dependencies installed:
```
numpy==1.16.5
sklearn==0.21.3
statsmodels==0.10.1
pandas==0.25.1
tensorflow==2.1.0
keras==2.3.1
xgboost==1.5.0
```

The stock data used in this repository was downloaded from [TuShare](https://tushare.pro/). The stock data on [TuShare](https://tushare.pro/) are with public availability.

### Usage

Firstly, run `ARIMA.py` for pre-processing step by ARIMA model. Then, run the neural network or XGBoost models.
- Run `LSTM.py` for the single-layer LSTM, multi-layer LSTM, and bidirectional LSTM models.
- Run `XGBoost.py` for the XGBoost model.
- Run `Main.py` for our proposed Attention-based CNN-LSTM and XGBoost hybrid model.

### Citation
```
@article{shi2022attclx,
    author={Zhuangwei Shi and Yang Hu and Guangliang Mo and Jian Wu},
    title={Attention-based CNN-LSTM and XGBoost hybrid model for stock prediction},
    journal={arXiv preprint arXiv:2204.02623},
    year={2022},
}
```

</details>
