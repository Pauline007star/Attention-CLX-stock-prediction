# Can market features beat the last close?

**The previous close had the lowest MAE and RMSE in every run.** The comparison covers 180 next-session close forecasts for 601988.SH from 2021-06-22 to 2022-03-17. All models are scored on the same dates. Recurrent rows report the mean of each metric across seeds 7, 21 and 42. The statistical baselines and residual tree returned identical metrics across those runs.

| Model | Mean MAE | Mean RMSE | Test MAE range across runs |
|---|---:|---:|---:|
| Last close | 0.01172 | 0.01645 | 0.01172 |
| Expanding ARIMA(2,1,0) | 0.01197 | 0.01650 | 0.01197 |
| Fixed-parameter ARIMA(2,1,0) | 0.01207 | 0.01655 | 0.01207 |
| LSTM, market | 0.01247 | 0.01659 | 0.01227–0.01258 |
| LSTM, market + six factors | 0.01238 | 0.01655 | 0.01225–0.01254 |
| LSTM, market + two factor PCs | 0.01228 | 0.01651 | 0.01211–0.01246 |
| Fixed ARIMA + XGBoost residual | 0.01210 | 0.01648 | 0.01210 |
| Attention CNN–BiLSTM backbone | 0.02038 | 0.02565 | 0.01523–0.02989 |

## What changed when the inputs changed

- **Compared with the same LSTM.** Adding two PCA components reduced average MAE from 0.01247 to 0.01228, a **1.49%** reduction calculated from the unrounded means. The reduction within individual seeds ranged from **0.02% to 3.45%**. The full six-factor inputs had mixed effects across seeds.
- **Compared with the forecast already available at the close.** The PCA LSTM had **4.76% higher MAE** than carrying forward the last close, which scored 0.01172. On this benchmark, the measured feature improvement does not justify switching forecasts based on price error.
- **Compared with the statistical model being corrected.** The XGBoost residual correction slightly reduced RMSE and slightly increased MAE versus its fixed-parameter ARIMA baseline. The choice of error metric matters when those results are interpreted.

The range across seeds describes sensitivity to initialization on the same dates. It does not estimate uncertainty across market periods. The data cover one stock and an evaluation window already inspected during the historical project.

The saved runs include sample counts, model selection, dependency versions and the input SHA-256. [Seed 7](seed-7.json) · [Seed 21](seed-21.json) · [Seed 42](seed-42.json)
