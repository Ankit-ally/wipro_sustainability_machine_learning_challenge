# ☀️ Solar Irradiance Prediction — MH Wipro Sustainable ML Challenge

> Predict three solar irradiance values (DHI, DNI, GHI) from weather data using ensemble machine learning. Built for the **MH Wipro Sustainable ML Challenge** on Kaggle.

---

## 🏆 Final Score

| Model | CV Score |
|-------|----------|
| Notebook 1 — Group Time K-Fold | ~261.97 |
| Notebook 2 — Stratified K-Fold (Lag Features) | ~279.00 |
| Notebook 3 — Lag + K-Fold | ~267.49 |
| **Ensemble (0.5 × N1 + 0.1 × N2 + 0.4 × N3)** | **~251.21** ✅ |

Lower is better (RMSE-based metric).

---

## 🎯 Problem Statement

Given hourly weather readings (temperature, wind speed, cloud type, pressure, etc.) for multiple years, predict three **clear-sky solar irradiance** values for each timestamp:

- **Clearsky DHI** — Diffuse Horizontal Irradiance
- **Clearsky DNI** — Direct Normal Irradiance
- **Clearsky GHI** — Global Horizontal Irradiance

---

## 📁 Project Structure

```
📦 wipro-solar-irradiance/
 ┣ 📓 mh-wipro-ml-frkgrouptimekf.ipynb     # Notebook 1: Group Time K-Fold
 ┣ 📓 mh-wipro-lag-notebook-kfolds.ipynb   # Notebook 2: Lag Features + K-Fold
 ┣ 📓 mh-wipro-notebook-stratfkf.ipynb     # Notebook 3: Stratified K-Fold
 ┣ 📓 ensemble.ipynb                        # Final: Blend all 3 predictions
 ┗ 📄 README.md
```

---

## 🔄 Pipeline Overview

```
Raw Data → Cleaning → Feature Engineering → Model Training → Predict → Ensemble → Submit
```

Each notebook follows this same flow but uses a different cross-validation strategy, producing diverse predictions that blend well.

---

## 🧹 Data Preprocessing (All Notebooks)

**1. Outlier Removal**
- Rows with `Cloud Type >= 10` are dropped (invalid readings)
- `Cloud Type = 1` is remapped to `0` (merged class)
- `Clearsky DHI > 400` values are capped at the 85th percentile of a reference day

**2. Outlier Clipping (Notebook 1)**
- For each weather column, values outside the 3rd–97th percentile per day are clipped using `np.clip`

---

## ⚙️ Feature Engineering

Features are created by grouping the data in different ways and computing statistics (min, max, mean):

| Group By | Prefix | What it captures |
|----------|--------|-----------------|
| Cloud Type + Day + Hour | `CDH_` | How weather varies by cloud type at each hour |
| Year + Month + Day | `YMD_` | Daily weather patterns |
| Cloud Type + Year + Month | `CYM_` | Monthly cloud-specific patterns |

**Time-based features added:**
- `day_part` — Splits the day into 5 parts (night, noon, afternoon, evening, late evening)
- `time_of_day` — Categorical label: dawn / morning / noon / afternoon / evening / midnight

**Lag Features (Notebook 2 & 3):**
- Previous year's target values shifted by 17,520 rows (one full year of 30-min data)
- Rolling averages on lagged targets to smooth predictions

---

## 🤖 Model

All notebooks use **LightGBM (LGBM Regressor)** — a fast gradient boosting framework well-suited for tabular data.

Three separate models are trained, one per target:
- Model A → predicts `Clearsky DHI`
- Model B → predicts `Clearsky DNI`
- Model C → predicts `Clearsky GHI`

---

## 🔀 Cross-Validation Strategies

Each notebook uses a different CV approach to produce diverse predictions:

| Notebook | Strategy | Why it helps |
|----------|----------|-------------|
| `mh-wipro-ml-frkgrouptimekf` | **Group Time K-Fold** | Respects time order; avoids data leakage |
| `mh-wipro-lag-notebook-kfolds` | **K-Fold + Lag Features** | Adds past-year context to each row |
| `mh-wipro-notebook-stratfkf` | **Stratified K-Fold** | Balances target distribution across folds |

---

## 🎛️ Ensemble (Final Step)

The `ensemble.ipynb` notebook blends the three submission CSVs using **weighted averaging**:

```python
final = (submission_1 * 0.5) + (submission_2 * 0.1) + (submission_3 * 0.4)
```

| Weight | Notebook | Reason |
|--------|----------|--------|
| 0.5 | Group Time K-Fold (best solo score) | Highest individual performance |
| 0.4 | Lag K-Fold | Strong lag signal, diverse predictions |
| 0.1 | Stratified K-Fold | Small contribution, adds diversity |

This blending reduced the score from ~261 (best solo) to **~251** — a meaningful improvement.

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn lightgbm pvlib scipy tqdm
```

### Steps

1. Download the dataset from [Kaggle — MH Wipro Sustainable ML Challenge](https://www.kaggle.com/competitions/mh-wipro-sustainable-ml-challenge)
2. Place `train.csv`, `test.csv`, `sample_submission.csv` in your working directory
3. Run the three model notebooks in any order:
   - `mh-wipro-ml-frkgrouptimekf.ipynb`
   - `mh-wipro-lag-notebook-kfolds.ipynb`
   - `mh-wipro-notebook-stratfkf.ipynb`
4. Each notebook saves a submission CSV
5. Run `ensemble.ipynb` pointing to those three CSV files to generate the final blended submission

---

## 📊 Input Features

| Feature | Description |
|---------|-------------|
| Year, Month, Day, Hour, Minute | Timestamp breakdown |
| Temperature | Air temperature (°C) |
| Dew Point | Moisture indicator |
| Relative Humidity | Humidity (%) |
| Pressure | Atmospheric pressure |
| Wind Speed / Direction | Wind readings |
| Solar Zenith Angle | Sun angle above horizon |
| Precipitable Water | Water vapor column |
| Cloud Type | Cloud classification (0–9) |
| Fill Flag | Data quality flag |

**Targets:** `Clearsky DHI`, `Clearsky DNI`, `Clearsky GHI`

---

## 💡 Key Learnings for Beginners

- **Ensemble > Single model** — blending three different CV strategies reduced error by ~10 points
- **Lag features work well for time series** — knowing last year's value at the same timestamp is very informative
- **Respect time in cross-validation** — using Group Time K-Fold prevents "future leakage" into training folds
- **Outlier handling matters** — removing bad `Cloud Type` rows and capping extreme DHI values significantly cleaned the signal

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.7+-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-green)
![Pandas](https://img.shields.io/badge/Pandas-data%20processing-lightblue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-cross%20validation-orange)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF)

---

## 👤 Author

**Mr. Ankit Jitendra Kumar Lalo Sharma**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/ankit-sharma-39b1881a0)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Ankit-ally)

---

## 📄 License

This project is open for learning and reference. Dataset belongs to the Wipro ML Challenge organizers.
