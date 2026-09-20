"""
hyderabad_aqi_pipeline.py
=========================
All-in-one Hyderabad AQI analysis pipeline.

Usage
-----
Run all steps:
    python hyderabad_aqi_pipeline.py

Run a single step:
    python hyderabad_aqi_pipeline.py --step preprocess
    python hyderabad_aqi_pipeline.py --step eda
    python hyderabad_aqi_pipeline.py --step models
    python hyderabad_aqi_pipeline.py --step forecast
    python hyderabad_aqi_pipeline.py --step shap
    python hyderabad_aqi_pipeline.py --step summary

Dataset
-------
  data/raw/city_day.csv      – city-level daily AQI (all Indian cities)
  data/raw/station_day.csv   – station-level daily readings
"""

# ---------------------------------------------------------------------------
# stdlib
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import argparse
import pickle
from datetime import datetime, timedelta

# Force UTF-8 on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# third-party
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
RAW_CITY_CSV    = os.path.join("data", "raw", "city_day.csv")
RAW_STATION_CSV = os.path.join("data", "raw", "station_day.csv")
CLEAN_CSV       = os.path.join("data", "processed", "hyderabad_clean.csv")
FIG_DIR         = os.path.join("outputs", "figures")
MODEL_DIR       = os.path.join("outputs", "models")
REPORT_DIR      = os.path.join("outputs", "reports")

for _d in [os.path.dirname(CLEAN_CSV), FIG_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(_d, exist_ok=True)

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
              "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]

CPCB_BINS   = [0, 50, 100, 200, 300, 400, float("inf")]
CPCB_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
CPCB_COLORS = {
    "Good":         "#27ae60",
    "Satisfactory": "#2ecc71",
    "Moderate":     "#f39c12",
    "Poor":         "#e67e22",
    "Very Poor":    "#e74c3c",
    "Severe":       "#8e44ad",
}

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _savefig(name: str):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"  [fig] {path}")


def classify_aqi(v):
    if pd.isna(v):
        return "Unknown"
    for lo, hi, label in zip(CPCB_BINS, CPCB_BINS[1:], CPCB_LABELS):
        if lo <= v < hi:
            return label
    return "Severe"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ===========================================================================
# STEP 1 — Preprocess
# ===========================================================================

def step_preprocess() -> pd.DataFrame:
    print("\n=== STEP 1: Preprocess ===")

    city = pd.read_csv(RAW_CITY_CSV, parse_dates=["Date"])
    hyd  = city[city["City"].str.strip().str.lower() == "hyderabad"].copy()
    print(f"  Raw Hyderabad rows (city_day): {len(hyd)}")

    # Supplement with station data if available
    if os.path.exists(RAW_STATION_CSV):
        stn = pd.read_csv(RAW_STATION_CSV, parse_dates=["Date"])
        # Only keep rows whose StationId or a City column suggests Hyderabad
        hyd_stn = stn[stn["StationId"].str.upper().str.startswith(("HY", "TS", "AP"))].copy() \
            if "StationId" in stn.columns else pd.DataFrame()
        if len(hyd_stn):
            # Average across stations per day
            cols = ["Date"] + [c for c in POLLUTANTS + ["AQI"] if c in hyd_stn.columns]
            daily_stn = hyd_stn[cols].groupby("Date").mean(numeric_only=True).reset_index()
            print(f"  Station supplement rows: {len(daily_stn)}")
            # Merge to fill gaps
            hyd = hyd.drop(columns=["City"], errors="ignore")
            hyd = pd.merge(hyd, daily_stn, on="Date", how="outer", suffixes=("", "_stn"))
            for col in POLLUTANTS + ["AQI"]:
                if col in hyd.columns and f"{col}_stn" in hyd.columns:
                    hyd[col] = hyd[col].fillna(hyd[f"{col}_stn"])
            hyd.drop(columns=[c for c in hyd.columns if c.endswith("_stn")], inplace=True)

    hyd = hyd.sort_values("Date").reset_index(drop=True)

    # Drop AQI_Bucket if present — we'll recompute
    hyd.drop(columns=["AQI_Bucket", "City"], errors="ignore", inplace=True)

    # Interpolate numeric columns (linear, then forward/back fill boundaries)
    num_cols = [c for c in hyd.columns if c != "Date"]
    hyd[num_cols] = hyd[num_cols].interpolate(method="linear", limit_direction="both")
    hyd[num_cols] = hyd[num_cols].ffill().bfill()

    # Clamp negative values to 0
    for c in num_cols:
        hyd[c] = hyd[c].clip(lower=0)

    # Recompute AQI_Bucket
    hyd["AQI_Bucket"] = hyd["AQI"].apply(classify_aqi)

    # Date features
    hyd["Year"]    = hyd["Date"].dt.year
    hyd["Month"]   = hyd["Date"].dt.month
    hyd["DayOfYear"] = hyd["Date"].dt.dayofyear
    hyd["DayOfWeek"] = hyd["Date"].dt.dayofweek

    # Lag & rolling features for AQI
    for lag in [1, 3, 7]:
        hyd[f"AQI_lag{lag}"] = hyd["AQI"].shift(lag)
    hyd["AQI_roll7"]  = hyd["AQI"].rolling(7,  min_periods=1).mean()
    hyd["AQI_roll30"] = hyd["AQI"].rolling(30, min_periods=1).mean()

    hyd.dropna(subset=["AQI"], inplace=True)
    hyd.reset_index(drop=True, inplace=True)

    hyd.to_csv(CLEAN_CSV, index=False)
    print(f"  Clean dataset saved → {CLEAN_CSV}  ({len(hyd)} rows, {hyd['Date'].min().date()} – {hyd['Date'].max().date()})")
    return hyd


# ===========================================================================
# STEP 2 — EDA
# ===========================================================================

def step_eda(df: pd.DataFrame):
    print("\n=== STEP 2: EDA ===")

    # --- 2a. AQI time-series ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["Date"], df["AQI"], lw=0.8, color="#3498db", alpha=0.7, label="Daily AQI")
    ax.plot(df["Date"], df["AQI_roll30"], lw=2, color="#e74c3c", label="30-day MA")
    ax.set_title("Hyderabad Daily AQI (2015–2020)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig("01_aqi_timeseries.png")

    # --- 2b. AQI category distribution (pie) ---
    cat_counts = df["AQI_Bucket"].value_counts().reindex(CPCB_LABELS, fill_value=0)
    colors     = [CPCB_COLORS.get(l, "#95a5a6") for l in cat_counts.index]
    fig, ax    = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        cat_counts, labels=cat_counts.index, colors=colors,
        autopct="%1.1f%%", startangle=140, pctdistance=0.82)
    ax.set_title("AQI Category Distribution", fontsize=13)
    _savefig("02_aqi_category_pie.png")

    # --- 2c. Monthly AQI box-plot ---
    fig, ax = plt.subplots(figsize=(12, 5))
    month_data = [df[df["Month"] == m]["AQI"].dropna().values for m in range(1, 13)]
    bp = ax.boxplot(month_data, patch_artist=True, notch=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#aed6f1")
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_title("Monthly AQI Distribution", fontsize=13)
    ax.set_ylabel("AQI")
    ax.grid(axis="y", alpha=0.3)
    _savefig("03_monthly_aqi_boxplot.png")

    # --- 2d. Yearly average AQI bar ---
    yr_avg = df.groupby("Year")["AQI"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(yr_avg.index.astype(str), yr_avg.values, color="#3498db", edgecolor="white")
    for b, v in zip(bars, yr_avg.values):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_title("Average AQI by Year", fontsize=13)
    ax.set_ylabel("AQI")
    ax.grid(axis="y", alpha=0.3)
    _savefig("04_yearly_avg_aqi.png")

    # --- 2e. Pollutant correlation heatmap ---
    corr_cols = [c for c in POLLUTANTS if c in df.columns]
    fig, ax   = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Pollutant Correlation Matrix", fontsize=13)
    _savefig("05_pollutant_correlation.png")

    # --- 2f. Pollutant trends (small multiples) ---
    available = [c for c in POLLUTANTS if c in df.columns]
    ncols = 3
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3), sharex=False)
    axes = axes.flatten()
    for i, col in enumerate(available):
        axes[i].plot(df["Date"], df[col], lw=0.6, color="#3498db", alpha=0.7)
        roll = df[col].rolling(30, min_periods=1).mean()
        axes[i].plot(df["Date"], roll, lw=1.5, color="#e74c3c")
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(axis="x", rotation=30, labelsize=7)
        axes[i].grid(alpha=0.2)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Pollutant Trends (Blue=Daily, Red=30-day MA)", fontsize=12, y=1.01)
    plt.tight_layout()
    _savefig("06_pollutant_trends.png")

    print("  EDA figures saved.")


# ===========================================================================
# STEP 3 — Models
# ===========================================================================

FEATURE_COLS = [
    "Month", "DayOfYear", "DayOfWeek", "Year",
    "PM2.5", "PM10", "NO2", "SO2", "O3", "CO", "NH3",
    "AQI_lag1", "AQI_lag3", "AQI_lag7",
    "AQI_roll7", "AQI_roll30",
]

def _get_xy(df: pd.DataFrame):
    feat = [c for c in FEATURE_COLS if c in df.columns]
    sub  = df.dropna(subset=feat + ["AQI"])
    X    = sub[feat].values
    y    = sub["AQI"].values
    return X, y, feat, sub


def _eval(name, y_true, y_pred, results: dict):
    r = {
        "RMSE":  round(rmse(y_true, y_pred), 3),
        "MAE":   round(float(mean_absolute_error(y_true, y_pred)), 3),
        "R2":    round(float(r2_score(y_true, y_pred)), 4),
    }
    results[name] = r
    print(f"    {name:30s}  RMSE={r['RMSE']:.2f}  MAE={r['MAE']:.2f}  R²={r['R2']:.4f}")
    return r


def step_models(df: pd.DataFrame) -> dict:
    print("\n=== STEP 3: Models ===")

    X, y, feat, sub = _get_xy(df)
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}
    best_models = {}

    models = {
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest":    RandomForestRegressor(n_estimators=200, max_depth=10,
                                                   random_state=42, n_jobs=-1),
        "XGBoost":          xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                              max_depth=6, subsample=0.8,
                                              colsample_bytree=0.8, random_state=42,
                                              verbosity=0),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                        max_depth=5, random_state=42),
    }

    for name, model in models.items():
        preds_all, true_all = [], []
        for tr, te in tscv.split(X):
            model.fit(X[tr], y[tr])
            preds_all.extend(model.predict(X[te]))
            true_all.extend(y[te])
        _eval(name, np.array(true_all), np.array(preds_all), results)

        # Refit on full data and save
        model.fit(X, y)
        best_models[name] = model
        slug = name.lower().replace(" ", "_")
        with open(os.path.join(MODEL_DIR, f"{slug}.pkl"), "wb") as fh:
            pickle.dump({"model": model, "features": feat}, fh)

    # Save metrics
    with open(os.path.join(REPORT_DIR, "model_metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Metrics → {REPORT_DIR}/model_metrics.json")

    # --- Model comparison bar chart ---
    names  = list(results.keys())
    rmses  = [results[n]["RMSE"] for n in names]
    r2s    = [results[n]["R2"]   for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bars1 = ax1.barh(names, rmses, color="#3498db")
    ax1.set_xlabel("RMSE (lower = better)")
    ax1.set_title("Model Comparison — RMSE")
    for b, v in zip(bars1, rmses):
        ax1.text(v + 0.3, b.get_y() + b.get_height()/2, f"{v:.2f}", va="center", fontsize=9)

    bars2 = ax2.barh(names, r2s, color="#27ae60")
    ax2.set_xlabel("R² (higher = better)")
    ax2.set_title("Model Comparison — R²")
    for b, v in zip(bars2, r2s):
        ax2.text(v + 0.002, b.get_y() + b.get_height()/2, f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    _savefig("07_model_comparison.png")

    # --- Actual vs Predicted (best model by R²) ---
    best_name = max(results, key=lambda n: results[n]["R2"])
    best_m    = best_models[best_name]
    y_hat     = best_m.predict(X)
    fig, ax   = plt.subplots(figsize=(6, 6))
    ax.scatter(y, y_hat, alpha=0.3, s=10, color="#3498db")
    lims = [min(y.min(), y_hat.min()) - 5, max(y.max(), y_hat.max()) + 5]
    ax.plot(lims, lims, "r--", lw=1.5)
    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    ax.set_title(f"Actual vs Predicted — {best_name}")
    ax.grid(alpha=0.3)
    _savefig("08_actual_vs_predicted.png")

    return {"results": results, "best_models": best_models, "features": feat}


# ===========================================================================
# STEP 4 — Forecast (30-day rolling XGBoost)
# ===========================================================================

def step_forecast(df: pd.DataFrame, model_info: dict) -> pd.DataFrame:
    print("\n=== STEP 4: Forecast (30-day) ===")

    feat       = model_info["features"]
    best_m     = model_info["best_models"].get("XGBoost") or \
                 next(iter(model_info["best_models"].values()))

    last_row   = df.sort_values("Date").iloc[-1].copy()
    last_date  = pd.to_datetime(last_row["Date"])
    history    = df["AQI"].tolist()

    forecast_rows = []
    for i in range(1, 31):
        next_date = last_date + timedelta(days=i)
        row = {
            "Date":      next_date,
            "Month":     next_date.month,
            "DayOfYear": next_date.timetuple().tm_yday,
            "DayOfWeek": next_date.weekday(),
            "Year":      next_date.year,
        }
        # Use seasonal average for pollutant features
        same_month = df[df["Month"] == next_date.month]
        for col in ["PM2.5","PM10","NO2","SO2","O3","CO","NH3"]:
            if col in feat:
                row[col] = same_month[col].mean() if col in df.columns else 0.0

        n = len(history)
        row["AQI_lag1"]   = history[-1]      if n >= 1 else 0
        row["AQI_lag3"]   = history[-3]      if n >= 3 else history[0]
        row["AQI_lag7"]   = history[-7]      if n >= 7 else history[0]
        row["AQI_roll7"]  = float(np.mean(history[-7:]))  if n >= 7  else float(np.mean(history))
        row["AQI_roll30"] = float(np.mean(history[-30:])) if n >= 30 else float(np.mean(history))

        X_fc = np.array([[row.get(f, 0.0) for f in feat]])
        pred = float(best_m.predict(X_fc)[0])
        pred = max(0.0, pred)
        row["AQI_forecast"] = pred
        history.append(pred)
        forecast_rows.append(row)

    fc_df = pd.DataFrame(forecast_rows)
    fc_df["AQI_Bucket"] = fc_df["AQI_forecast"].apply(classify_aqi)
    fc_df.to_csv(os.path.join(REPORT_DIR, "forecast_30day.csv"), index=False)
    print(f"  Forecast saved → {REPORT_DIR}/forecast_30day.csv")

    # --- Plot ---
    hist_plot = df.tail(90)
    fig, ax   = plt.subplots(figsize=(13, 5))
    ax.plot(hist_plot["Date"], hist_plot["AQI"],
            lw=1.2, color="#3498db", label="Historical AQI")
    ax.plot(fc_df["Date"], fc_df["AQI_forecast"],
            lw=2, color="#e74c3c", linestyle="--", label="30-day Forecast")
    ax.axvline(x=last_date, color="grey", linestyle=":", lw=1.5)
    ax.fill_between(fc_df["Date"],
                    fc_df["AQI_forecast"] * 0.9,
                    fc_df["AQI_forecast"] * 1.1,
                    alpha=0.15, color="#e74c3c", label="±10% band")
    ax.set_title("Hyderabad AQI — Historical & 30-day Forecast", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig("09_forecast_30day.png")

    return fc_df


# ===========================================================================
# STEP 5 — SHAP feature importance
# ===========================================================================

def step_shap(df: pd.DataFrame, model_info: dict):
    print("\n=== STEP 5: SHAP Feature Importance ===")
    try:
        import shap
    except ImportError:
        print("  shap not installed — skipping.")
        return

    feat  = model_info["features"]
    model = model_info["best_models"].get("XGBoost") or \
            next(iter(model_info["best_models"].values()))

    X, y, _, _ = _get_xy(df)
    sample_X = X[:500] if len(X) > 500 else X

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_X)

    # Bar summary
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    feat_arr = np.array(feat)
    ax.barh(feat_arr[order][::-1], mean_abs[order][::-1], color="#8e44ad")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig("10_shap_importance.png")
    print("  SHAP plot saved.")


# ===========================================================================
# STEP 6 — Summary report (JSON + text)
# ===========================================================================

def step_summary(df: pd.DataFrame, model_info: dict, fc_df: pd.DataFrame):
    print("\n=== STEP 6: Summary Report ===")

    metrics_path = os.path.join(REPORT_DIR, "model_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as fh:
            metrics = json.load(fh)

    best_name = max(metrics, key=lambda n: metrics[n]["R2"]) if metrics else "N/A"
    best_r2   = metrics[best_name]["R2"] if metrics else "N/A"
    best_rmse = metrics[best_name]["RMSE"] if metrics else "N/A"

    summary = {
        "project":         "Hyderabad AQI Analysis & Forecasting",
        "generated":       datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "rows":        int(len(df)),
            "date_range":  f"{df['Date'].min().date()} — {df['Date'].max().date()}",
            "features":    list(df.columns),
        },
        "aqi_stats": {
            "mean":   round(float(df["AQI"].mean()), 2),
            "median": round(float(df["AQI"].median()), 2),
            "min":    round(float(df["AQI"].min()), 2),
            "max":    round(float(df["AQI"].max()), 2),
            "std":    round(float(df["AQI"].std()), 2),
        },
        "category_distribution": df["AQI_Bucket"].value_counts().to_dict(),
        "models":          metrics,
        "best_model": {
            "name": best_name,
            "R2":   best_r2,
            "RMSE": best_rmse,
        },
        "forecast_30day": {
            "avg_aqi": round(float(fc_df["AQI_forecast"].mean()), 2),
            "max_aqi": round(float(fc_df["AQI_forecast"].max()), 2),
            "min_aqi": round(float(fc_df["AQI_forecast"].min()), 2),
        } if fc_df is not None and len(fc_df) else {},
    }

    summary_json = os.path.join(REPORT_DIR, "summary.json")
    with open(summary_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Summary JSON → {summary_json}")

    # Plain-text report
    txt_path = os.path.join(REPORT_DIR, "summary_report.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("=" * 65 + "\n")
        fh.write("  HYDERABAD AQI ANALYSIS — SUMMARY REPORT\n")
        fh.write("=" * 65 + "\n\n")
        fh.write(f"Generated  : {summary['generated']}\n")
        fh.write(f"Dataset    : {summary['dataset']['rows']} days  "
                 f"({summary['dataset']['date_range']})\n\n")
        fh.write("AQI Statistics\n")
        fh.write("-" * 40 + "\n")
        for k, v in summary["aqi_stats"].items():
            fh.write(f"  {k:<10}: {v}\n")
        fh.write("\nCategory Distribution\n")
        fh.write("-" * 40 + "\n")
        for cat in CPCB_LABELS:
            cnt = summary["category_distribution"].get(cat, 0)
            fh.write(f"  {cat:<15}: {cnt}\n")
        fh.write("\nModel Performance (5-fold time-series CV)\n")
        fh.write("-" * 40 + "\n")
        for mname, mres in metrics.items():
            fh.write(f"  {mname:<28} RMSE={mres['RMSE']:.2f}  "
                     f"MAE={mres['MAE']:.2f}  R²={mres['R2']:.4f}\n")
        fh.write(f"\nBest Model : {best_name}  (R²={best_r2}, RMSE={best_rmse})\n")
        if summary["forecast_30day"]:
            fc = summary["forecast_30day"]
            fh.write(f"\n30-day Forecast (avg/min/max AQI)\n")
            fh.write(f"  Avg={fc['avg_aqi']}  Min={fc['min_aqi']}  Max={fc['max_aqi']}\n")
        fh.write("\n" + "=" * 65 + "\n")
    print(f"  Summary text → {txt_path}")


# ===========================================================================
# Main dispatcher
# ===========================================================================

ALL_STEPS = ["preprocess", "eda", "models", "forecast", "shap", "summary"]

def run_all():
    df          = step_preprocess()
    step_eda(df)
    model_info  = step_models(df)
    fc_df       = step_forecast(df, model_info)
    step_shap(df, model_info)
    step_summary(df, model_info, fc_df)
    print("\n✓ All steps complete.")
    print(f"  Figures  → {FIG_DIR}/")
    print(f"  Models   → {MODEL_DIR}/")
    print(f"  Reports  → {REPORT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Hyderabad AQI Pipeline")
    parser.add_argument("--step", choices=ALL_STEPS,
                        help="Run a single step instead of the full pipeline.")
    args = parser.parse_args()

    if args.step is None:
        run_all()
        return

    # Always need clean data except for preprocess itself
    if args.step == "preprocess":
        step_preprocess()
        return

    if not os.path.exists(CLEAN_CSV):
        print(f"Clean data not found ({CLEAN_CSV}). Running preprocess first...")
        df = step_preprocess()
    else:
        df = pd.read_csv(CLEAN_CSV, parse_dates=["Date"])

    if args.step == "eda":
        step_eda(df)
    elif args.step == "models":
        step_models(df)
    elif args.step == "forecast":
        model_info = step_models(df)
        step_forecast(df, model_info)
    elif args.step == "shap":
        model_info = step_models(df)
        step_shap(df, model_info)
    elif args.step == "summary":
        model_info = step_models(df)
        fc_df      = step_forecast(df, model_info)
        step_summary(df, model_info, fc_df)


if __name__ == "__main__":
    main()
