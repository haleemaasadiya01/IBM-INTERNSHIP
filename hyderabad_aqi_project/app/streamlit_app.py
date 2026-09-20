"""
Hyderabad AQI — Streamlit Dashboard
=====================================
Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV  = os.path.join(ROOT, "data", "processed", "hyderabad_clean.csv")
MODEL_DIR  = os.path.join(ROOT, "outputs", "models")
REPORT_DIR = os.path.join(ROOT, "outputs", "reports")
FIG_DIR    = os.path.join(ROOT, "outputs", "figures")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CPCB_BINS   = [0, 50, 100, 200, 300, 400, float("inf")]
CPCB_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
CPCB_COLORS = {
    "Good":         "#27ae60",
    "Satisfactory": "#2ecc71",
    "Moderate":     "#f39c12",
    "Poor":         "#e67e22",
    "Very Poor":    "#e74c3c",
    "Severe":       "#8e44ad",
    "Unknown":      "#95a5a6",
}
HEALTH_ADVICE = {
    "Good":         "Air quality is satisfactory. Enjoy outdoor activities.",
    "Satisfactory": "Air quality is acceptable. Unusually sensitive people should limit prolonged outdoor exertion.",
    "Moderate":     "Members of sensitive groups may experience health effects. General public unlikely to be affected.",
    "Poor":         "Everyone may begin to experience health effects. Sensitive groups should avoid prolonged outdoor activities.",
    "Very Poor":    "Health alert: everyone may experience more serious health effects. Avoid outdoor activities.",
    "Severe":       "Health emergency! Entire population likely to be affected. Stay indoors and avoid all outdoor exertion.",
    "Unknown":      "No data available.",
}

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
              "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]

FEATURE_COLS = [
    "Month", "DayOfYear", "DayOfWeek", "Year",
    "PM2.5", "PM10", "NO2", "SO2", "O3", "CO", "NH3",
    "AQI_lag1", "AQI_lag3", "AQI_lag7",
    "AQI_roll7", "AQI_roll30",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_aqi(v):
    if pd.isna(v):
        return "Unknown"
    for lo, hi, label in zip(CPCB_BINS, CPCB_BINS[1:], CPCB_LABELS):
        if lo <= v < hi:
            return label
    return "Severe"


@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(CLEAN_CSV):
        return None
    df = pd.read_csv(CLEAN_CSV, parse_dates=["Date"])
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    path = os.path.join(MODEL_DIR, "xgboost.pkl")
    if not os.path.exists(path):
        # Fallback to any available model
        for fname in ["random_forest.pkl", "gradient_boosting.pkl", "ridge_regression.pkl"]:
            alt = os.path.join(MODEL_DIR, fname)
            if os.path.exists(alt):
                path = alt
                break
        else:
            return None, None
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    return obj["model"], obj["features"]


@st.cache_data(show_spinner=False)
def load_metrics():
    path = os.path.join(REPORT_DIR, "model_metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_forecast():
    path = os.path.join(REPORT_DIR, "forecast_30day.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["Date"])


def aqi_color_badge(aqi_val):
    cat   = classify_aqi(aqi_val)
    color = CPCB_COLORS.get(cat, "#95a5a6")
    return f"""
    <div style="background:{color};color:white;padding:16px 24px;border-radius:12px;
                text-align:center;font-size:1.6rem;font-weight:700;display:inline-block;">
        AQI {aqi_val:.0f}
        <div style="font-size:1rem;font-weight:400;margin-top:4px;">{cat}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hyderabad AQI Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌫️ Hyderabad AQI")
    st.markdown("Air Quality Index analysis and 30-day forecasting for Hyderabad, India.")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔬 EDA", "🤖 Models", "🔮 Forecast", "🔍 Custom Predict"],
    )
    st.markdown("---")
    st.caption("Dataset: 'Air Quality Data in India' by Rohan Rao (Kaggle) · Built with Streamlit")

# ---------------------------------------------------------------------------
# Load assets
# ---------------------------------------------------------------------------
df      = load_data()
model, features = load_model()
metrics = load_metrics()
fc_df   = load_forecast()

pipeline_not_run = df is None

if pipeline_not_run:
    st.warning(
        "⚠️ Processed data not found. "
        "Please run the pipeline first:\n\n"
        "```\npython hyderabad_aqi_pipeline.py\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------------------
if page == "📊 Overview":
    st.title("📊 Hyderabad Air Quality — Overview")
    st.markdown(f"**Dataset:** {len(df):,} daily observations · "
                f"{df['Date'].min().date()} → {df['Date'].max().date()}")

    # KPI cards
    latest = df.sort_values("Date").iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest AQI",   f"{latest['AQI']:.0f}")
    col2.metric("Category",     latest.get("AQI_Bucket", classify_aqi(latest["AQI"])))
    col3.metric("Overall Mean", f"{df['AQI'].mean():.1f}")
    col4.metric("Overall Max",  f"{df['AQI'].max():.0f}")

    # AQI badge + health advice
    cat    = classify_aqi(latest["AQI"])
    st.markdown(aqi_color_badge(latest["AQI"]), unsafe_allow_html=True)
    st.info(f"**Health Advice:** {HEALTH_ADVICE.get(cat, '')}")

    st.markdown("---")

    # Time-series plot
    st.subheader("Daily AQI Time-series")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df["Date"], df["AQI"], lw=0.8, color="#3498db", alpha=0.6, label="Daily")
    ax.plot(df["Date"], df["AQI_roll30"], lw=2, color="#e74c3c", label="30-day MA")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Category distribution
    st.subheader("AQI Category Distribution")
    cat_counts = df["AQI_Bucket"].value_counts().reindex(CPCB_LABELS, fill_value=0)
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.dataframe(cat_counts.rename("Days").reset_index().rename(columns={"index": "Category"}),
                     use_container_width=True, hide_index=True)
    with col_b:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = [CPCB_COLORS.get(l, "#95a5a6") for l in cat_counts.index]
        ax2.bar(cat_counts.index, cat_counts.values, color=colors, edgecolor="white")
        ax2.set_ylabel("Days")
        ax2.set_xlabel("Category")
        ax2.tick_params(axis="x", rotation=25)
        ax2.grid(axis="y", alpha=0.3)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # Yearly average
    st.subheader("Yearly Average AQI")
    yr_avg = df.groupby("Year")["AQI"].mean()
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.bar(yr_avg.index.astype(str), yr_avg.values, color="#3498db", edgecolor="white")
    ax3.set_ylabel("Avg AQI")
    ax3.grid(axis="y", alpha=0.3)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)


# ---------------------------------------------------------------------------
# PAGE: EDA
# ---------------------------------------------------------------------------
elif page == "🔬 EDA":
    st.title("🔬 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Pollutant Trends", "Correlation", "Monthly Patterns"])

    with tab1:
        st.subheader("Pollutant Trends Over Time")
        available = [c for c in POLLUTANTS if c in df.columns]
        sel = st.multiselect("Select pollutants", available, default=available[:4])
        if sel:
            fig, axes = plt.subplots(len(sel), 1, figsize=(13, 3 * len(sel)), sharex=True)
            if len(sel) == 1:
                axes = [axes]
            for ax, col in zip(axes, sel):
                ax.plot(df["Date"], df[col], lw=0.7, color="#3498db", alpha=0.7)
                ax.plot(df["Date"], df[col].rolling(30, min_periods=1).mean(),
                        lw=1.5, color="#e74c3c")
                ax.set_ylabel(col, fontsize=9)
                ax.grid(alpha=0.2)
            axes[-1].set_xlabel("Date")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with tab2:
        st.subheader("Pollutant Correlation Heatmap")
        import seaborn as sns
        corr_cols = [c for c in POLLUTANTS if c in df.columns]
        fig, ax   = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", linewidths=0.4, ax=ax, annot_kws={"size": 7})
        ax.set_title("Pollutant Correlations")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab3:
        st.subheader("Monthly AQI Box-plots")
        MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]
        month_data = [df[df["Month"] == m]["AQI"].dropna().values for m in range(1, 13)]
        fig, ax    = plt.subplots(figsize=(12, 5))
        bp = ax.boxplot(month_data, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#aed6f1")
        ax.set_xticklabels(MONTH_LABELS)
        ax.set_ylabel("AQI")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Average AQI by Day of Week")
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_avg    = df.groupby("DayOfWeek")["AQI"].mean()
        fig2, ax2  = plt.subplots(figsize=(8, 3))
        ax2.bar(dow_labels, dow_avg.values, color="#9b59b6", edgecolor="white")
        ax2.set_ylabel("Avg AQI")
        ax2.grid(axis="y", alpha=0.3)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


# ---------------------------------------------------------------------------
# PAGE: Models
# ---------------------------------------------------------------------------
elif page == "🤖 Models":
    st.title("🤖 Machine Learning Models")

    if not metrics:
        st.warning("No model metrics found. Run the pipeline first.")
    else:
        # Metrics table
        st.subheader("Model Performance (5-fold Time-series CV)")
        rows = []
        for name, m in metrics.items():
            rows.append({"Model": name, "RMSE": m["RMSE"], "MAE": m["MAE"], "R²": m["R2"]})
        mdf = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)

        def color_r2(val):
            color = "#27ae60" if val >= 0.8 else "#f39c12" if val >= 0.6 else "#e74c3c"
            return f"color: {color}; font-weight: bold"

        st.dataframe(
            mdf.style.map(color_r2, subset=["R²"]),
            use_container_width=True, hide_index=True
        )

        # RMSE bar chart
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(mdf["Model"], mdf["RMSE"], color="#3498db")
            ax.set_xlabel("RMSE (lower = better)")
            ax.set_title("RMSE by Model")
            ax.grid(axis="x", alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(mdf["Model"], mdf["R²"], color="#27ae60")
            ax.set_xlabel("R² (higher = better)")
            ax.set_title("R² by Model")
            ax.grid(axis="x", alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # SHAP figure if exists
        shap_fig = os.path.join(FIG_DIR, "10_shap_importance.png")
        if os.path.exists(shap_fig):
            st.subheader("SHAP Feature Importance")
            st.image(shap_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Forecast
# ---------------------------------------------------------------------------
elif page == "🔮 Forecast":
    st.title("🔮 30-Day AQI Forecast")

    if fc_df is None:
        st.warning("No forecast data. Run `python hyderabad_aqi_pipeline.py --step forecast`.")
    else:
        st.markdown(
            f"Forecast period: **{fc_df['Date'].min().date()}** → **{fc_df['Date'].max().date()}**"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Forecast AQI", f"{fc_df['AQI_forecast'].mean():.1f}")
        col2.metric("Min Forecast AQI", f"{fc_df['AQI_forecast'].min():.1f}")
        col3.metric("Max Forecast AQI", f"{fc_df['AQI_forecast'].max():.1f}")

        # Combined plot
        hist_plot = df.tail(90)
        last_date = df["Date"].max()
        fig, ax   = plt.subplots(figsize=(13, 5))
        ax.plot(hist_plot["Date"], hist_plot["AQI"],
                lw=1.2, color="#3498db", label="Historical AQI")
        ax.plot(fc_df["Date"], fc_df["AQI_forecast"],
                lw=2.2, color="#e74c3c", linestyle="--", label="Forecast")
        ax.fill_between(fc_df["Date"],
                        fc_df["AQI_forecast"] * 0.9,
                        fc_df["AQI_forecast"] * 1.1,
                        alpha=0.15, color="#e74c3c", label="±10% band")
        ax.axvline(x=last_date, color="grey", linestyle=":", lw=1.5, label="Today")
        ax.set_ylabel("AQI")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Forecast Table")
        display_fc = fc_df[["Date", "AQI_forecast", "AQI_Bucket"]].copy()
        display_fc["AQI_forecast"] = display_fc["AQI_forecast"].round(1)
        display_fc.columns = ["Date", "Forecast AQI", "Category"]

        def color_cat(val):
            c = CPCB_COLORS.get(val, "#95a5a6")
            return f"background-color: {c}; color: white"

        st.dataframe(
            display_fc.style.map(color_cat, subset=["Category"]),
            use_container_width=True, hide_index=True
        )

        csv_bytes = display_fc.to_csv(index=False).encode()
        st.download_button("⬇ Download Forecast CSV", csv_bytes,
                           "hyderabad_aqi_forecast.csv", "text/csv")


# ---------------------------------------------------------------------------
# PAGE: Custom Predict
# ---------------------------------------------------------------------------
elif page == "🔍 Custom Predict":
    st.title("🔍 Custom AQI Predictor")
    st.markdown("Enter current pollutant readings to predict the AQI instantly.")

    if model is None or features is None:
        st.warning("Model not found. Run the pipeline first.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            pm25  = st.number_input("PM2.5 (µg/m³)", 0.0, 500.0, 60.0, step=1.0)
            pm10  = st.number_input("PM10 (µg/m³)",  0.0, 600.0, 90.0, step=1.0)
            no2   = st.number_input("NO2 (µg/m³)",   0.0, 200.0, 30.0, step=1.0)
        with col2:
            so2   = st.number_input("SO2 (µg/m³)",   0.0, 200.0, 10.0, step=1.0)
            o3    = st.number_input("O3 (µg/m³)",    0.0, 300.0, 50.0, step=1.0)
            co    = st.number_input("CO (mg/m³)",     0.0,  50.0,  1.0, step=0.1)
        with col3:
            nh3   = st.number_input("NH3 (µg/m³)",   0.0, 200.0, 15.0, step=1.0)
            month = st.slider("Month", 1, 12, int(pd.Timestamp.now().month))
            year  = st.number_input("Year", 2015, 2030, int(pd.Timestamp.now().year), step=1)

        row = {
            "Month":      month,
            "DayOfYear":  pd.Timestamp(f"{int(year)}-{month:02d}-15").timetuple().tm_yday,
            "DayOfWeek":  pd.Timestamp(f"{int(year)}-{month:02d}-15").weekday(),
            "Year":       year,
            "PM2.5":      pm25, "PM10": pm10, "NO2": no2,
            "SO2":        so2,  "O3":   o3,   "CO":  co, "NH3": nh3,
            "AQI_lag1":   100.0, "AQI_lag3": 100.0, "AQI_lag7": 100.0,
            "AQI_roll7":  100.0, "AQI_roll30": 100.0,
        }

        if st.button("🔮 Predict AQI", type="primary"):
            X_in = np.array([[row.get(f, 0.0) for f in features]])
            pred = float(model.predict(X_in)[0])
            pred = max(0.0, pred)
            cat  = classify_aqi(pred)
            st.markdown(aqi_color_badge(pred), unsafe_allow_html=True)
            st.info(f"**Health Advice:** {HEALTH_ADVICE.get(cat, '')}")

            # CPCB scale visualization
            st.markdown("#### CPCB AQI Scale")
            scale_fig, scale_ax = plt.subplots(figsize=(10, 1.2))
            scale_ax.set_xlim(0, 500)
            scale_ax.set_ylim(0, 1)
            bands = [(0,50,"Good","#27ae60"), (50,100,"Satisfactory","#2ecc71"),
                     (100,200,"Moderate","#f39c12"), (200,300,"Poor","#e67e22"),
                     (300,400,"Very Poor","#e74c3c"), (400,500,"Severe","#8e44ad")]
            for lo, hi, lbl, clr in bands:
                scale_ax.barh(0, hi - lo, left=lo, height=1, color=clr, edgecolor="white")
                scale_ax.text((lo + hi) / 2, 0.5, lbl, ha="center", va="center",
                              color="white", fontsize=8, fontweight="bold")
            scale_ax.axvline(x=min(pred, 499), color="black", lw=2.5, linestyle="--")
            scale_ax.axis("off")
            st.pyplot(scale_fig, use_container_width=True)
            plt.close(scale_fig)
