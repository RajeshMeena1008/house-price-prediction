# app.py
import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
from io import StringIO, BytesIO

# ---------------------------
# Config / Paths
# ---------------------------
MODEL_PATH = "house_price_model.pkl"
SCALER_PATH = "scaler.pkl"
PRED_CSV = "prediction_history.csv"
NOTEBOOK_PATH = "/mnt/data/House Price Prediction.ipynb"  # your notebook path (as requested)

# ---------------------------
# Load model & scaler
# ---------------------------
@st.cache_resource
def load_model_and_scaler():
    model = None
    scaler = None
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load model from {MODEL_PATH}: {e}")
    else:
        st.info(f"Model file not found at {MODEL_PATH} — Manual prediction disabled until provided.")

    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load scaler from {SCALER_PATH}: {e}")
    else:
        st.info(f"Scaler file not found at {SCALER_PATH} — Manual prediction disabled until provided.")

    return model, scaler

model, scaler = load_model_and_scaler()

# ---------------------------
# Page config & styling
# ---------------------------
st.set_page_config(page_title="House Price Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .title {
            background: linear-gradient(90deg, #1e3c72, #2a5298);
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            color: white;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 16px;
            box-shadow: 0px 3px 10px rgba(0,0,0,0.2);
        }
        .section-title { font-size: 20px; font-weight: bold; margin-bottom: 8px; color: #1e3c72; }
        .card { background: #ffffff; padding: 14px; border-radius: 10px; box-shadow: 0px 3px 8px rgba(0,0,0,0.06); margin-bottom: 14px; }
        .desc { font-size: 13px; color: #444; margin-bottom: 6px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🏠 House Price Predictor — Manual, CSV, History, Chat Bot & EMI</div>', unsafe_allow_html=True)

# ---------------------------
# Sidebar - modes
# ---------------------------
mode = st.sidebar.radio("Choose mode", (
    "Manual Input & Predict",
    "Upload CSV & Compare",
    "Prediction History & Graph",
    "Chat Bot",
    "EMI Calculator"
))

st.sidebar.markdown("### Uploaded Notebook")
st.sidebar.markdown(f"[Open notebook]({NOTEBOOK_PATH})")

# ---------------------------
# Helpers
# ---------------------------
def find_column(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def map_uploaded_columns(df):
    mapping = {}
    mapping['bedrooms'] = find_column(df, ["bedrooms","bed","bhk"])
    mapping['bathrooms'] = find_column(df, ["bathrooms","baths"])
    mapping['living_area'] = find_column(df, ["living_area","sqft","area"])
    mapping['lot_area'] = find_column(df, ["lot_area","lot_sqft"])
    mapping['floors'] = find_column(df, ["floors","stories"])
    mapping['waterfront'] = find_column(df, ["waterfront"])
    mapping['view'] = find_column(df, ["view"])
    mapping['condition'] = find_column(df, ["condition"])
    mapping['grade'] = find_column(df, ["grade"])
    mapping['area_excl_basement'] = find_column(df, ["area_excl_basement","area_excluding_basement"])
    mapping['area_basement'] = find_column(df, ["area_basement"])
    mapping['built_year'] = find_column(df, ["built_year","year_built","yr_built"])
    mapping['lat'] = find_column(df, ["lat","latitude"])
    mapping['long'] = find_column(df, ["long","longitude","lng"])
    mapping['living_area_renov'] = find_column(df, ["living_area_renov"])
    mapping['lot_area_renov'] = find_column(df, ["lot_area_renov"])
    mapping['schools'] = find_column(df, ["schools","schools_nearby"])
    mapping['airport_dist'] = find_column(df, ["airport_dist","distance_airport","dist_airport"])
    mapping['price'] = find_column(df, ["price","sale_price","amount"])
    return mapping

def append_prediction_history(row_dict):
    """Append a single prediction (row_dict) to PRED_CSV safely."""
    try:
        row_df = pd.DataFrame([row_dict])
        if os.path.exists(PRED_CSV):
            row_df.to_csv(PRED_CSV, mode='a', header=False, index=False)
        else:
            row_df.to_csv(PRED_CSV, index=False)
    except Exception as e:
        st.warning(f"Could not write to history CSV: {e}")

def load_history():
    if os.path.exists(PRED_CSV):
        try:
            h = pd.read_csv(PRED_CSV)
            return h
        except Exception as e:
            st.warning(f"Could not read history CSV: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def plot_price_trend(history_df, title="Predicted Prices Over Time"):
    """Line chart used in CSV & History sections (unchanged)."""
    if history_df.empty or "predicted_price" not in history_df.columns:
        st.info("No price data to plot yet.")
        return
    fig, ax = plt.subplots(figsize=(8,3))
    if "date_time" in history_df.columns:
        try:
            history_df['date_parsed'] = pd.to_datetime(history_df['date_time'], errors='coerce')
            valid = history_df.dropna(subset=['date_parsed'])
            if not valid.empty:
                sorted_hist = valid.sort_values('date_parsed')
                ax.plot(sorted_hist['date_parsed'], sorted_hist['predicted_price'], marker='o', linestyle='-')
                ax.set_xlabel("Date")
            else:
                ax.plot(history_df.index, history_df['predicted_price'], marker='o', linestyle='-')
                ax.set_xlabel("Record Index")
        except Exception:
            ax.plot(history_df.index, history_df['predicted_price'], marker='o', linestyle='-')
            ax.set_xlabel("Record Index")
    else:
        ax.plot(history_df.index, history_df['predicted_price'], marker='o', linestyle='-')
        ax.set_xlabel("Record Index")
    ax.set_ylabel("Predicted Price (₹)")
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def plot_price_pie(history_df, title="Price Distribution (latest 10)"):
    """
    Pie chart for Manual Input & Predict section.
    - Uses latest 10 records.
    - Legend on the left with labels.
    - Percentages shown inside slices.
    """
    if history_df.empty or "predicted_price" not in history_df.columns:
        st.info("No price data to plot yet.")
        return

    # Take latest 10 by date_time if available, else by index
    if "date_time" in history_df.columns:
        try:
            history_df["date_parsed"] = pd.to_datetime(history_df["date_time"], errors="coerce")
            valid = history_df.dropna(subset=["date_parsed"])
            latest = valid.sort_values("date_parsed", ascending=False).head(10)
        except Exception:
            latest = history_df.sort_index(ascending=False).head(10)
    else:
        latest = history_df.sort_index(ascending=False).head(10)

    prices = pd.to_numeric(latest["predicted_price"], errors="coerce").dropna()
    if prices.empty:
        st.info("No valid price data to plot yet.")
        return

    # Labels: #index + price
    labels = [f"#{i+1}: ₹{v:,.0f}" for i, v in enumerate(prices)]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        prices,
        labels=None,             # labels handled by legend
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title(title)
    ax.axis("equal")  # Equal aspect ratio for a proper circle

    # Legend on the left side
    ax.legend(
        wedges,
        labels,
        title="Predictions",
        loc="center left",
        bbox_to_anchor=(-0.3, 0.5),
        frameon=False
    )

    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------
# MODE 1 — Manual Input & Predict
# ---------------------------
if mode == "Manual Input & Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📌 Manual Input</div>", unsafe_allow_html=True)

    with st.form("manual_form"):
        col1, col2 = st.columns(2)
        bedrooms = col1.number_input("Bedrooms", 1, 10, 3)
        bathrooms = col2.number_input("Bathrooms", 1, 6, 2)

        living_area = st.number_input("Living Area (sqft)", 370, 8000, 2000)
        lot_area = st.number_input("Lot Area (sqft)", 520, 50000, 4000)

        col3, col4 = st.columns(2)
        floors = col3.number_input("Floors", 1.0, 3.0, 1.0, step=0.5)
        waterfront = col4.selectbox("Waterfront", ["No", "Yes"])
        waterfront_val = 1 if waterfront == "Yes" else 0

        view = st.number_input("View Rating", 0, 4, 1)

        col5, col6 = st.columns(2)
        condition = col5.number_input("Condition", 1, 5, 3)
        grade = col6.number_input("Grade", 4, 13, 7)

        area_excl = st.number_input("Area Excluding Basement", 370, 5000, 1500)
        basement = st.number_input("Basement Area", 0, 2000, 200)

        built_year = st.number_input("Built Year", 1950, 2023, 1990)

        col7, col8 = st.columns(2)
        living_reno = col7.number_input("Renovated Living Area", 0, 4000, 0)
        lot_reno = col8.number_input("Renovated Lot Area", 0, 50000, 0)

        col9, col10 = st.columns(2)
        lat = col9.number_input("Latitude", -90.0, 90.0, 52.7)
        long = col10.number_input("Longitude", -180.0, 180.0, -114.3)

        col11, col12 = st.columns(2)
        schools = col11.number_input("Schools Nearby", 0, 10, 2)
        airport = col12.number_input("Distance to Airport (km)", 0, 1000, 50)

        submit = st.form_submit_button("🔍 Predict House Price")

    st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        if model is None or scaler is None:
            st.error("Model or Scaler missing. Place model & scaler files in app folder.")
        else:
            data = np.array([
                bedrooms, bathrooms, living_area, lot_area,
                floors, waterfront_val, view,
                condition, grade, area_excl, basement,
                built_year, lat, long,
                living_reno, lot_reno,
                schools, airport
            ]).reshape(1, -1)

            try:
                scaled = scaler.transform(data)
                price = model.predict(scaled)[0]
                st.success(f"💰 Predicted House Price: ₹{price:,.2f}")

                # Save prediction to CSV history
                row = {
                    "date_time": datetime.now().isoformat(),
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "living_area": living_area,
                    "lot_area": lot_area,
                    "floors": floors,
                    "waterfront": waterfront_val,
                    "view": view,
                    "condition": condition,
                    "grade": grade,
                    "area_excl_basement": area_excl,
                    "area_basement": basement,
                    "built_year": built_year,
                    "lat": lat,
                    "long": long,
                    "living_area_renov": living_reno,
                    "lot_area_renov": lot_reno,
                    "schools": schools,
                    "airport_dist": airport,
                    "predicted_price": float(price)
                }
                append_prediction_history(row)
                st.info(f"Saved this prediction to {PRED_CSV}")

                # Show recent history and pie chart immediately under Manual mode
                st.markdown("----")
                st.subheader("📜 Recent Prediction History (latest 10)")
                history_df = load_history()
                if not history_df.empty:
                    st.dataframe(
                        history_df.sort_values(by="date_time", ascending=False)
                        .head(10)
                        .reset_index(drop=True)
                    )
                    st.subheader("📊 Price Distribution (history, latest 10)")
                    plot_price_pie(history_df, title="Price Distribution (latest 10 predictions)")
                else:
                    st.info("No history to display yet.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------------------
# MODE 2 — Upload CSV & Compare
# ---------------------------
elif mode == "Upload CSV & Compare":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📥 Upload CSV</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload house dataset CSV", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV loaded successfully")
        except Exception as e:
            st.error(f"Failed to read CSV. Make sure it's a valid CSV file. {e}")
            df = None

        if df is not None:
            st.subheader("Preview (first 10 rows)")
            st.dataframe(df.head(10))

            # Map column names
            col_map = map_uploaded_columns(df)

            st.write("Detected column mapping (best-effort):")
            st.json(col_map)

            # Allow user to select which columns should be used for matching (override)
            st.markdown("---")
            st.markdown("### Column selectors (override auto-detected if needed)")
            col_bed = st.selectbox("Column for bedrooms", options=[None] + list(df.columns), index=0)
            col_bath = st.selectbox("Column for bathrooms", options=[None] + list(df.columns), index=0)
            col_year = st.selectbox("Column for built year", options=[None] + list(df.columns), index=0)
            col_price = st.selectbox("Column for price", options=[None] + list(df.columns), index=0)

            # If user didn't override, use detected mapping
            col_bed = col_bed if col_bed else col_map.get('bedrooms')
            col_bath = col_bath if col_bath else col_map.get('bathrooms')
            col_year = col_year if col_year else col_map.get('built_year')
            col_price = col_price if col_price else col_map.get('price')

            if not all([col_bed, col_bath, col_year, col_price]):
                st.warning("Uploaded CSV is missing columns we need for matching/display. Please select them above.")
            else:
                st.markdown("---")
                st.markdown("### Enter target values to find similar houses in uploaded CSV")
                t_col1, t_col2, t_col3 = st.columns(3)
                try:
                    default_bed = int(df[col_bed].dropna().iloc[0]) if df[col_bed].dropna().shape[0] > 0 else 3
                except Exception:
                    default_bed = 3
                try:
                    default_bath = int(df[col_bath].dropna().iloc[0]) if df[col_bath].dropna().shape[0] > 0 else 2
                except Exception:
                    default_bath = 2
                try:
                    default_year = int(df[col_year].dropna().iloc[0]) if df[col_year].dropna().shape[0] > 0 else 2000
                except Exception:
                    default_year = 2000

                tgt_bed = int(t_col1.number_input("Bedrooms to match", min_value=0, value=default_bed))
                tgt_bath = int(t_col2.number_input("Bathrooms to match", min_value=0, value=default_bath))
                tgt_year = int(t_col3.number_input("Built Year to match", min_value=1800, value=default_year))

                if st.button("🔎 Find similar houses in CSV"):
                    # Filter: exact equality on bedrooms, bathrooms, built_year (as requested)
                    try:
                        filtered = df[
                            (pd.to_numeric(df[col_bed], errors='coerce').round(0) == tgt_bed) &
                            (pd.to_numeric(df[col_bath], errors='coerce').round(0) == tgt_bath) &
                            (pd.to_numeric(df[col_year], errors='coerce').round(0) == tgt_year)
                        ].copy()
                    except Exception as e:
                        st.warning("Exact match with numeric conversion failed, trying fallback. " + str(e))
                        filtered = df[
                            (df[col_bed].astype(str) == str(tgt_bed)) &
                            (df[col_bath].astype(str) == str(tgt_bath)) &
                            (df[col_year].astype(str) == str(tgt_year))
                        ].copy()

                    n_matches = filtered.shape[0]
                    st.info(f"Found {n_matches} matching house(s) in the uploaded CSV (matching on bedrooms, bathrooms, built year).")

                    if n_matches == 0:
                        st.warning("No exact matches found. You can try uploading a different CSV, or adjust the target values.")
                    else:
                        # Show filtered table with key columns
                        display_cols = [c for c in [col_bed, col_bath, col_year, col_price] if c in filtered.columns]
                        for extra in ['living_area','sqft','area','lot_area','grade','condition']:
                            if extra in df.columns and extra not in display_cols:
                                display_cols.append(extra)
                        st.subheader("Matching Houses (showing key columns)")
                        st.dataframe(filtered[display_cols].reset_index(drop=True))

                        # Show price comparison graph (bar chart)
                        st.subheader("📊 Price Comparison of Matching Houses")
                        try:
                            prices = pd.to_numeric(filtered[col_price], errors='coerce').dropna()
                            fig, ax = plt.subplots()
                            ax.bar(range(len(prices)), prices)
                            ax.set_xlabel("Matching house index")
                            ax.set_ylabel("Price")
                            ax.set_title(f"Prices of {n_matches} matching houses")
                            st.pyplot(fig)

                            st.markdown("Price summary for matches")
                            st.write(prices.describe().to_frame(name="price_stats"))
                        except Exception as e:
                            st.error("Could not plot prices: " + str(e))

                        # Save matched rows to prediction history (use price column as predicted_price)
                        saved = 0
                        for _, r in filtered.iterrows():
                            try:
                                p_val = None
                                if col_price in r and pd.notna(r[col_price]):
                                    p_val = float(pd.to_numeric(r[col_price], errors='coerce'))
                                entry = {
                                    "date_time": datetime.now().isoformat(),
                                    "bedrooms": r.get(col_bed, np.nan),
                                    "bathrooms": r.get(col_bath, np.nan),
                                    "built_year": r.get(col_year, np.nan),
                                    "predicted_price": p_val
                                }
                                append_prediction_history(entry)
                                saved += 1
                            except Exception:
                                continue
                        st.success(f"Saved {saved} matched rows to {PRED_CSV} (price used from CSV).")

                        # Show history + trend after saving
                        history_df = load_history()
                        if not history_df.empty:
                            st.subheader("📜 Recent Prediction History (latest 10)")
                            st.dataframe(
                                history_df.sort_values(by="date_time", ascending=False)
                                .head(10)
                                .reset_index(drop=True)
                            )
                            st.subheader("📈 Price Trend (history)")
                            plot_price_trend(history_df, title="Predicted Prices Over Time (combined)")

                        # Option to download matched rows as CSV
                        csv_buffer = StringIO()
                        filtered.to_csv(csv_buffer, index=False)
                        st.download_button(
                            "Download matched houses as CSV",
                            csv_buffer.getvalue(),
                            file_name="matched_houses.csv",
                            mime="text/csv"
                        )

# ---------------------------
# MODE 3 — Prediction History & Graph
# ---------------------------
elif mode == "Prediction History & Graph":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Prediction History & Price Trend</div>", unsafe_allow_html=True)

    history = load_history()
    if history.empty:
        st.info("No prediction history found yet. Make some predictions in 'Manual Input & Predict' or save matches from CSV.")
    else:
        if "date_time" in history.columns:
            try:
                history["date_time_parsed"] = pd.to_datetime(history["date_time"], errors='coerce')
            except Exception:
                history["date_time_parsed"] = pd.NaT

        st.subheader("Recent Predictions (latest 50)")
        st.dataframe(history.sort_values(by="date_time", ascending=False).head(50).reset_index(drop=True))

        if "predicted_price" in history.columns and history["predicted_price"].notna().any():
            st.subheader("Price Summary")
            st.write(history["predicted_price"].describe().to_frame("stats"))

        st.subheader("Price Trend (all predictions)")
        plot_price_trend(history)

        csv_buffer = StringIO()
        history.to_csv(csv_buffer, index=False)
        st.download_button(
            "📥 Download Prediction History (CSV)",
            csv_buffer.getvalue(),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# MODE 4 — Chat Bot
# ---------------------------
elif mode == "Chat Bot":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>💬 AI Chat Bot</div>", unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask Anything...")

    if st.button("Send"):
        if msg.strip():
            st.session_state.chat.append(("You", msg))
            text = msg.lower()
            if "price" in text:
                reply = "House price depends on area, grade, condition, and location."
            elif "model" in text:
                reply = "This app uses RandomForestRegressor (or any regressor you trained & saved as model)."
            elif "predict" in text:
                reply = "Use Manual Input mode to predict price."
            elif "csv" in text:
                reply = "Upload CSV in Compare mode and save matches to history."
            else:
                reply = "I'm here to help with predictions, CSV uploads and EMI calculations!"
            st.session_state.chat.append(("Bot", reply))

    st.write("### Chat History")
    for s, m in st.session_state.chat:
        st.write(f"{s}:** {m}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# MODE 5 — EMI Calculator
# ---------------------------
elif mode == "EMI Calculator":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📈 EMI Calculator</div>", unsafe_allow_html=True)

    loan = st.number_input("Loan Amount (₹)", 10000, 100000000, 500000)
    rate = st.number_input("Interest Rate (%)", 0.0, 50.0, 8.0)
    years = st.number_input("Loan Tenure (Years)", 1, 40, 20)

    if st.button("Calculate EMI"):
        r = rate / 12 / 100
        n = years * 12
        if r == 0:
            emi = loan / n
        else:
            # Correct EMI formula
            emi = loan * r * (1 + r) ** n / ((1 + r) ** n - 1)

        total_payment = emi * n
        total_interest = total_payment - loan

        st.success(f"📌 Monthly EMI: ₹{emi:,.2f}")
        st.info(f"💰 Total Payment: ₹{total_payment:,.2f}")
        st.info(f"📉 Total Interest: ₹{total_interest:,.2f}")

    st.markdown("</div>", unsafe_allow_html=True)