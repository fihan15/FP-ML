# -*- coding: utf-8 -*-
"""
Streamlit Dashboard - Sistem Rekomendasi Wisata
================================================
Jalankan:
    streamlit run app.py

Pastikan file berikut berada satu folder dengan app.py:
- tour.csv
- tour_rating.csv
- user.csv
- output_rekomendasi_wisata_colab/*.csv
"""

import ast
import os
import random
import warnings
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sistem Rekomendasi Wisata",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL CONFIG
# ============================================================

TOUR_PATH = "tour.csv"
RATING_PATH = "tour_rating.csv"
USER_PATH = "user.csv"
OUTPUT_DIR = "output_rekomendasi_wisata_colab"

RANDOM_STATE = 42
RATING_MIN = 1.0
RATING_MAX = 5.0
TOP_K = 10
DIVERSE_MAX_PER_CATEGORY = 3

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.3rem;
    }
    .metric-card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 16px;
        padding: 18px;
        background: rgba(250, 250, 250, 0.65);
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .small-note {
        font-size: 0.9rem;
        color: #666;
        line-height: 1.45;
    }
    .section-header {
        margin-top: 1rem;
        font-size: 1.45rem;
        font-weight: 750;
    }
    .success-box {
        padding: 1rem;
        border-radius: 12px;
        background: rgba(46, 204, 113, 0.08);
        border: 1px solid rgba(46, 204, 113, 0.25);
    }
    .warning-box {
        padding: 1rem;
        border-radius: 12px;
        background: rgba(241, 196, 15, 0.10);
        border: 1px solid rgba(241, 196, 15, 0.30);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# UTILITY
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def clip_rating(x: float) -> float:
    if pd.isna(x):
        return 3.0
    return float(np.clip(x, RATING_MIN, RATING_MAX))


def price_level(price: float) -> str:
    price = float(price)
    if price <= 0:
        return "Gratis"
    if price <= 10000:
        return "Murah"
    if price <= 25000:
        return "Sedang"
    return "Mahal"


def rupiah(x) -> str:
    try:
        x = float(x)
        if x <= 0:
            return "Gratis"
        return "Rp{:,.0f}".format(x).replace(",", ".")
    except Exception:
        return str(x)


def percent(x: float) -> str:
    return f"{x * 100:.2f}%"


def show_title():
    st.markdown('<div class="main-title">🏝️ Sistem Rekomendasi Destinasi Wisata</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Dashboard hasil eksperimen machine learning untuk prediksi rating dan rekomendasi wisata.</div>',
        unsafe_allow_html=True,
    )


def download_df_button(df: pd.DataFrame, filename: str, label: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv_safe(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=True)
def load_all_data() -> Dict[str, pd.DataFrame]:
    tour = read_csv_safe(TOUR_PATH)
    rating_raw = read_csv_safe(RATING_PATH)
    user = read_csv_safe(USER_PATH)

    base = OUTPUT_DIR
    base_results = read_csv_safe(os.path.join(base, "hasil_evaluasi_split_model_dasar.csv"))
    tuning_summary = read_csv_safe(os.path.join(base, "hasil_tuning_ringkasan.csv"))
    tuning_detail = read_csv_safe(os.path.join(base, "hasil_tuning_detail.csv"))
    final_summary = read_csv_safe(os.path.join(base, "ringkasan_model_final.csv"))
    data_clean = read_csv_safe(os.path.join(base, "data_gabungan_bersih.csv"))
    rec_normal = read_csv_safe(os.path.join(base, "rekomendasi_user_1_normal.csv"))
    rec_diverse = read_csv_safe(os.path.join(base, "rekomendasi_user_1_diverse.csv"))

    return {
        "tour": tour,
        "rating_raw": rating_raw,
        "user": user,
        "base_results": base_results,
        "tuning_summary": tuning_summary,
        "tuning_detail": tuning_detail,
        "final_summary": final_summary,
        "data_clean": data_clean,
        "rec_normal": rec_normal,
        "rec_diverse": rec_diverse,
    }


@st.cache_data(show_spinner=True)
def prepare_data(tour: pd.DataFrame, rating_raw: pd.DataFrame, user: pd.DataFrame):
    tour = tour.copy()
    rating_raw = rating_raw.copy()
    user = user.copy()

    tour["Place_Id"] = tour["Place_Id"].astype(int)
    rating_raw["User_Id"] = rating_raw["User_Id"].astype(int)
    rating_raw["Place_Id"] = rating_raw["Place_Id"].astype(int)
    user["User_Id"] = user["User_Id"].astype(int)

    if "Time_Minutes" in tour.columns:
        median_time = tour["Time_Minutes"].median()
        if pd.isna(median_time):
            median_time = 60
        tour["Time_Minutes"] = tour["Time_Minutes"].fillna(median_time)
    else:
        tour["Time_Minutes"] = 60

    if "Description" not in tour.columns:
        tour["Description"] = ""
    tour["Description"] = tour["Description"].fillna("")

    for col in ["Category", "City"]:
        if col not in tour.columns:
            tour[col] = "Unknown"
        tour[col] = tour[col].fillna("Unknown")

    for col in ["Price", "Rating", "Latitude", "Longitude"]:
        if col not in tour.columns:
            tour[col] = 0
        tour[col] = pd.to_numeric(tour[col], errors="coerce")
        fill_value = tour[col].median() if tour[col].notna().any() else 0
        tour[col] = tour[col].fillna(fill_value)

    if "Location" not in user.columns:
        user["Location"] = "Unknown"
    user["Location"] = user["Location"].fillna("Unknown")

    if "Age" not in user.columns:
        user["Age"] = 0
    user["Age"] = pd.to_numeric(user["Age"], errors="coerce")
    user["Age"] = user["Age"].fillna(user["Age"].median())

    rating = (
        rating_raw
        .groupby(["User_Id", "Place_Id"], as_index=False)["Place_Ratings"]
        .mean()
    )
    rating["Place_Ratings"] = rating["Place_Ratings"].clip(RATING_MIN, RATING_MAX)

    data = rating.merge(tour, on="Place_Id", how="left")
    data = data.merge(user, on="User_Id", how="left")
    data = data.dropna(subset=["Place_Name", "Category", "Place_Ratings"])
    data["User_Id"] = data["User_Id"].astype(int)
    data["Place_Id"] = data["Place_Id"].astype(int)

    n_users = data["User_Id"].nunique()
    n_places = data["Place_Id"].nunique()
    possible = n_users * n_places
    sparsity = 1 - (len(data) / possible)

    stats = {
        "raw_rating_count": len(rating_raw),
        "dedup_rating_count": len(rating),
        "merged_count": len(data),
        "n_users": n_users,
        "n_places": n_places,
        "sparsity": sparsity,
        "avg_rating_per_user": len(data) / n_users,
        "avg_rating_per_place": len(data) / n_places,
    }

    return tour, user, rating, data, stats

# ============================================================
# FINAL RECOMMENDER MODEL FOR DYNAMIC RECOMMENDATION
# ============================================================

class BiasBaselineModel:
    def __init__(self, n_epochs=20, lr=0.006, reg=20.0, random_state=42):
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state

    def fit(self, train_df, tour_df=None):
        set_seed(self.random_state)
        self.global_mean = float(train_df["Place_Ratings"].mean())
        users = sorted(train_df["User_Id"].unique())
        items = sorted(train_df["Place_Id"].unique())
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {it: j for j, it in enumerate(items)}
        self.bu = np.zeros(len(users), dtype=float)
        self.bi = np.zeros(len(items), dtype=float)
        samples = train_df[["User_Id", "Place_Id", "Place_Ratings"]].values.tolist()

        for _ in range(self.n_epochs):
            random.shuffle(samples)
            for uid, pid, rating in samples:
                uid = int(uid)
                pid = int(pid)
                r = float(rating)
                u = self.user_to_idx.get(uid)
                i = self.item_to_idx.get(pid)
                if u is None or i is None:
                    continue
                pred = self.global_mean + self.bu[u] + self.bi[i]
                err = r - pred
                self.bu[u] += self.lr * (err - self.reg * self.bu[u] / len(samples))
                self.bi[i] += self.lr * (err - self.reg * self.bi[i] / len(samples))
        return self

    def predict(self, user_id, place_id):
        pred = self.global_mean
        u = self.user_to_idx.get(int(user_id))
        i = self.item_to_idx.get(int(place_id))
        if u is not None:
            pred += self.bu[u]
        if i is not None:
            pred += self.bi[i]
        return clip_rating(pred)


class UserPreferenceContentModel:
    def __init__(self, w_category=0.45, w_price=0.20, w_popularity=0.25, w_public=0.10, smoothing=8.0):
        self.w_category = w_category
        self.w_price = w_price
        self.w_popularity = w_popularity
        self.w_public = w_public
        self.smoothing = smoothing

    def fit(self, train_df, tour_df=None):
        self.global_mean = float(train_df["Place_Ratings"].mean())
        self.user_mean = train_df.groupby("User_Id")["Place_Ratings"].mean().to_dict()
        self.place_count = train_df.groupby("Place_Id")["Place_Ratings"].count().to_dict()
        self.place_sum = train_df.groupby("Place_Id")["Place_Ratings"].sum().to_dict()

        place_info = tour_df.copy() if tour_df is not None else train_df.drop_duplicates("Place_Id").copy()
        place_info["Place_Id"] = place_info["Place_Id"].astype(int)
        place_info["Price_Level"] = place_info["Price"].apply(price_level)
        self.place_info = place_info.set_index("Place_Id").to_dict("index")

        train_aug = train_df.copy()
        train_aug["Price_Level"] = train_aug["Price"].apply(price_level)

        self.user_category_sum = train_aug.groupby(["User_Id", "Category"])["Place_Ratings"].sum().to_dict()
        self.user_category_count = train_aug.groupby(["User_Id", "Category"])["Place_Ratings"].count().to_dict()
        self.user_price_sum = train_aug.groupby(["User_Id", "Price_Level"])["Place_Ratings"].sum().to_dict()
        self.user_price_count = train_aug.groupby(["User_Id", "Price_Level"])["Place_Ratings"].count().to_dict()

        public = place_info["Rating"].astype(float)
        self.public_mean = float(public.mean()) if len(public) else 0.0
        self.public_std = float(public.std()) if public.std() and not pd.isna(public.std()) else 1.0
        return self

    def smoothed_place_rating(self, place_id: int) -> float:
        cnt = self.place_count.get(int(place_id), 0)
        s = self.place_sum.get(int(place_id), 0.0)
        val = (s + self.smoothing * self.global_mean) / (cnt + self.smoothing)
        return clip_rating(val)

    def smoothed_user_category_rating(self, user_id: int, category: str) -> float:
        user_base = self.user_mean.get(int(user_id), self.global_mean)
        key = (int(user_id), category)
        cnt = self.user_category_count.get(key, 0)
        s = self.user_category_sum.get(key, 0.0)
        val = (s + self.smoothing * user_base) / (cnt + self.smoothing)
        return clip_rating(val)

    def smoothed_user_price_rating(self, user_id: int, level: str) -> float:
        user_base = self.user_mean.get(int(user_id), self.global_mean)
        key = (int(user_id), level)
        cnt = self.user_price_count.get(key, 0)
        s = self.user_price_sum.get(key, 0.0)
        val = (s + self.smoothing * user_base) / (cnt + self.smoothing)
        return clip_rating(val)

    def public_rating_score(self, place_id: int) -> float:
        info = self.place_info.get(int(place_id), {})
        public_rating = float(info.get("Rating", self.public_mean))
        z = (public_rating - self.public_mean) / self.public_std
        val = self.global_mean + 0.25 * z
        return clip_rating(val)

    def predict(self, user_id, place_id):
        info = self.place_info.get(int(place_id), {})
        category = info.get("Category", "Unknown")
        level = price_level(info.get("Price", 0))
        cat_score = self.smoothed_user_category_rating(user_id, category)
        price_score = self.smoothed_user_price_rating(user_id, level)
        pop_score = self.smoothed_place_rating(place_id)
        public_score = self.public_rating_score(place_id)
        pred = (
            self.w_category * cat_score
            + self.w_price * price_score
            + self.w_popularity * pop_score
            + self.w_public * public_score
        )
        return clip_rating(pred)


class HybridPreferenceBiasModel:
    def __init__(self, alpha=0.60, beta=0.30, gamma=0.10, base_params=None, content_params=None):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.base_params = base_params or {"reg": 20, "n_epochs": 20, "lr": 0.006, "random_state": 42}
        self.content_params = content_params or {"w_category": 0.45, "w_price": 0.20, "w_popularity": 0.25, "w_public": 0.10, "smoothing": 8.0}

    def fit(self, train_df, tour_df=None):
        self.base_model = BiasBaselineModel(**self.base_params)
        self.content_model = UserPreferenceContentModel(**self.content_params)
        self.base_model.fit(train_df, tour_df)
        self.content_model.fit(train_df, tour_df)
        return self

    def predict(self, user_id, place_id):
        base_pred = self.base_model.predict(user_id, place_id)
        content_pred = self.content_model.predict(user_id, place_id)
        pop_score = self.content_model.smoothed_place_rating(place_id)
        pred = self.alpha * base_pred + self.beta * content_pred + self.gamma * pop_score
        return clip_rating(pred)


@st.cache_resource(show_spinner=True)
def train_final_model(data: pd.DataFrame, tour: pd.DataFrame, tuning_summary: pd.DataFrame):
    params = {
        "alpha": 0.6,
        "beta": 0.3,
        "gamma": 0.1,
        "base_params": {"reg": 20, "n_epochs": 20, "lr": 0.006, "random_state": 42},
        "content_params": {"w_category": 0.45, "w_price": 0.2, "w_popularity": 0.25, "w_public": 0.1, "smoothing": 8.0},
    }

    if not tuning_summary.empty and "Best_Params" in tuning_summary.columns:
        try:
            best_row = tuning_summary.iloc[0]
            model_name = str(best_row.get("Model", ""))
            if "HybridPreferenceBias" in model_name:
                params = ast.literal_eval(best_row["Best_Params"])
        except Exception:
            pass

    model = HybridPreferenceBiasModel(**params)
    model.fit(data, tour)
    return model, params


def make_recommendations(model, user_id: int, rating: pd.DataFrame, tour: pd.DataFrame, top_n=10, diverse=False, max_per_category=3):
    rated_places = set(rating.loc[rating["User_Id"] == int(user_id), "Place_Id"].astype(int).tolist())
    candidates = tour[~tour["Place_Id"].astype(int).isin(rated_places)].copy()

    rows = []
    for row in candidates.itertuples(index=False):
        pid = int(getattr(row, "Place_Id"))
        pred = model.predict(user_id, pid)
        rows.append({
            "Place_Id": pid,
            "Place_Name": getattr(row, "Place_Name"),
            "Category": getattr(row, "Category"),
            "City": getattr(row, "City"),
            "Price": getattr(row, "Price"),
            "Harga": rupiah(getattr(row, "Price")),
            "Rating": getattr(row, "Rating"),
            "Time_Minutes": getattr(row, "Time_Minutes"),
            "Latitude": getattr(row, "Latitude"),
            "Longitude": getattr(row, "Longitude"),
            "Predicted_Rating": round(float(pred), 4),
        })

    rec = pd.DataFrame(rows)
    if rec.empty:
        return rec

    rec = rec.sort_values("Predicted_Rating", ascending=False).reset_index(drop=True)

    if not diverse:
        return rec.head(top_n).reset_index(drop=True)

    selected = []
    cat_count = {}
    for _, row in rec.iterrows():
        cat = row["Category"]
        if cat_count.get(cat, 0) < max_per_category:
            selected.append(row)
            cat_count[cat] = cat_count.get(cat, 0) + 1
        if len(selected) >= top_n:
            break

    return pd.DataFrame(selected).reset_index(drop=True)

# ============================================================
# USER PROFILE HELPERS
# ============================================================

def get_user_rating_persona(avg: float, count: int) -> str:
    """Klasifikasi gaya pemberian rating user."""
    if avg >= 4.2:
        return "Enthusiast 🌟"
    elif avg >= 3.5:
        return "Balanced ⚖️"
    elif avg >= 2.8:
        return "Selective 🔍"
    else:
        return "Critic 🎯"


def get_user_category_preference(user_id: int, rating: pd.DataFrame, tour: pd.DataFrame) -> pd.DataFrame:
    """Hitung rata-rata rating user per kategori."""
    user_ratings = rating[rating["User_Id"] == int(user_id)].copy()
    merged = user_ratings.merge(tour[["Place_Id", "Category"]], on="Place_Id", how="left")
    if merged.empty:
        return pd.DataFrame(columns=["Category", "Avg_Rating", "Count"])
    cat_pref = (
        merged.groupby("Category")
        .agg(Avg_Rating=("Place_Ratings", "mean"), Count=("Place_Ratings", "count"))
        .reset_index()
        .sort_values("Avg_Rating", ascending=False)
    )
    return cat_pref


def get_user_top_rated(user_id: int, rating: pd.DataFrame, tour: pd.DataFrame, n=5) -> pd.DataFrame:
    """Ambil tempat wisata yang diberi rating tertinggi user."""
    user_ratings = rating[rating["User_Id"] == int(user_id)].copy()
    merged = user_ratings.merge(
        tour[["Place_Id", "Place_Name", "Category", "City", "Price"]],
        on="Place_Id",
        how="left"
    )
    merged["Harga"] = merged["Price"].apply(rupiah)
    return merged.sort_values("Place_Ratings", ascending=False).head(n)


def find_similar_users(
    user_id: int,
    rating: pd.DataFrame,
    user: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Temukan user paling mirip berdasarkan cosine similarity vektor rating.
    Hanya membandingkan tempat yang sama-sama pernah dirating.
    """
    pivot = rating.pivot_table(index="User_Id", columns="Place_Id", values="Place_Ratings")
    if int(user_id) not in pivot.index:
        return pd.DataFrame()

    target = pivot.loc[int(user_id)].fillna(0).values
    norm_target = np.linalg.norm(target)
    if norm_target == 0:
        return pd.DataFrame()

    rows = []
    for uid in pivot.index:
        if int(uid) == int(user_id):
            continue
        other = pivot.loc[uid].fillna(0).values
        norm_other = np.linalg.norm(other)
        if norm_other == 0:
            continue
        sim = float(np.dot(target, other) / (norm_target * norm_other))
        rows.append({"User_Id": uid, "Similarity": sim})

    if not rows:
        return pd.DataFrame()

    sim_df = (
        pd.DataFrame(rows)
        .sort_values("Similarity", ascending=False)
        .head(top_n)
        .merge(user, on="User_Id", how="left")
    )
    sim_df["Similarity_Pct"] = (sim_df["Similarity"] * 100).round(1)
    return sim_df.reset_index(drop=True)

# ============================================================
# DATA INIT
# ============================================================

all_data = load_all_data()
tour_raw = all_data["tour"]
rating_raw = all_data["rating_raw"]
user_raw = all_data["user"]

missing_main_files = tour_raw.empty or rating_raw.empty or user_raw.empty

if missing_main_files:
    show_title()
    st.error("File dataset belum ditemukan. Pastikan tour.csv, tour_rating.csv, dan user.csv berada satu folder dengan app.py.")
    st.stop()

try:
    tour, user, rating, data, stats = prepare_data(tour_raw, rating_raw, user_raw)
except Exception as e:
    st.error(f"Gagal memproses dataset: {e}")
    st.stop()

base_results = all_data["base_results"]
tuning_summary = all_data["tuning_summary"]
tuning_detail = all_data["tuning_detail"]
final_summary = all_data["final_summary"]
rec_normal_file = all_data["rec_normal"]
rec_diverse_file = all_data["rec_diverse"]

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏝️ Menu")
page = st.sidebar.radio(
    "Pilih halaman",
    [
        "Dashboard",
        "Eksplorasi Data",
        "Evaluasi Model",
        "Hyperparameter Tuning",
        "Rekomendasi Wisata",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Dataset")
st.sidebar.write(f"User: **{stats['n_users']}**")
st.sidebar.write(f"Tempat wisata: **{stats['n_places']}**")
st.sidebar.write(f"Rating dedup: **{stats['dedup_rating_count']}**")
st.sidebar.write(f"Sparsity: **{stats['sparsity'] * 100:.2f}%**")

# ============================================================
# PAGE: DASHBOARD
# ============================================================

if page == "Dashboard":
    show_title()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah user", f"{stats['n_users']:,}")
    col2.metric("Jumlah tempat", f"{stats['n_places']:,}")
    col3.metric("Rating setelah dedup", f"{stats['dedup_rating_count']:,}")
    col4.metric("Sparsity matrix", f"{stats['sparsity'] * 100:.2f}%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Rating raw", f"{stats['raw_rating_count']:,}")
    col6.metric("Data gabungan", f"{stats['merged_count']:,}")
    col7.metric("Avg rating / user", f"{stats['avg_rating_per_user']:.2f}")
    col8.metric("Avg rating / tempat", f"{stats['avg_rating_per_place']:.2f}")

    st.markdown("### Ringkasan model final")
    if not final_summary.empty:
        st.dataframe(final_summary, use_container_width=True, hide_index=True)
    elif not tuning_summary.empty and not base_results.empty:
        global_best = base_results[base_results["Model"] == "GlobalMean"].sort_values("RMSE").head(1)
        best_tuned = tuning_summary.sort_values("RMSE").head(1)
        col_a, col_b = st.columns(2)
        with col_a:
            st.info("Baseline terbaik")
            st.dataframe(global_best[["Model", "Split", "RMSE", "MAE", "R2"]], hide_index=True, use_container_width=True)
        with col_b:
            st.success("Model rekomendasi final")
            st.dataframe(best_tuned[["Model", "Split", "RMSE", "MAE", "R2", "NDCG@10"]], hide_index=True, use_container_width=True)

    st.markdown("### Grafik performa model tuning")
    if not tuning_summary.empty:
        chart_df = tuning_summary.sort_values("RMSE", ascending=True).copy()
        fig = px.bar(
            chart_df,
            x="RMSE",
            y="Model",
            orientation="h",
            text="RMSE",
            title="Perbandingan RMSE Model Tuning",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=430)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Catatan interpretasi")
    st.markdown(
        """
        <div class="warning-box">
        <b>GlobalMean</b> memiliki RMSE terbaik sebagai baseline, tetapi tidak dipilih sebagai sistem rekomendasi utama karena tidak personal.
        Model final menggunakan <b>HybridPreferenceBias_Tuned</b> karena mempertimbangkan bias user, bias tempat, kategori, harga, popularitas, dan rating publik tempat wisata.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# PAGE: EXPLORATION
# ============================================================

elif page == "Eksplorasi Data":
    show_title()
    st.markdown("## Eksplorasi Data")

    tab1, tab2, tab3, tab4 = st.tabs(["Preview data", "Distribusi rating", "Kategori wisata", "User & tempat"])

    with tab1:
        st.markdown("### Dataset tempat wisata")
        st.dataframe(tour.head(20), use_container_width=True)
        st.markdown("### Dataset rating")
        st.dataframe(rating.head(20), use_container_width=True)
        st.markdown("### Dataset user")
        st.dataframe(user.head(20), use_container_width=True)

    with tab2:
        rating_dist = data["Place_Ratings"].value_counts().sort_index().reset_index()
        rating_dist.columns = ["Rating", "Jumlah"]
        fig = px.bar(rating_dist, x="Rating", y="Jumlah", text="Jumlah", title="Distribusi Rating User")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rating_dist, use_container_width=True, hide_index=True)

    with tab3:
        cat_count = tour["Category"].value_counts().reset_index()
        cat_count.columns = ["Category", "Jumlah"]
        fig = px.bar(cat_count, x="Jumlah", y="Category", orientation="h", text="Jumlah", title="Jumlah Tempat Wisata per Kategori")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=470)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cat_count, use_container_width=True, hide_index=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            rating_per_user = rating.groupby("User_Id")["Place_Id"].count().reset_index(name="Jumlah_Rating")
            fig = px.histogram(rating_per_user, x="Jumlah_Rating", nbins=20, title="Distribusi Jumlah Rating per User")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(rating_per_user.describe().T, use_container_width=True)
        with col2:
            rating_per_place = rating.groupby("Place_Id")["User_Id"].count().reset_index(name="Jumlah_Rating")
            fig = px.histogram(rating_per_place, x="Jumlah_Rating", nbins=20, title="Distribusi Jumlah Rating per Tempat")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(rating_per_place.describe().T, use_container_width=True)

# ============================================================
# PAGE: MODEL EVALUATION
# ============================================================

elif page == "Evaluasi Model":
    show_title()
    st.markdown("## Evaluasi Model Dasar")

    if base_results.empty:
        st.warning("File hasil_evaluasi_split_model_dasar.csv belum ditemukan.")
    else:
        metric = st.selectbox("Pilih metrik untuk grafik", ["RMSE", "MAE", "R2", "HitRate@10", "Precision@10", "Recall@10", "NDCG@10", "MAP@10", "MRR@10"], index=0)
        sort_ascending = metric in ["RMSE", "MAE"]
        plot_df = base_results.sort_values(metric, ascending=sort_ascending)

        fig = px.bar(
            plot_df,
            x=metric,
            y="Model",
            color="Split",
            orientation="h",
            title=f"Perbandingan {metric} Model Dasar",
            hover_data=["MAE", "RMSE", "R2", "NDCG@10"],
        )
        fig.update_layout(height=620, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Tabel hasil evaluasi")
        show_cols = ["Split", "Model", "Train_Size", "Test_Size", "MAE", "RMSE", "R2", "HitRate@10", "Precision@10", "Recall@10", "NDCG@10", "MAP@10", "MRR@10"]
        show_cols = [c for c in show_cols if c in base_results.columns]
        st.dataframe(base_results[show_cols], use_container_width=True, hide_index=True)
        download_df_button(base_results, "hasil_evaluasi_split_model_dasar.csv", "Download hasil evaluasi model dasar")

# ============================================================
# PAGE: TUNING
# ============================================================

elif page == "Hyperparameter Tuning":
    show_title()
    st.markdown("## Hyperparameter Tuning")

    if tuning_summary.empty:
        st.warning("File hasil_tuning_ringkasan.csv belum ditemukan.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                tuning_summary.sort_values("RMSE"),
                x="RMSE",
                y="Model",
                orientation="h",
                text="RMSE",
                title="RMSE Model Tuning",
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=430)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                tuning_summary.sort_values("NDCG@10", ascending=False),
                x="NDCG@10",
                y="Model",
                orientation="h",
                text="NDCG@10",
                title="NDCG@10 Model Tuning",
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=430)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Ringkasan hasil tuning")
        st.dataframe(tuning_summary, use_container_width=True, hide_index=True)
        download_df_button(tuning_summary, "hasil_tuning_ringkasan.csv", "Download ringkasan tuning")

        if not tuning_detail.empty:
            st.markdown("### Detail validasi tuning")
            st.dataframe(tuning_detail, use_container_width=True, hide_index=True)
            download_df_button(tuning_detail, "hasil_tuning_detail.csv", "Download detail tuning")

# ============================================================
# PAGE: RECOMMENDATION
# ============================================================

elif page == "Rekomendasi Wisata":
    show_title()
    st.markdown("## Rekomendasi Wisata")

    with st.sidebar:
        st.divider()
        st.caption("Pengaturan rekomendasi")
        available_users = sorted(rating["User_Id"].unique().tolist())
        selected_user = st.selectbox("Pilih User_Id", available_users, index=0)
        top_n = st.slider("Jumlah rekomendasi", min_value=5, max_value=20, value=10)

    model, params = train_final_model(data, tour, tuning_summary)

    rec = make_recommendations(
        model,
        int(selected_user),
        rating,
        tour,
        top_n=top_n,
        diverse=False,
        max_per_category=DIVERSE_MAX_PER_CATEGORY,
    )

    user_profile = user[user["User_Id"] == int(selected_user)]
rated_user = rating[rating["User_Id"] == int(selected_user)]
st.markdown("### Profil user")

age_val = "-"
loc_val = "-"
if not user_profile.empty:
    if "Age" in user_profile.columns:
        age_val = int(user_profile.iloc[0]["Age"])
    if "Location" in user_profile.columns:
        loc_val = str(user_profile.iloc[0]["Location"])

avg_rating_user = float(rated_user["Place_Ratings"].mean()) if not rated_user.empty else 0.0
persona = get_user_rating_persona(avg_rating_user, len(rated_user))

# Baris metrik ringkas
col1, col2, col3, col4 = st.columns(4)
col1.metric("User ID", selected_user)
col2.metric("Lokasi", loc_val)
col3.metric("Usia", age_val)
col4.metric("Persona", persona)

# Dua kolom: preferensi kategori & riwayat rating
cat_col, hist_col = st.columns(2)

with cat_col:
    st.markdown("#### Preferensi kategori")
    cat_pref = get_user_category_preference(int(selected_user), rating, tour)
    if not cat_pref.empty:
        max_avg = cat_pref["Avg_Rating"].max()
        for _, row in cat_pref.iterrows():
            bar_pct = int((row["Avg_Rating"] / 5.0) * 100)
            st.markdown(
                f"""
                <div style="margin-bottom:8px">
                  <div style="display:flex;justify-content:space-between;font-size:13px">
                    <span>{row['Category']}</span>
                    <span style="color:#888">{row['Avg_Rating']:.2f} ★ &nbsp;({int(row['Count'])} ulasan)</span>
                  </div>
                  <div style="background:rgba(49,51,63,.1);border-radius:4px;height:7px;overflow:hidden;margin-top:3px">
                    <div style="width:{bar_pct}%;height:7px;background:#4F8EF7;border-radius:4px"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("Belum ada data rating.")

with hist_col:
    st.markdown("#### Riwayat rating tertinggi")
    top_rated = get_user_top_rated(int(selected_user), rating, tour, n=6)
    if not top_rated.empty:
        display_hist = top_rated[["Place_Name", "Category", "Harga", "Place_Ratings"]].rename(
            columns={"Place_Ratings": "Rating", "Place_Name": "Tempat"}
        )
        st.dataframe(display_hist, use_container_width=True, hide_index=True)
    else:
        st.caption("Belum ada riwayat rating.")

# Distribusi rating user (chart kecil)
if not rated_user.empty:
    with st.expander("Lihat distribusi rating user ini"):
        dist = rated_user["Place_Ratings"].value_counts().sort_index().reset_index()
        dist.columns = ["Rating", "Jumlah"]
        fig_dist = px.bar(
            dist, x="Rating", y="Jumlah",
            text="Jumlah",
            title=f"Distribusi rating yang diberikan User {selected_user}",
            height=300,
        )
        fig_dist.update_traces(textposition="outside")
        fig_dist.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig_dist, use_container_width=True)

# User serupa
st.markdown("#### User dengan selera serupa")
with st.spinner("Menghitung kesamaan antar user..."):
    similar_users = find_similar_users(int(selected_user), rating, user, top_n=5)

if not similar_users.empty:
    show_sim_cols = ["User_Id", "Location", "Age", "Similarity_Pct"]
    show_sim_cols = [c for c in show_sim_cols if c in similar_users.columns]
    similar_users_display = similar_users[show_sim_cols].copy()
    similar_users_display = similar_users_display.rename(columns={
        "User_Id": "User ID",
        "Location": "Lokasi",
        "Age": "Usia",
        "Similarity_Pct": "Kesamaan (%)",
    })
    st.dataframe(similar_users_display, use_container_width=True, hide_index=True)

    # Kategori favorit tiap user serupa sebagai konteks
    with st.expander("Lihat preferensi user serupa"):
        for _, sim_row in similar_users.head(3).iterrows():
            sim_uid = int(sim_row["User_Id"])
            sim_loc = sim_row.get("Location", "-")
            sim_pct = sim_row["Similarity_Pct"]
            st.markdown(f"**User {sim_uid}** ({sim_loc}) — kesamaan {sim_pct}%")
            sim_cat = get_user_category_preference(sim_uid, rating, tour)
            if not sim_cat.empty:
                top_cats = ", ".join(sim_cat.head(3)["Category"].tolist())
                st.caption(f"Kategori favorit: {top_cats}")
            st.divider()
else:
    st.caption("Tidak cukup data untuk menemukan user serupa.")

st.markdown("---")

    st.markdown("### Hasil rekomendasi")
    if rec.empty:
        st.warning("Tidak ada rekomendasi. Kemungkinan user sudah memiliki rating untuk semua tempat.")
    else:
        display_cols = ["Place_Name", "Category", "Harga", "Rating", "Time_Minutes", "Predicted_Rating"]
        st.dataframe(rec[display_cols], use_container_width=True, hide_index=True)
        download_df_button(rec, f"rekomendasi_user_{selected_user}.csv", "Download rekomendasi user ini")

        fig = px.bar(
            rec.sort_values("Predicted_Rating"),
            x="Predicted_Rating",
            y="Place_Name",
            color="Category",
            orientation="h",
            text="Predicted_Rating",
            title=f"Top {top_n} Rekomendasi untuk User {selected_user}",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(height=max(450, top_n * 42), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        map_df = rec[["Latitude", "Longitude", "Place_Name", "Predicted_Rating"]].dropna()
        if not map_df.empty:
            st.markdown("### Peta lokasi rekomendasi")
            st.map(map_df.rename(columns={"Latitude": "lat", "Longitude": "lon"}), latitude="lat", longitude="lon", size=80)

    with st.expander("Lihat parameter model final"):
        st.json(params)
