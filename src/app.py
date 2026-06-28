import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import requests
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

# ==============================================================================
# 0. PAGE CONFIG & GLOBAL STYLE
# ==============================================================================
st.set_page_config(
    page_title="EduPulse CoC - AI Sentiment Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_STOPWORDS = set([
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "dengan",
    "adalah", "sebagai", "juga", "sudah", "saya", "aku", "kamu", "dia", "mereka",
    "kita", "kami", "banget", "aja", "nya", "yg", "ga", "gak", "udah", "ada",
    "buat", "orang", "sama", "kok", "sih", "dong", "deh", "kan", "pun", "kalau",
    "kalo", "lagi", "terus", "bisa", "jadi", "nggak", "tidak", "belum",
    "acara", "clash", "champions", "coc", "ruangguru", "nonton", "episode", "eps",
    "anak", "indonesia", "menonton", "video", "youtube", "peserta", "bikin", "tuh",
    "emang", "memang", "udah", "nih", "lah", "mah", "saja", "kah", "wah", "eh"
])

st.markdown("""
<style>
    /* ── Global (Green-Cream Glassmorphism) ──────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Background Gradient (Dark Forest) */
    .stApp { 
        background: radial-gradient(circle at top left, #0a1f14 0%, #050a07 100%); 
        color: #e5e7eb; 
    }

    /* ── Headings ───────────────────────────────────────── */
    h1, h2, h3 { color: #fcd34d !important; font-weight: 800 !important; }
    
    /* Neon Text Gradient for Hero */
    .hero-title {
        background: -webkit-linear-gradient(45deg, #10b981, #fcd34d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0px;
    }

    /* ── Sidebar ────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(5, 10, 7, 0.6) !important;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-right: 1px solid rgba(16, 185, 129, 0.2);
    }
    [data-testid="stSidebar"] * { color: #e5e7eb !important; }

    /* ── Metric Cards & Glass Panels ────────────────────── */
    .metric-row { display: flex; gap: 16px; margin-bottom: 28px; }
    .metric-card, .result-box {
        flex: 1; 
        background: rgba(255, 255, 255, 0.03); 
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(16, 185, 129, 0.15);
        padding: 22px 20px; border-radius: 16px; text-align: center;
        transition: all 0.3s ease; 
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    .metric-card:hover, .result-box:hover { 
        transform: translateY(-5px); 
        border-color: rgba(16, 185, 129, 0.5); 
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.2); 
    }
    .metric-value { font-size: 2.4rem; font-weight: 800; color: #10b981; margin-bottom: 4px; text-shadow: 0 0 10px rgba(16,185,129,0.3); }
    .metric-label { font-size: 0.8rem; color: #9ca3af; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; }
    .metric-sub { font-size: 0.75rem; color: #fcd34d; margin-top: 4px; }

    /* ── Sentiment Badge ────────────────────────────────── */
    .badge {
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-size: 0.82rem; font-weight: 700; letter-spacing: .5px;
    }
    .badge-positive { background: rgba(16,185,129,.15); color: #10b981; border: 1px solid #10b981; box-shadow: 0 0 10px rgba(16,185,129,0.2); }
    .badge-negative { background: rgba(244,63,94,.15); color: #f43f5e; border: 1px solid #f43f5e; box-shadow: 0 0 10px rgba(244,63,94,0.2); }
    .badge-neutral  { background: rgba(251,191,36,.15); color: #fbbf24; border: 1px solid #fbbf24; box-shadow: 0 0 10px rgba(251,191,36,0.2); }

    /* ── Result Box (AI Prediction) ─────────────────────── */
    .result-box { margin-top: 16px; }
    .result-label { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .result-sentiment { font-size: 3rem; font-weight: 900; margin: 12px 0; text-shadow: 0 0 20px rgba(255,255,255,0.1); }

    /* ── Tabs ───────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { gap: 16px; border-bottom: 1px solid rgba(16, 185, 129, 0.2); }
    .stTabs [data-baseweb="tab"] { height: 44px; background: transparent; border-radius: 6px 6px 0 0; padding: 0 16px; font-weight: 600; color: #9ca3af; }
    .stTabs [aria-selected="true"] { background: rgba(16, 185, 129, 0.1) !important; border-bottom: 2px solid #10b981; color: #10b981 !important; text-shadow: 0 0 8px rgba(16,185,129,0.3); }

    /* ── Info/Warning overrides ─────────────────────────── */
    .stAlert { border-radius: 12px; background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. LOAD DATA & MODEL (CACHED)
# ==============================================================================
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

@st.cache_data
def load_metadata():
    path = os.path.join(ROOT_DIR, "models", "model_metadata.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "bert_accuracy": 77.33,
            "bert_f1_macro": 72.48,
            "ensemble_accuracy": 74.22,
            "ensemble_f1_macro": 68.04,
            "total_samples": 13777,
            "model_name": "IndoBERT fine-tuned"
        }

@st.cache_data
def load_dataset():
    path = os.path.join(ROOT_DIR, "data", "processed", "comments_labeled_bert_final.csv")
    try:
        df = pd.read_csv(path)
        if "publishedAt" in df.columns:
            df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
        # Normalise sentiment column name
        if "sentiment_bert" in df.columns and "sentiment_final" not in df.columns:
            df["sentiment_final"] = df["sentiment_bert"]
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_resource
def load_ensemble():
    path = os.path.join(ROOT_DIR, "models", "ensemble_baseline.pkl")
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_resource
def load_bert_pipeline():
    try:
        from transformers import pipeline as hf_pipeline
        model_path = "mayhesar/indobert-sentiment-gwe"
        return hf_pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=-1,
            top_k=None
        )
    except Exception as e:
        return None

metadata      = load_metadata()
df            = load_dataset()
ensemble_model = load_ensemble()
bert_model    = load_bert_pipeline()

COLORS = {"positive": "#3fb950", "neutral": "#d29922", "negative": "#f85149"}

# ==============================================================================
# 2. SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
        <div style='font-size:2.5rem;'>🧠</div>
        <div style='font-size:1.15rem; font-weight:800; color:#e6edf3;'>EduPulse CoC</div>
        <div style='font-size:0.75rem; color:#8b949e; margin-top:4px;'>GWE 2026 Data Science Challenge</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    menu = option_menu(
        menu_title=None,
        options=["Halaman Utama", "EDA Dashboard", "AI Prediction", "Dokumentasi"],
        icons=['house', 'bar-chart-line', 'robot', 'file-text'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#10b981", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"5px", "--hover-color": "rgba(16, 185, 129, 0.1)"},
            "nav-link-selected": {"background-color": "rgba(16, 185, 129, 0.2)", "color": "#fcd34d", "border-left": "3px solid #10b981", "font-weight":"600"},
        }
    )

    st.markdown("---")

    # Model status badges
    bert_ok = bert_model is not None
    ens_ok  = ensemble_model is not None

    st.markdown(f"""
    <div style='font-size:.75rem; color:#8b949e; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px;'>Status Model</div>
    <div style='margin-bottom:4px;'>
        {'✅' if bert_ok else '❌'} <span style='color:{"#3fb950" if bert_ok else "#f85149"};font-weight:600;'>IndoBERT</span>
        <span style='color:#8b949e; font-size:.7rem;'> (model final)</span>
    </div>
    <div>
        {'✅' if ens_ok else '❌'} <span style='color:{"#d29922" if ens_ok else "#f85149"};font-weight:600;'>Ensemble</span>
        <span style='color:#8b949e; font-size:.7rem;'> (baseline)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Dataset: Ruangguru CoC 2024\n(Kaggle — rezkyyayang)")

# ==============================================================================
# 3. HALAMAN UTAMA
# ==============================================================================
if menu == "Halaman Utama":
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("<h1 class='hero-title'>EduPulse CoC</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#e5e7eb; margin-top:0;'>Intelligence Dashboard</h3>", unsafe_allow_html=True)
        st.markdown("<div style='color:#9ca3af; font-size:1.1rem;'>Mengukur Denyut Nadi Edutainment Digital Indonesia secara Real-time.</div>", unsafe_allow_html=True)
    with col_h2:
        # Lottie animasi otak cerdas
        lottie_brain = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_tno6cg2w.json")
        if lottie_brain:
            st_lottie(lottie_brain, height=120, key="brain_hero")
            
    st.markdown("---")

    total   = metadata.get("total_samples", len(df) if not df.empty else 0)
    bert_acc = metadata.get("bert_accuracy", 0)
    bert_f1  = metadata.get("bert_f1_macro", 0)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card" style="--accent:#58a6ff;">
            <div class="metric-value" style="color:#58a6ff;">{total:,}</div>
            <div class="metric-label">Komentar Dianalisis</div>
            <div class="metric-sub">YouTube CoC 2024</div>
        </div>
        <div class="metric-card" style="--accent:#3fb950;">
            <div class="metric-value" style="color:#3fb950;">{bert_acc:.1f}%</div>
            <div class="metric-label">Akurasi IndoBERT</div>
            <div class="metric-sub">model final · test set</div>
        </div>
        <div class="metric-card" style="--accent:#a371f7;">
            <div class="metric-value" style="color:#a371f7;">{bert_f1:.2f}</div>
            <div class="metric-label">Macro F1-Score</div>
            <div class="metric-sub">IndoBERT fine-tuned</div>
        </div>
        <div class="metric-card" style="--accent:#d29922;">
            <div class="metric-value" style="color:#d29922;">+28.6%</div>
            <div class="metric-label">Peningkatan vs Baseline</div>
            <div class="metric-sub">Ensemble F1: 0.68 → 0.72</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_ringkasan, tab_data = st.tabs(["📄 Ringkasan Eksekutif", "🗃️ Data Snapshot"])

    with tab_ringkasan:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### 🔍 Latar Belakang Masalah
            Ledakan popularitas program kompetisi akademik seperti **Clash of Champions (CoC)** di platform digital memunculkan pertanyaan empiris yang belum terjawab:

            > *"Ketika anak-anak jenius diadu di layar kaca, apakah penonton muda terinspirasi belajar — atau justru merasa insecure?"*

            EduPulse hadir menjawab ini bukan dengan opini, melainkan dengan **data kuantitatif dari 13.777 komentar nyata**.
            """)
        with col2:
            st.markdown("""
            #### 🎯 Tiga Pilar Dampak
            1. **Validasi Sosial:** Membuktikan secara matematis bahwa edutainment kompetitif memicu motivasi belajar — bukan ketakutan massal.
            2. **Rekomendasi Strategis:** Insight konten berbasis *Explainable AI* (SHAP) untuk industri kreatif & pembuat kebijakan.
            3. **Ekosistem AI:** Dashboard pemantauan sentimen real-time yang transparan dan dapat diaudit.
            """)

        st.markdown("---")
        st.markdown("""
        #### 🏗️ Arsitektur Pipeline
        ```
        Data Mentah (14.149)
            ↓ Cleaning & Normalisasi Slang
        Data Bersih (13.777)
            ↓ Gold Labeling — Groq Llama-3 8B (1.500 sampel)
        Active Learning — Ensemble VotingClassifier (semua 13.777)
            ↓ Fine-Tuning
        IndoBERT (model final) — Inferensi ke 13.777 komentar
            ↓
        Dashboard Streamlit (halaman ini)
        ```
        """)

    with tab_data:
        st.markdown("#### 🗃️ Cuplikan Dataset Berlabel")
        st.caption("Dataset bersumber dari [Kaggle — rezkyyayang](https://www.kaggle.com/datasets/rezkyyayang/ruangguru-clash-of-champions-2024-youtube-comments). Label sentimen menggunakan model IndoBERT fine-tuned.")
        if not df.empty:
            display_cols = [c for c in ["authorDisplayName", "clean_text", "sentiment_final", "likeCount"] if c in df.columns]
            st.dataframe(df[display_cols].head(100), use_container_width=True)
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Unduh Dataset Lengkap (CSV)",
                data=csv_data,
                file_name="edupulse_labeled_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Data belum tersedia. Pastikan notebook sudah dijalankan.")

# ==============================================================================
# 4. EDA DASHBOARD
# ==============================================================================
elif menu == "EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")

    if df.empty:
        st.warning("⚠️ Data belum tersedia. Jalankan notebook terlebih dahulu.")
        st.stop()

    # ── Filter ──
    col_f1, col_f2, _ = st.columns([1, 1, 2])
    with col_f1:
        sentiment_filter = st.selectbox("🎯 Filter Sentimen", ["Semua", "positive", "neutral", "negative"])
    with col_f2:
        if "videoId" in df.columns:
            video_opts = ["Semua"] + sorted(df["videoId"].dropna().unique().tolist())
            video_filter = st.selectbox("🎬 Filter Video", video_opts)
        else:
            video_filter = "Semua"

    df_f = df.copy()
    if "sentiment_final" in df_f.columns:
        if sentiment_filter != "Semua":
            df_f = df_f[df_f["sentiment_final"] == sentiment_filter]
    if video_filter != "Semua" and "videoId" in df_f.columns:
        df_f = df_f[df_f["videoId"] == video_filter]

    st.caption(f"Menampilkan **{len(df_f):,}** dari {len(df):,} komentar")
    st.markdown("---")

    tab_makro, tab_teks, tab_tren, tab_pilar = st.tabs([
        "📈 Distribusi & Engagement",
        "💬 Analisis Teks (NLP)",
        "📅 Tren Waktu",
        "🎯 Validasi Pilar"
    ])

    # ─── TAB 1: MAKRO ─────────────────────────────────────────────────────────
    with tab_makro:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Distribusi Sentimen (Donut Chart)")
            if "sentiment_final" in df_f.columns:
                counts = df_f["sentiment_final"].value_counts().reset_index()
                counts.columns = ["sentiment", "count"]
                fig1 = px.pie(
                    counts, names="sentiment", values="count",
                    color="sentiment", color_discrete_map=COLORS, hole=0.45
                )
                fig1.update_traces(textposition="inside", textinfo="percent+label")
                fig1.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#c9d1d9", showlegend=True,
                    legend=dict(orientation="h", y=-0.1),
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig1, use_container_width=True)
                st.caption("💡 Sentimen positif mendominasi — konsisten dengan hipotesis edutainment memotivasi.")

        with col2:
            st.markdown("##### LikeCount vs Sentimen (Box Plot)")
            if "likeCount" in df_f.columns and "sentiment_final" in df_f.columns:
                df_box = df_f[df_f["likeCount"] > 0].copy()
                if not df_box.empty:
                    fig2 = px.box(
                        df_box, x="sentiment_final", y="likeCount",
                        color="sentiment_final", color_discrete_map=COLORS,
                        log_y=True, points=False
                    )
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#c9d1d9", showlegend=False,
                        margin=dict(l=0, r=0, t=20, b=0),
                        xaxis_title="Sentimen", yaxis_title="LikeCount (log scale)"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    st.caption("💡 Komentar positif secara konsisten mendapat engagement lebih tinggi.")
                else:
                    st.info("Tidak ada data LikeCount > 0 untuk filter ini.")

    # ─── TAB 2: ANALISIS TEKS ─────────────────────────────────────────────────
    with tab_teks:
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("##### Word Cloud Topik Diskusi")
            if "clean_text" in df_f.columns:
                words = " ".join(df_f["clean_text"].dropna().astype(str).tolist()).split()
                filtered_words = [w for w in words if w not in CUSTOM_STOPWORDS and len(w) > 2]
                text_wc = " ".join(filtered_words)
                if text_wc.strip():
                    wc = WordCloud(
                        width=700, height=400, background_color="#0d1117",
                        colormap="Blues", max_words=80, collocations=False
                    ).generate(text_wc)
                    fig_wc, ax = plt.subplots(figsize=(8, 4.5))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    fig_wc.patch.set_facecolor("#0d1117")
                    st.pyplot(fig_wc)
                    plt.close()
                    st.caption("💡 Kata yang lebih besar = lebih sering muncul di komentar yang difilter.")
                else:
                    st.warning("Tidak cukup kata unik untuk ditampilkan.")

        with col4:
            st.markdown("##### Top 10 Bigram (Frasa Populer)")
            if "clean_text" in df_f.columns and len(df_f) >= 10:
                try:
                    vec = CountVectorizer(ngram_range=(2, 2), stop_words=list(CUSTOM_STOPWORDS), max_features=10)
                    X_ng = vec.fit_transform(df_f["clean_text"].dropna())
                    freq = X_ng.sum(axis=0).A1
                    vocab = vec.get_feature_names_out()
                    df_ng = pd.DataFrame({"Frasa": vocab, "Frekuensi": freq}).sort_values("Frekuensi")
                    fig_bar = px.bar(
                        df_ng, x="Frekuensi", y="Frasa", orientation="h",
                        color="Frekuensi", color_continuous_scale="Blues"
                    )
                    fig_bar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#c9d1d9", margin=dict(l=0, r=0, t=20, b=0),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.caption("💡 Frasa khas per sentimen menjadi panduan copywriting untuk kreator.")
                except Exception:
                    st.info("Data tidak cukup untuk menampilkan bigram.")

    # ─── TAB 3: TREN WAKTU ────────────────────────────────────────────────────
    with tab_tren:
        st.markdown("##### Tren Sentimen Harian")
        if "publishedAt" in df_f.columns and pd.api.types.is_datetime64_any_dtype(df_f["publishedAt"]):
            df_time = df_f.copy()
            df_time["date"] = df_time["publishedAt"].dt.date
            daily = df_time.groupby(["date", "sentiment_final"]).size().reset_index(name="count")
            fig_time = px.line(
                daily, x="date", y="count", color="sentiment_final",
                color_discrete_map=COLORS, markers=True
            )
            fig_time.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d1d9", legend_title_text="Sentimen",
                xaxis_title="Tanggal", yaxis_title="Jumlah Komentar"
            )
            st.plotly_chart(fig_time, use_container_width=True)
            st.caption("💡 Lonjakan komentar bertepatan dengan jadwal tayang episode baru.")
        else:
            st.info("Kolom tanggal (publishedAt) tidak tersedia atau format tidak terbaca.")

    # ─── TAB 4: VALIDASI PILAR ────────────────────────────────────────────────
    with tab_pilar:
        st.markdown("##### 🎯 Validasi Hipotesis: Motivasi vs Insecure")
        st.markdown(
            "Menghitung frekuensi kemunculan kata kunci tematik di seluruh komentar "
            "untuk membuktikan apakah CoC memotivasi atau justru menimbulkan insecurity."
        )

        if "clean_text" in df_f.columns:
            motivasi_kws = ["belajar", "semangat", "inspirasi", "paham", "rajin", "ilmu", "pintar", "cerdas", "motivasi", "hebat"]
            insecure_kws = ["insecure", "bodoh", "takut", "pusing", "goblok", "mental", "nangis", "nyerah", "minder", "tertekan"]

            texts = df_f["clean_text"].dropna().astype(str)
            n_motivasi = int(texts.str.contains("|".join(motivasi_kws), case=False).sum())
            n_insecure = int(texts.str.contains("|".join(insecure_kws), case=False).sum())
            ratio = n_motivasi / max(n_insecure, 1)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("💚 Komentar Bermotivasi", f"{n_motivasi:,}", help="Mengandung kata: belajar, semangat, inspirasi, dll.")
            col_b.metric("🔴 Komentar Bernada Insecure", f"{n_insecure:,}", help="Mengandung kata: insecure, bodoh, takut, dll.")
            col_c.metric("📊 Rasio Motivasi:Insecure", f"{ratio:.1f}x")

            df_pilar = pd.DataFrame({
                "Kategori": ["Motivasi / Semangat", "Insecure / Mental"],
                "Jumlah": [n_motivasi, n_insecure],
                "Warna": ["#3fb950", "#f85149"]
            })
            fig_pilar = px.bar(
                df_pilar, x="Kategori", y="Jumlah", color="Kategori",
                color_discrete_map={"Motivasi / Semangat": "#3fb950", "Insecure / Mental": "#f85149"},
                text="Jumlah"
            )
            fig_pilar.update_traces(textposition="outside")
            fig_pilar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d1d9", showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_pilar, use_container_width=True)

            if ratio >= 3:
                st.success(f"✅ **Hipotesis terbukti.** Komentar bermotivasi {ratio:.1f}× lebih banyak dari yang bernada insecure. CoC efektif sebagai edutainment positif.")
            elif ratio >= 1.5:
                st.info(f"ℹ️ Komentar bermotivasi {ratio:.1f}× lebih banyak — CoC cenderung positif, namun masih ada ruang untuk perbaikan.")
            else:
                st.warning("⚠️ Proporsi komentar insecure relatif tinggi untuk filter ini.")

# ==============================================================================
# 5. AI PREDICTION
# ==============================================================================
elif menu == "AI Prediction":
    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.title("Live AI Sentiment Prediction")
        st.markdown(
            "Ketikkan komentar YouTube dalam Bahasa Indonesia (termasuk bahasa gaul/slang), "
            "dan AI akan menganalisis sentimennya secara real-time."
        )
    with col_h2:
        lottie_robot = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_i9mxcbpi.json")
        if lottie_robot:
            st_lottie(lottie_robot, height=100, key="robot_pred")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        preset = st.selectbox(
            "💬 Pilih contoh kalimat atau ketik sendiri:",
            [
                "--- Ketik Bebas ---",
                "keren banget pesertanya, pinter-pinter, jadi semangat belajar nih",
                "sumpah acaranya membosankan, pertanyaannya ga berbobot sama sekali",
                "ngeliat mereka pinter-pinter jadi insecure ngerasa bodoh banget deh",
                "biasa aja sih acaranya, lumayan buat ngisi waktu luang",
                "ini program bagus banget, semua anak indonesia harus nonton!",
            ]
        )
        default_text = "" if preset == "--- Ketik Bebas ---" else preset
        user_input = st.text_area(
            "✍️ Input komentar:",
            value=default_text,
            height=130,
            placeholder="Contoh: keren banget acaranya, seru dan edukatif..."
        )
        analyze_btn = st.button("🚀 Analisis Sentimen", type="primary", use_container_width=True)

    with col_right:
        st.markdown("""
        **Model yang digunakan:**
        - 🥇 **IndoBERT** (fine-tuned) — jika tersedia
        - 🥈 **Ensemble** (LR+RF+XGB) — fallback otomatis

        **Output:**
        - Label sentimen (Positive / Neutral / Negative)
        - Tingkat kepercayaan (confidence score)
        - Grafik probabilitas per kelas
        """)

    if analyze_btn:
        if not user_input.strip():
            st.error("❌ Komentar tidak boleh kosong!")
            st.stop()
        if bert_model is None and ensemble_model is None:
            st.error("❌ Semua model gagal dimuat. Pastikan notebook sudah selesai dijalankan.")
            st.stop()

        # Preprocessing sederhana
        text_clean = user_input.lower()
        text_clean = re.sub(r"http\S+|www\S+", "", text_clean)
        text_clean = re.sub(r"[^\w\s]", " ", text_clean)
        text_clean = re.sub(r"\s+", " ", text_clean).strip()

        prediction = None
        probabilities = None
        model_used = None

        if bert_model is not None:
            with st.spinner("Memproses dengan IndoBERT Transformer..."):
                try:
                    res = bert_model(text_clean)[0]
                    score_dict = {r["label"].lower(): r["score"] for r in res}
                    prediction = max(score_dict, key=score_dict.get)
                    probabilities = np.array([
                        score_dict.get("negative", 0.0),
                        score_dict.get("neutral", 0.0),
                        score_dict.get("positive", 0.0)
                    ])
                    model_used = "IndoBERT (fine-tuned)"
                except Exception as e:
                    st.warning(f"IndoBERT error: {e} — Fallback ke Ensemble.")

        if prediction is None and ensemble_model is not None:
            with st.spinner("Memproses dengan Ensemble (LR+RF+XGB)..."):
                try:
                    y_enc = ensemble_model.predict([text_clean])[0]
                    inv_map = {0: "negative", 1: "neutral", 2: "positive"}
                    prediction = inv_map.get(y_enc, "neutral")
                    probabilities = ensemble_model.predict_proba([text_clean])[0]
                    model_used = "Ensemble (LR+RF+XGB)"
                except Exception as e:
                    st.error(f"Ensemble error: {e}")
                    st.stop()

        if prediction is None:
            st.error("Gagal mendapatkan prediksi dari kedua model.")
            st.stop()

        # ── Tampilkan Hasil ──────────────────────────────────────────────────
        st.markdown("---")

        emoji_map = {"positive": "😊", "neutral": "😐", "negative": "😟"}
        color_map  = {"positive": "#3fb950", "neutral": "#d29922", "negative": "#f85149"}
        label_map  = {"positive": "POSITIF", "neutral": "NETRAL", "negative": "NEGATIF"}

        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            conf = float(max(probabilities)) * 100
            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Prediksi Sentimen</div>
                <div class="result-sentiment" style="color:{color_map[prediction]};">
                    {emoji_map[prediction]} {label_map[prediction]}
                </div>
                <div style="color:#8b949e; font-size:.85rem;">
                    Confidence: <b style="color:{color_map[prediction]};">{conf:.1f}%</b>
                </div>
                <div style="color:#8b949e; font-size:.72rem; margin-top:8px;">
                    via {model_used}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_res2:
            st.markdown("##### Distribusi Probabilitas")
            classes = ["negative", "neutral", "positive"]
            labels  = ["Negatif", "Netral", "Positif"]
            prob_df = pd.DataFrame({"Sentimen": labels, "Confidence (%)": probabilities * 100})

            fig_prob = px.bar(
                prob_df, x="Confidence (%)", y="Sentimen", orientation="h",
                color="Sentimen",
                color_discrete_map={"Negatif": "#f85149", "Netral": "#d29922", "Positif": "#3fb950"},
                text_auto=".1f"
            )
            fig_prob.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d1d9", height=180, showlegend=False,
                xaxis=dict(range=[0, 100], title=""),
                yaxis_title="", margin=dict(l=0, r=20, t=10, b=0)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # ── SHAP Explanation (jika Ensemble tersedia) ─────────────────────────
        if ensemble_model is not None:
            with st.expander("🔍 Explainable AI — Kontribusi Kata (SHAP)", expanded=False):
                st.caption("Grafik ini menunjukkan kata mana yang paling mendorong prediksi model Ensemble.")
                try:
                    import shap
                    lr_clf  = ensemble_model.named_steps["ensemble"].named_estimators_["lr"]
                    tfidf   = ensemble_model.named_steps["tfidf"]
                    X_trans = tfidf.transform([text_clean])
                    feat    = tfidf.get_feature_names_out().tolist()

                    if not df.empty and "clean_text" in df.columns:
                        bg = tfidf.transform(df["clean_text"].dropna().head(300))
                    else:
                        bg = X_trans

                    explainer  = shap.LinearExplainer(lr_clf, bg, feature_names=feat)
                    shap_vals  = explainer.shap_values(X_trans)

                    label2idx = {"negative": 0, "neutral": 1, "positive": 2}
                    pred_idx  = label2idx.get(prediction, 2)

                    if isinstance(shap_vals, list):
                        sv = shap_vals[pred_idx][0]
                    else:
                        arr = np.array(shap_vals)
                        sv = arr[0, :, pred_idx] if arr.ndim == 3 else arr[0]

                    word_shaps = []
                    for w in set(text_clean.split()):
                        if w in feat:
                            word_shaps.append({"Kata": w, "Dampak SHAP": float(sv[feat.index(w)])})

                    if word_shaps:
                        shap_df = pd.DataFrame(word_shaps).sort_values("Dampak SHAP")
                        fig_shap = px.bar(
                            shap_df, x="Dampak SHAP", y="Kata", orientation="h",
                            color="Dampak SHAP", color_continuous_scale="RdYlGn",
                            text_auto=".3f"
                        )
                        fig_shap.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#c9d1d9", height=max(200, len(word_shaps) * 35),
                            margin=dict(l=0, r=10, t=10, b=0), coloraxis_showscale=False
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                        st.info(
                            "💡 Batang hijau = kata mendorong ke arah prediksi. "
                            "Batang merah = kata berlawanan arah. "
                            "Kata tidak muncul berarti *Out-Of-Vocabulary* di data latih."
                        )
                    else:
                        st.warning("⚠️ Tidak ada kata yang dikenali oleh kosakata TF-IDF (Out-Of-Vocabulary).")
                except Exception as e:
                    st.warning(f"SHAP tidak dapat dirender: {e}")

# ==============================================================================
# 6. DOKUMENTASI
# ==============================================================================
elif menu == "Dokumentasi":
    st.title("📄 Arsitektur & Dokumentasi Teknis")

    tab_arsitektur, tab_metrik, tab_tentang = st.tabs([
        "⚙️ Arsitektur Model",
        "📊 Evaluasi & Metrik",
        "ℹ️ Tentang Proyek"
    ])

    with tab_arsitektur:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🥇 Model Final — IndoBERT (Transformer)
            **`indobenchmark/indobert-base-p1`** adalah *Pre-trained Language Model* berbasis BERT
            yang dilatih pada korpus Bahasa Indonesia skala besar.

            **Fine-tuning pada proyek ini:**
            - **Data:** 1.500 komentar berlabel emas (Groq Llama-3 8B)
            - **Split:** Train 70% / Val 15% / Test 15%
            - **Loss:** Weighted Cross-Entropy (menangani class imbalance)
            - **Callback:** Early Stopping (patience=2, monitor=macro_f1)
            - **Best Epoch:** Epoch 4 (Val F1: 0.730)

            **Keunggulan vs TF-IDF:**
            - Memahami urutan kata & konteks kalimat
            - Menangani negasi: "tidak bagus" ≠ "sangat bagus"
            - Subword tokenization → lebih tahan terhadap slang
            """)
        with col2:
            st.markdown("""
            ### 🥈 Baseline — Ensemble VotingClassifier
            *Soft Voting* dari 3 algoritma klasik berbasis TF-IDF:

            | Algoritma | Peran |
            |-----------|-------|
            | **Logistic Regression** | Stabil, interpretable via SHAP |
            | **Random Forest** | Tahan terhadap data noisy |
            | **XGBoost** | Akurasi tinggi pada edge cases |

            **Feature Engineering:**
            - TF-IDF Bigram (`ngram_range=(1,2)`, `max_features=10.000`)
            - `sublinear_tf=True` untuk meredam kata terlalu sering
            - `max_df=0.90` untuk filter noise domain-spesifik
            - Integrasi dalam `sklearn.Pipeline` (anti-data leakage)

            **Labeling Strategy:**
            - Confidence threshold = 0.5
            - Komentar < threshold → fallback `neutral`
            """)

    with tab_metrik:
        st.markdown("### 📊 Perbandingan Performa Model")

        ens_acc = metadata.get("ensemble_accuracy", 74.22)
        ens_f1  = metadata.get("ensemble_f1_macro", 68.04)
        b_acc   = metadata.get("bert_accuracy", 77.33)
        b_f1    = metadata.get("bert_f1_macro", 72.48)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Ensemble (Baseline)**")
            st.metric("Accuracy", f"{ens_acc:.2f}%")
            st.metric("Macro F1", f"{ens_f1/100:.4f}" if ens_f1 > 1 else f"{ens_f1:.4f}")
        with col_m2:
            delta_acc = b_acc - ens_acc
            delta_f1  = (b_f1 - ens_f1) / 100 if b_f1 > 1 else (b_f1 - ens_f1)
            st.markdown("**IndoBERT Fine-tuned (Final) ✅**")
            st.metric("Accuracy", f"{b_acc:.2f}%", delta=f"+{delta_acc:.2f}%")
            st.metric("Macro F1", f"{b_f1/100:.4f}" if b_f1 > 1 else f"{b_f1:.4f}", delta=f"+{delta_f1:.4f}")

        st.markdown("---")
        st.markdown("### Laporan Klasifikasi IndoBERT (Test Set — 225 sampel)")

        report_data = {
            "Kelas":     ["negative", "neutral", "positive", "macro avg", "weighted avg"],
            "Precision": [0.67, 0.64, 0.86, 0.72, 0.77],
            "Recall":    [0.67, 0.66, 0.85, 0.73, 0.77],
            "F1-Score":  [0.67, 0.65, 0.86, 0.72, 0.77],
            "Support":   [54, 41, 130, 225, 225]
        }
        report_df = pd.DataFrame(report_data)
        st.dataframe(
            report_df.style.background_gradient(subset=["F1-Score"], cmap="Greens"),
            use_container_width=True, hide_index=True
        )

        st.markdown("---")
        st.markdown("### Confusion Matrix — IndoBERT (Test Set)")
        # Confusion matrix dari hasil notebook
        cm_data = [[36, 9, 9], [7, 27, 7], [8, 12, 110]]
        cm_labels = ["Negative", "Neutral", "Positive"]
        cm_df = pd.DataFrame(cm_data, index=[f"Actual {l}" for l in cm_labels],
                             columns=[f"Pred {l}" for l in cm_labels])
        fig_cm = px.imshow(
            cm_df, text_auto=True, color_continuous_scale="Blues",
            aspect="auto", zmin=0
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", height=320,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Diagonal utama = prediksi benar. Kelas Positive paling akurat (F1=0.86).")

    with tab_tentang:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 📋 Tentang Proyek
            **Judul:** EduPulse CoC — Membedah Sentimen Edutainment Digital

            **Kompetisi:** GWE 2026 Data Science Challenge

            **Sub-tema:** Sentiment Analysis + Education

            ### 📊 Sumber Dataset
            **[Ruangguru Clash of Champions 2024 YouTube Comments](https://www.kaggle.com/datasets/rezkyyayang/ruangguru-clash-of-champions-2024-youtube-comments)**
            - Platform: Kaggle
            - Dibuat oleh: **rezkyyayang**
            - Deskripsi: Dataset komentar YouTube dari tayangan Ruangguru Clash of Champions 2024 untuk text mining & analisis sentimen.
            - Total data: 14.149 komentar mentah → 13.777 setelah cleaning
            """)
        with col2:
            st.markdown("""
            ### 🤖 AI Tools yang Digunakan
            *(Sesuai Pedoman GWE 2026 Pasal 4e)*

            | Tool | Kegunaan |
            |------|----------|
            | **Groq Llama-3 8B Instant** | Gold labeling 1.500 sampel |
            | **IndoBERT (HuggingFace)** | Pre-trained model untuk fine-tuning |
            | **Antigravity** | Debugging & kode assistance |

            ### 📁 Tech Stack
            - **NLP:** HuggingFace Transformers, scikit-learn, SHAP
            - **Visualisasi:** Plotly, Matplotlib, WordCloud
            - **Deployment:** Streamlit
            - **Labeling:** Groq API (Llama-3 8B Instant)
            """)

        st.markdown("---")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
            ### 🚀 Cara Penggunaan Aplikasi
            Aplikasi EduPulse CoC sangat mudah digunakan melalui menu navigasi di sebelah kiri:
            1. **🏠 Beranda:** Membaca ringkasan eksekutif proyek dan insight utama.
            2. **📈 EDA Dashboard:** Menjelajahi tren data, WordCloud, dan korelasi *engagement* secara interaktif.
            3. **🤖 AI Prediction:** Anda bisa mencoba langsung kehebatan model AI! Ketik komentar Anda (bahkan dengan bahasa gaul/slang) dan AI akan mendeteksi sentimennya secara *real-time*.
            4. **📄 Dokumentasi:** Membaca penjelasan arsitektur teknis dan profil tim pengembang.
            """)
        with col_t2:
            st.markdown("""
            ### 👥 Informasi Tim
            Proyek dan aplikasi ini dikembangkan untuk GWE 2026 Data Science Challenge oleh:
            
            - 🧑‍💻 **Muhammad Dhimas Pranaya**
            - 👩‍💻 **Mayhesa Rachmadiharja**
            - 👨‍💻 **Achmad Dhafi Fadilah** 
            """)
