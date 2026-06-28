# GWE 2026 Data Science Challenge: EduPulse CoC

## 🚀 Gambaran Proyek
**EduPulse CoC** adalah sistem analisis sentimen cerdas yang membedah fenomena *edutainment* (edukasi + hiburan) di Indonesia, berfokus pada program kompetisi **Clash of Champions (CoC)** dari Ruangguru.

Proyek ini mentransformasi **13.777 komentar** organik dari penonton YouTube menjadi sebuah **Barometer Sentimen Publik** — mengubah data mentah menjadi insight terstruktur yang memetakan suasana hati, ekspektasi, dan motivasi belajar generasi muda Indonesia.

## 🎯 Tujuan Proyek (Tiga Pilar Dampak)
1. **Validasi Efektivitas Edutainment:** Membuktikan secara empiris apakah tayangan edukasi mampu meningkatkan semangat belajar atau justru memicu tekanan mental (*insecure*) pada audiensnya.
2. **Menyediakan Wawasan Berbasis Data (Actionable Insights):**
   - *Kreator & Platform:* Memahami elemen konten yang paling disukai penonton.
   - *Pembuat Kebijakan & Institusi Pendidikan:* Menjadikan media digital sebagai alat pemicu diskusi akademik.
3. **Membangun Ekosistem Diskusi yang Sehat:** Melalui *dashboard* interaktif berbasis Streamlit.

## 🏆 Hasil Model Final

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Ensemble (LR + RF + XGBoost) — *Baseline* | 74.22% | 0.68 |
| **IndoBERT Fine-tuned** — ***Model Final*** | **77.33%** | **0.72** |

IndoBERT dipilih sebagai model final karena peningkatan Macro F1 **+28.6%** vs baseline Ensemble.

## 🤖 Penggunaan AI Tools
*(Sesuai Pedoman GWE 2026 Pasal 4e — semua penggunaan dicantumkan)*
- **Groq Llama-3 8B Instant API:** Digunakan untuk *gold labeling* 1.500 sampel komentar pada Fase 2. Dipilih karena kemampuannya memahami bahasa gaul, sarkasme, dan slang Bahasa Indonesia.
- **IndoBERT (`mayhesar/indobert-sentiment-gwe`):** Model transformer fine-tuned yang disimpan dan di-host di [Hugging Face Hub](https://huggingface.co/mayhesar/indobert-sentiment-gwe).
- **Antigravity (Google DeepMind):** Digunakan untuk membantu debugging notebook, penyusunan kerangka kode, dan *troubleshooting*.



## 📊 Sumber Dataset
- **Nama Dataset:** [Ruangguru Clash of Champions 2024 YouTube Comments](https://www.kaggle.com/datasets/rezkyyayang/ruangguru-clash-of-champions-2024-youtube-comments)
- **Sumber:** Kaggle — dibuat oleh **rezkyyayang**
- **Deskripsi:** Dataset berisi komentar YouTube dari tayangan Ruangguru Clash of Champions 2024, cocok untuk *text mining* dan analisis sentimen.
- **File Lokal:** `data/raw/01_ruangguru_clash_of_champions.csv`

## 🧠 Metodologi Machine Learning
1. **Data Cleaning & Preprocessing:** Normalisasi slang, hapus emoji/URL, pertahankan kata negasi.
2. **Gold Labeling via LLM (Groq Llama-3 8B):** 1.500 sampel dilabeli secara otomatis dengan LLM.
3. **Active Learning — Ensemble Student (Fase 3):** VotingClassifier (LR + RF + XGBoost) + TF-IDF bigram, melakukan inferensi ke 13.777 komentar.
4. **Fine-Tuning IndoBERT (Fase 3.5):** Transformer BERT berbasis korpus Bahasa Indonesia; model final.
5. **EDA:** 5 visualisasi interaktif (Plotly) + WordCloud + validasi pilar edutainment.
6. **SHAP Interpretability:** LinearExplainer (LR) + TreeExplainer (XGBoost) untuk actionable insights.

## 📁 Struktur Repository
```
EduPulse CoC/
├── README.md
├── requirements.txt
├── notebooks/
│   └── GWE.ipynb          ← Notebook analisis end-to-end
├── src/
│   └── app.py             ← Streamlit dashboard
├── data/
│   ├── raw/               ← Dataset mentah dari Kaggle
│   └── processed/         ← Dataset hasil labeling & inferensi
├── models/
│   ├── indobert_sentiment_final/   ← Model final (IndoBERT)
│   └── ensemble_baseline.pkl       ← Model baseline (Ensemble)
└── presentation/
    └── slides.pdf         ← Slide presentasi
```

## 💻 Cara Menjalankan Secara Lokal
1. Clone repository ini.
2. Install dependencies: `pip install -r requirements.txt`
3. Jalankan aplikasi Streamlit: `streamlit run src/app.py`

## 🔗 Link
- **Dataset Kaggle:** [Ruangguru Clash of Champions 2024 YouTube Comments](https://www.kaggle.com/datasets/rezkyyayang/ruangguru-clash-of-champions-2024-youtube-comments)
- **Demo Streamlit:** *(coming soon — akan diupdate setelah deployment)*
