import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocess import pra_pemrosesan

# ===========================
# KONFIGURASI MODEL HF
# ===========================

# GANTI dengan repo dan folder kamu sendiri
HF_REPO_ID = "RizaldyDeputra/Indobert-TA-5"   # contoh: "arancranel/analisis-sentimen-film"
HF_SUBFOLDER = "IndoBERT_EPOCH5"       # contoh: nama folder di dalam repo HF

ID2LABEL = {0: "negatif", 1: "positif"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# LOAD MODEL + TOKENIZER
# ===========================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        HF_REPO_ID,
        subfolder=HF_SUBFOLDER,
        use_fast=False,         # aman untuk IndoBERT
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_REPO_ID,
        subfolder=HF_SUBFOLDER,
    )
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ===========================
# FUNGSI PREDIKSI
# ===========================

def prediksi_sentimen(teks_input: str):
    # konsisten dengan preprocessing saat training
    teks_bersih = pra_pemrosesan(teks_input)

    encoding = tokenizer(
        teks_bersih,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        out = model(**encoding)
        probs = F.softmax(out.logits, dim=-1).cpu().numpy()[0]

    idx = int(probs.argmax())
    label_pred = ID2LABEL[idx]

    return label_pred, probs

# ===========================
# UI STREAMLIT
# ===========================

st.title("Analisis Sentimen Film Drama Keluarga – IndoBERT")

st.write(
    "Masukkan kalimat ulasan film (misalnya tentang *1 Kakak 7 Ponakan*, "
    "*Perayaan Mati Rasa*, dll), lalu klik **Prediksi**."
)

teks_input = st.text_area("Tulis ulasan di sini:", height=150)

if st.button("Prediksi"):
    if not teks_input.strip():
        st.warning("Tolong isi dulu teks ulasannya 😊")
    else:
        label, probs = prediksi_sentimen(teks_input)

        st.subheader("Hasil Prediksi")
        st.write(f"**Label sentimen:** `{label}`")
        st.write("**Probabilitas:**")
        st.write(f"- Negatif: `{probs[0]:.4f}`")
        st.write(f"- Positif: `{probs[1]:.4f}`")
