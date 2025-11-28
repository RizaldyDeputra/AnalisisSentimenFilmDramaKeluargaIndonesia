import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# import preprocessing MU
from preprocess import pra_pemrosesan

# Repo model Hugging Face
MODEL_REPO = "RizaldyDeputra/Indobert-TA-5/IndoBERT_EPOCH5"  
ID2LABEL = {0: "negatif", 1: "positif"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def prediksi_sentimen(teks_input: str):
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
    return ID2LABEL[idx], probs

# ===========================
# STREAMLIT UI
# ===========================

st.title("Analisis Sentimen Film – IndoBERT Fine-tuned")

st.write("Masukkan kalimat ulasan film, lalu klik **Prediksi**.")

text = st.text_area("Masukkan teks ulasan:", height=150)

if st.button("Prediksi"):
    if text.strip() == "":
        st.warning("Tolong masukkan teks terlebih dahulu.")
    else:
        label, probs = prediksi_sentimen(text)

        st.subheader("Hasil Prediksi")
        st.write(f"**Label:** `{label}`")
        st.write(f"Negatif: `{probs[0]:.4f}`")
        st.write(f"Positif: `{probs[1]:.4f}`")
