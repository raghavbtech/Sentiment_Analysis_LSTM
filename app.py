import streamlit as st
import torch
import re
import json
import numpy as np

from model import LSTMModel


def pad_sequences(sequences, maxlen):
    out = np.zeros((len(sequences), maxlen), dtype="int32")
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            seq = seq[len(seq) - maxlen:]
        out[i, maxlen - len(seq):] = seq
    return out


with open("tokenizer.json") as f:
    raw = json.load(f)
    wi = raw["config"]["word_index"]
    word_index = json.loads(wi) if isinstance(wi, str) else wi


def texts_to_sequences(texts):
    return [[word_index[w] for w in text.split() if w in word_index] for text in texts]

vocab_size = 5001
model = LSTMModel(vocab_size)
model.load_state_dict(torch.load("LSTMmodel.pth", map_location=torch.device("cpu")))
model.eval()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text


def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)
    tensor = torch.tensor(padded, dtype=torch.long)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.sigmoid(out).item()
    return prob


st.set_page_config(page_title="Reel Feelings", page_icon="🎞️", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0e0e0e;
        color: #ede8de;
    }

    .header {
        text-align: center;
        padding: 3rem 0 2rem;
        border-bottom: 1px solid #1f1f1f;
        margin-bottom: 2.4rem;
    }

    .header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: #e2bc6e;
        margin: 0;
        letter-spacing: -0.5px;
        line-height: 1;
    }

    .header p {
        color: #555;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-top: 0.5rem;
    }

    .stTextArea textarea {
        background-color: #181818 !important;
        border: 1px solid #2c2c2c !important;
        border-radius: 7px !important;
        color: #ede8de !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.97rem !important;
        padding: 14px 16px !important;
        line-height: 1.6 !important;
        caret-color: #e2bc6e;
    }

    .stTextArea textarea::placeholder {
        color: #3a3a3a !important;
    }

    .stTextArea textarea:focus {
        border-color: #e2bc6e !important;
        box-shadow: 0 0 0 2px rgba(226, 188, 110, 0.1) !important;
        outline: none !important;
    }

    .stButton > button {
        background-color: #e2bc6e !important;
        color: #0e0e0e !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.65rem 2.2rem !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.2px !important;
        width: 100%;
        transition: background-color 0.15s ease, transform 0.1s ease;
        font-family: 'Inter', sans-serif !important;
    }

    .stButton > button:hover {
        background-color: #c9a555 !important;
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    .result-card {
        padding: 2rem;
        border-radius: 8px;
        margin-top: 1.6rem;
        text-align: center;
    }

    .result-pos {
        background-color: #131f13;
        border: 1px solid #204020;
    }

    .result-neg {
        background-color: #1f1313;
        border: 1px solid #402020;
    }

    .result-emoji {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }

    .result-label {
        font-family: 'Playfair Display', serif;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
    }

    .result-pos .result-label { color: #6abf6a; }
    .result-neg .result-label { color: #bf6a6a; }

    .result-meta {
        font-size: 0.78rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        margin-top: 0.3rem;
    }

    .conf-bar {
        margin: 1.1rem auto 0;
        max-width: 180px;
        height: 3px;
        background: #222;
        border-radius: 2px;
        overflow: hidden;
    }

    .conf-fill-pos { height: 100%; background: #6abf6a; border-radius: 2px; }
    .conf-fill-neg { height: 100%; background: #bf6a6a; border-radius: 2px; }

    .examples-section {
        margin-top: 2.8rem;
        padding-top: 1.8rem;
        border-top: 1px solid #191919;
    }

    .examples-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        color: #444;
        margin-bottom: 0.9rem;
    }

    .pill {
        display: inline-block;
        background: #181818;
        border: 1px solid #252525;
        color: #666;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.82rem;
        margin: 0.2rem 0.15rem;
        font-family: 'Inter', sans-serif;
    }

    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #191919;
        color: #333;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0 !important; max-width: 660px; }
    div[data-testid="stVerticalBlock"] { gap: 0.6rem; }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="header">
    <h1>Reel Feelings</h1>
    <p>Movie Sentiment Analyzer</p>
</div>
""", unsafe_allow_html=True)

user_input = st.text_area(
    "Your review",
    placeholder="What did you think of the film...",
    height=130,
    label_visibility="collapsed"
)

_, mid, _ = st.columns([1, 2, 1])
with mid:
    run = st.button("Analyze review")

if run:
    if not user_input.strip():
        st.warning("Write something first.")
    else:
        prob = predict_sentiment(user_input)
        pct = int(prob * 100)

        if prob > 0.5:
            st.markdown(f"""
            <div class="result-card result-pos">
                <div class="result-emoji">✦</div>
                <div class="result-label">Positive</div>
                <div class="result-meta">{pct}% confidence</div>
                <div class="conf-bar">
                    <div class="conf-fill-pos" style="width:{pct}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            neg_pct = 100 - pct
            st.markdown(f"""
            <div class="result-card result-neg">
                <div class="result-emoji">✦</div>
                <div class="result-label">Negative</div>
                <div class="result-meta">{neg_pct}% confidence</div>
                <div class="conf-bar">
                    <div class="conf-fill-neg" style="width:{neg_pct}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div class="examples-section">
    <div class="examples-label">Try one of these</div>
    <span class="pill">This movie was breathtaking</span>
    <span class="pill">Worst film I've seen this year</span>
    <span class="pill">A genuine masterpiece</span>
    <span class="pill">Complete waste of time</span>
    <span class="pill">Left me in tears, in the best way</span>
</div>

<div class="footer">built for movie lovers &nbsp;·&nbsp; LSTM · PyTorch</div>
""", unsafe_allow_html=True)
