import streamlit as st
import pymupdf  # PyMuPDF
from transformers import pipeline
from gtts import gTTS
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
import tempfile
import nltk
import os
import sys

st.write(f"Python version: {sys.version}")

nltk.download("punkt_tab")


# === FUNCTIONS ===

def extract_text(pdf_file):
    doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
    return " ".join(page.get_text() for page in doc)


@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")


def summarize_text(text, chunk_size=1000):
    summarizer = load_summarizer()
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = [summarizer(chunk)[0]["summary_text"] for chunk in chunks]
    return " ".join(summaries)


def split_summary(summary, max_chars=1000):
    sentences = sent_tokenize(summary)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def text_to_speech(chunks, output_folder="tts_outputs"):
    with tempfile.TemporaryDirectory() as tempdir:
        combined = AudioSegment.empty()
        for i, chunk in enumerate(chunks):
            tts = gTTS(text=chunk, lang='en')
            path = os.path.join(tempdir, f"chunk_{i}.mp3")
            tts.save(path)

            audio = AudioSegment.from_mp3(path)
            combined += audio + AudioSegment.silent(duration=500)

        final_path = os.path.join(tempdir, "final_podcast.mp3")
        combined.export(final_path, format="mp3")

        with open(final_path, "rb") as f:
            final_audio = f.read()

    return final_audio


# === STREAMLIT UI ===

st.set_page_config(page_title="PDF to Podcast", layout="centered")
st.title("PDF to Audio Summary")
st.write("Upload a PDF and get a summarized audio version.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        raw_text = extract_text(uploaded_file)

    with st.spinner("Summarizing..."):
        summary = summarize_text(raw_text)

    with st.spinner("Converting to audio..."):
        segments = split_summary(summary)
        audio_file = text_to_speech(segments)

    with open(audio_file, "rb") as f:
        st.success("Audio ready!")
        st.audio(f.read(), format="audio/mp3")
        st.download_button("Download MP3", f, file_name="summary_audio.mp3", mime="audio/mpeg")
