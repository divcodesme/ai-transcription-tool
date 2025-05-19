import whisper
import streamlit as st
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# core transcription logic using Open AI Whisper model

model = whisper.load_model("base")

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]


# simple streamlit interface
st.title('Smart Transcriber and Summarizer')
uploaded_file = st.file_uploader("Upload an audio file")

if uploaded_file:
    with open("temp_audio_file.wav", 'wb') as f:
        f.write(uploaded_file.read())

    transcript = transcribe_audio("temp_audio_file.wav")
    st.subheader("Transcript")
    st.text_area("Full Transcript...", transcript, height = 300)


# Langchain Summarizer

llm = Ollama(model='llama2')

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize this transcript:\n\n{text}\n\nSummary:"
)

summarizer = LLMChain(llm=llm, prompt=summary_prompt)

def sumamrize_transcript(text):
    return summarizer.run(text)


# Final Interface

if st.button("Summarize"):
    with st.spinner("Summarizing..."):
        summary = sumamrize_transcript(transcript)
    st.subheader("Summary")
    st.write(summary)