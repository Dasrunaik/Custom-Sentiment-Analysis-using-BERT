import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load your fine-tuned model (or use a public one for testing)
MODEL_DIR = "saved_bert_model"

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    return tokenizer, model

tokenizer, model = load_model()
model.eval()

st.title("🧠 Sentiment Classifier (BERT)")
text = st.text_input("Enter a sentence:")

if st.button("Classify"):
    if not text.strip():
        st.warning("Please enter something.")
    else:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        label = "Positive 😊" if prediction == 1 else "Negative 😠"
        st.success(f"Prediction: {label}")




