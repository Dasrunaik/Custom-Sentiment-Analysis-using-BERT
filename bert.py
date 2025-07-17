import streamlit as st
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

@st.cache_resource
def get_model():
    tokenizer=DistilBertTokenizer.from_pretrained('saved_distilbert_model')
    model=DistilBertForSequenceClassification.from_pretrained('saved_distilbert_model')
    return tokenizer,model


tokenizer,model=get_model()

st.title("Text Classification with Hugging Face BERT")
st.subheader("Enter text below to classify (e.g., spam detection, sentiment, etc.)")

# User input
user_input = st.text_area("Enter your text here:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text before classifying.")
    else:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

        # Customize labels as per your model (e.g., 0 = ham, 1 = spam)
        label_map = {
            0: "Ham",
            1: "Spam"
            
        }

        st.success(f" Predicted Label: **{label_map.get(prediction, 'Unknown')}**")


