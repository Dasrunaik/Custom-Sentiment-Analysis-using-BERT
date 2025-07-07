# Custom-Sentiment-Analysis-using-BERT
Pretrained and Fine Tuning using the Transformers 

# 📩 SMS Spam Detection with BERT


A complete machine learning pipeline that classifies SMS messages as **Spam** or **Ham** using a fine-tuned **BERT** model. 

The project also includes **EDA**, **training**, and a **Streamlit web app** for real-time prediction.

---

## 🚀 Features

- ✅ Exploratory Data Analysis (EDA)
- ✅ Label encoding (`spam` → 1, `ham` → 0)
- ✅ BERT tokenizer and classifier (from Hugging Face Transformers)
- ✅ Model training using `Trainer`
- ✅ Model evaluation
- ✅ Streamlit app for real-time SMS classification
- ✅ Model saving and loading for deployment

---

## 🗂 Dataset

Uses the [SMS Spam Collection Dataset](https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/), a set of 5,572 labeled messages.


Load and clean the dataset

## Train  model:

Train the BERT model using the Hugging Face

Save the model and tokenizer to saved_bert_model/

## PROJECT STRUCTURE:

sms-spam-detector/

├── SMSSpamCollection.txt         # Dataset

├── train_model.py                # Training script

├── app.py                        # Streamlit app

└── saved_bert_model/             # Saved model and tokenizer



## 🧪 Example Prediction

Input:
"Congratulations! You've won a free iPhone!"

Prediction:
🛑 Spam


## 📚 Technologies Used 

Transformers

BERT (Hugging Face)

PyTorch

Streamlit

Scikit-learn

Seaborn
