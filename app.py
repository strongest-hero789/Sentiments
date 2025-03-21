import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load Data
def load_data():
    data = pd.read_csv("financial_sentiment_data.csv")
    return data

data = load_data()

# Preprocessing
X = data['Sentence']
y = data['Sentiment']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

def predict_sentiment(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    sentiment = label_encoder.inverse_transform(prediction)[0]
    
    if sentiment == "positive":
        return "😊 Positive", "#4CAF50"
    elif sentiment == "negative":
        return "😠 Negative", "#F44336"
    else:
        return "😐 Neutral", "#FFC107"

# Streamlit App
st.title("📊 Financial Sentiment Analysis")
st.write("Enter a financial news headline to predict its sentiment.")

user_input = st.text_area("📝 Enter text:")

if st.button("🔍 Predict"):
    if user_input:
        prediction, color = predict_sentiment(user_input)
        st.markdown(f"<h3 style='color: {color};'>{prediction}</h3>", unsafe_allow_html=True)
    else:
        st.write("❗ Please enter some text.")
