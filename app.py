# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

# Set the title and description of the web app
st.title("Text Classification with DistilBERT")
st.write("This app allows you to classify text as sincere or insincere.")

# Sidebar for user input
st.sidebar.title("User Input")
text_input = st.sidebar.text_area("Enter text for classification:")

# Add a placeholder to push content to the top


# Add the person's name at the bottom
st.write("Made by: Nainil Maladkar and Simran Nagpurkar")



# Load the pretrained DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Function to classify text using DistilBERT
def classify_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = np.argmax(logits.detach().numpy(), axis=1).item()
    return predicted_class

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Sincere', 'Insincere'], output_dict=True)
    return accuracy, report

# Function to load and preprocess the Quora dataset
def load_and_preprocess_data():
    # Load the Quora dataset (you need to have train.csv)
    data = pd.read_csv('/Users/simrannagpurkar/Downloads/NLP_team_project/train.csv')
    # Split data into features and labels
    X = data['question_text']
    y = data['target']
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Return the data
    return X_train, X_test, y_train, y_test

# Check if the user input is provided
if text_input:
    predicted_class = classify_text(text_input)
    result = "Insincere" if predicted_class == 1 else "Sincere"
    st.write(f"Predicted Class: {result}")

# Load and evaluate the model when the user provides a CSV file
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for evaluation:", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    X_test, y_test = data['question_text'], data['target']
    accuracy, report = evaluate_model(model, X_test, y_test)
    
    st.write(f"Model Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.dataframe(pd.DataFrame(report).transpose())

# App initialization
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    st.write("App is ready. Use the sidebar for user input or to evaluate a CSV file.")
