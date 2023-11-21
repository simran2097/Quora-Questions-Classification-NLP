## Quora_Question_Classification

## Objective:
The Quora Sincere/Insincere Questions Classification is a Python NLP project that aims to 
segregate the given dataset of questions posted on Quora into relevant Sincere or Insincere buckets 
for optimized text classification.

### Sincere Questions: 
These are questions that are genuinely seeking helpful answers, contributing to 
the knowledge-sharing ethos of Quora. Sincere questions are founded on a desire for information, 
insights, and solutions.

### Insincere Questions: 
Insincere questions are problematic as they are typically not genuine inquiries 
for information. They may be based on false premises, intended to make statements, or can be 
offensive, divisive, or inappropriate in nature. These questions often violate Quora's "Be Nice, Be 
Respectful" policy

## Project Scope
1. Utilized Quora's Insincere Questions Classification Kaggle Dataset.
2. Employed Natural Language Processing (NLP) techniques for text analysis.
3. Implemented various models including Logistic Regression, Naive Bayes, Convolutional Neural Network (CNN), and BERT. 
4. Evaluated models based on accuracy, precision, recall, and F1 score.
5. Explored text characteristics crucial for classification, such as word count, character count, and stopword frequency.
6. Developed a Streamlit application utilizing DistilBERT for real-time question classification.

## Exploratory Data Analysis (EDA)
#### Analysis Overview
1. Word Cloud Visualization: Identified most frequent words in sincere and insincere questions. 
2. Bigram Frequency Analysis: Explored pairs of words occurring frequently together in both sincere and insincere questions.
3. Text Characteristics Examination: Analyzed various text attributes crucial for classification, including word count, character count, unique word count, etc.

## Feature Extraction
1. Word Count: Calculating the total number of words in each question.
2. Unique Word Count: Identifying the count of distinct words used in a question.
3. Character Count: Determining the total number of characters present in the question text.
4. Stopwords Count: Counting the occurrences of commonly used stopwords (e.g., "the," "is," "and") in questions.
5. Punctuation Count: Tracking the usage of punctuation marks within the questions.
6. Title and Uppercase Words Count: Detecting the count of words in uppercase or within the title of the question.
7. Average Word Length: Calculating the average length of words used in a question.
