Description:
This project performs sentiment analysis on textual data such as tweets, product reviews, or customer feedback. The goal is to automatically classify text into three categories: positive, negative, or neutral, enabling insights into customer opinions, social media trends, or product performance.
The application is fully dynamic, allowing users to:
Load any CSV dataset containing text and label columns.Train a machine learning model (Naive Bayes) on the dataset.Analyze new text inputs in real-time using both TextBlob (rule-based sentiment) and the trained ML model.This project combines Natural Language Processing (NLP) with machine learning to create a practical tool for social media analytics, e-commerce feedback monitoring, and customer satisfaction evaluation.

Key Features:
-Dynamic Dataset Loading: Users can provide any CSV file to train the model.
-Text Preprocessing: Handles punctuation, stopwords removal, and case normalization.
-Dual Sentiment Analysis: Uses TextBlob for quick polarity detection and a trained ML model for more robust predictions.
-Machine Learning Model: Uses TF-IDF vectorization and Naive Bayes classifier.
-Real-Time Predictions: Users can input text interactively to get immediate sentiment results.
-Extensible: Can be upgraded with larger datasets or advanced models like BERT for higher accuracy.

Technologies Used:
-Python
-NLTK (Natural Language Toolkit) 
-TextBlob 
-scikit-learn
-Pandas

