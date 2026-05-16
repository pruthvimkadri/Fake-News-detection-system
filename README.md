# Fake News Detection System

A Machine Learning + NLP based web application that detects whether a news article is Fake or Real.

## Features
- Fake/Real news prediction
- Confidence score
- Flask web interface
- TF-IDF vectorization
- Logistic Regression model
- Interactive UI

## Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas
- HTML/CSS

## Dataset
Kaggle Fake News Dataset

## Model
- TF-IDF Vectorizer
- Logistic Regression

## Accuracy
Achieved around 98% accuracy on test dataset.

## Project Structure

fake-news-detection/
│── app.py
│── train.py
│── model.pkl
│── vectorizer.pkl
│── templates/
│    └── index.html
│── static/
│    └── style.css
│── README.md

## How to Run

1. Install dependencies

pip install -r requirements.txt

2. Run the Flask app

python app.py

3. Open browser

http://127.0.0.1:5000/

## Screenshots
<img width="974" height="340" alt="image" src="https://github.com/user-attachments/assets/9ac5f565-e8b8-4822-aa1c-51c185b42c51" />

## Future Improvements
- Better real-world generalization
- Explainable AI features
- Deployment on cloud
