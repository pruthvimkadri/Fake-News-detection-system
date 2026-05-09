import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------
# LOAD DATASET
# ---------------------------
df = pd.read_csv("fake_news.csv")
df = df.dropna()

# Make sure label is integer
df["label"] = df["label"].astype(int)

print("Class distribution:")
print(df["label"].value_counts())
print("\n")

X = df["text"]
y = df["label"]

# ---------------------------
# SIMPLE TEXT CLEANING
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

X = X.apply(clean_text)

# ---------------------------
# TRAIN / TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------
# TF-IDF VECTORIZATION
# ---------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.9,
    min_df=2,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# MODEL
# ---------------------------
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="liblinear"
)

model.fit(X_train_vec, y_train)

# ---------------------------
# EVALUATION
# ---------------------------
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# SAVE MODEL + VECTORIZER
# ---------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully")