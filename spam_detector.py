# spam_detector.py
from preprocessing import preprocess_text
from model import train_model, evaluate_model
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset load karo (SMS Spam Collection)
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]

# Preprocess text
df["cleaned_message"] = df["message"].apply(preprocess_text)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_message"], df["label"], test_size=0.2, random_state=42
)

# Train model
model, vectorizer = train_model(X_train, y_train)

# Evaluate
evaluate_model(model, vectorizer, X_test, y_test)

