# model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, f1_score

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    print("📊 Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label="spam"))
    print("F1 Score:", f1_score(y_test, y_pred, pos_label="spam"))
