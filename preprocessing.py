# preprocessing.py
import re
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r"\d+", "", text)  
    text = text.translate(str.maketrans("", "", string.punctuation))  
    tokens = text.split()  
    tokens = [word for word in tokens if word not in stop_words]  
    return " ".join(tokens)

