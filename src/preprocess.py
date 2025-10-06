import re
import nltk
from nltk.corpus import stopwords

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-záéíóúñü\s]", " ", text)
    tokens = text.split()
    try:
        sw = set(stopwords.words("spanish"))
    except LookupError:
        nltk.download("stopwords")
        sw = set(stopwords.words("spanish"))
    tokens = [t for t in tokens if t not in sw and len(t) > 2]
    return " ".join(tokens)
