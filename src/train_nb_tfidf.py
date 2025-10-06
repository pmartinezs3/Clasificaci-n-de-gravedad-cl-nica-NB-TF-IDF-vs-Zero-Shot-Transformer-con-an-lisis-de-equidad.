import argparse, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess import preprocess_text

def main(args):
    df = pd.read_csv(args.data)
    text_col, y_col = "texto_clinico", "gravedad"
    df = df.dropna(subset=[text_col, y_col]).copy()
    df["texto_proc"] = df[text_col].astype(str).map(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["texto_proc"], df[y_col], test_size=0.25, stratify=df[y_col], random_state=42
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1,2))),
        ("clf", MultinomialNB())
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # Guardar reporte
    rep = classification_report(y_test, y_pred, digits=3)
    (out / "clasificacion_nb.txt").write_text(rep, encoding="utf-8")
    print(rep)

    # Matriz de confusión
    import seaborn as sns
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
    fig = plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=sorted(y_test.unique()),
                yticklabels=sorted(y_test.unique()))
    plt.title("Confusion Matrix — NB (TF-IDF)")
    plt.xlabel("Predicho"); plt.ylabel("Real")
    fig.tight_layout()
    fig.savefig(out / "matriz_confusion_nb.png", dpi=180)
    plt.close(fig)

    # Guardar predicciones para fairness
    pd.DataFrame({"y_pred": y_pred}, index=y_test.index).to_csv(out / "preds_nb.csv", index=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    main(args)
