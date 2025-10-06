import argparse, pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report
from pathlib import Path

def main(args):
    df = pd.read_csv(args.data).dropna(subset=["texto_clinico","gravedad"]).copy()
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["leve","moderado","severo"]

    preds = []
    for txt in df["texto_clinico"].astype(str).tolist():
        out = classifier(txt, candidate_labels=labels, multi_label=False)
        preds.append(out["labels"][0])

    rep = classification_report(df["gravedad"], preds, digits=3)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "clasificacion_zeroshot.txt").write_text(rep, encoding="utf-8")
    print(rep)

    # Guardar predicciones
    pd.DataFrame({"y_pred": preds}).to_csv(out / "preds_zeroshot.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    main(args)
