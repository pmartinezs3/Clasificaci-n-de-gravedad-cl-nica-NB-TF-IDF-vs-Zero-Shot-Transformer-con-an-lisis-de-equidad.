import argparse, pandas as pd, numpy as np
from sklearn.metrics import f1_score
from pathlib import Path

def f1_grouped(df, group_col, y_true_col="gravedad", y_pred_col="y_pred"):
    out = []
    for g, d in df.groupby(group_col):
        if d[y_true_col].nunique() < 2:
            f1w = 0.0
        else:
            f1w = f1_score(d[y_true_col], d[y_pred_col], average="weighted")
        out.append({"grupo": g, "f1_weighted": f1w, "n": len(d)})
    return pd.DataFrame(out)

def main(args):
    data = pd.read_csv(args.data)
    preds = pd.read_csv(args.pred)
    df = data.join(preds, how="inner")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # F1 por genero
    g_gen = f1_grouped(df, "genero")
    g_gen.to_csv(out / "f1_por_genero.csv", index=False)
    print("F1 por GENERO:\n", g_gen)

    # bandas etarias
    bins = [0,40,60,200]; labels = ["18-39","40-59","60+"]
    df["banda_edad"] = pd.cut(df["edad"], bins=bins, labels=labels, right=False)
    g_age = f1_grouped(df, "banda_edad")
    g_age.to_csv(out / "f1_por_edad.csv", index=False)
    print("F1 por EDAD:\n", g_age)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)   # reports/preds_nb.csv
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    main(args)
