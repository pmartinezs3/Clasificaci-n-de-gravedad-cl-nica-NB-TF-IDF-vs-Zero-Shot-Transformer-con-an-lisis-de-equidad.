# Clasificación de Gravedad Clínica (ES) — NB TF-IDF vs Zero‑Shot

**Objetivo.** Clasificar la **gravedad** de reportes clínicos breves (`leve`, `moderado`, `severo`)
y comparar un enfoque tradicional (**Naive Bayes + TF-IDF**) con un enfoque **Zero‑Shot Transformer**.
Además, estimar F1 por **género** y **banda etaria** como aproximación simple a **equidad** del modelo.

## Dataset
Conjunto de textos clínicos sintéticos/anónimos con columnas:
`texto_clinico`, `edad`, `genero` (`M`/`F`), `afeccion`, `gravedad`.

## Metodología
- **NB TF‑IDF**: limpieza (minúsculas, stopwords), TF‑IDF, `MultinomialNB`.
- **Zero‑Shot**: clasificación con `candidate_labels = ["leve","moderado","severo"]`.
- **Métricas**: `classification_report`, matriz de confusión.
- **Equidad (rápida)**: F1 ponderado por `genero` y bandas etarias (`18-39`, `40-59`, `60+`).

## Reproducir
```bash
# Crear entorno (ejemplo Windows)
python -m venv .venv
.\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

# 1) Naive Bayes + TF‑IDF
python src/train_nb_tfidf.py --data data/datos.csv --out reports

# 2) Zero‑Shot (opcional, requiere transformers/torch)
python src/eval_zero_shot.py --data data/datos.csv --out reports

# 3) Fairness: F1 por género/edad
python src/fairness.py --pred reports/preds_nb.csv --data data/datos.csv --out reports

# O bien abre el notebook:
jupyter notebook notebooks/Modulo9_Evaluacion_LIME_SHAP.ipynb
```

## Estructura
```
notebooks/  # código principal (exploración, NB, Zero‑Shot, fairness)
reports/    # imágenes y textos de resultados (se generan al ejecutar)
src/        # scripts reutilizables
data/       # coloca aquí el CSV
```

## Próximos pasos
- Validación cruzada y **calibración** de probabilidades.
- Hyperparameter tuning (NB y/o SVM lineal como baseline alternativo).
- Mejorar **prompting** Zero‑Shot y comparar con **fine‑tuning** supervisado.
- Métricas de equidad adicionales (paridad de oportunidades, etc.).

## Licencia
MIT © Pamela Martinez
