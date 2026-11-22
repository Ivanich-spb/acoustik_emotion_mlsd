"""Обучение и валидация baseline- и основной модели эмоций по аудиоканалу (ДЗ №3, Этап 2).

Вход - потабличные акустические признаки CREMA-D (см. extract_crema_acoustic_features.py:
базовые признаки + MFCC, спектральные статистики, перцентили pitch), выход - метрики по
фолдам, важность признаков и confusion matrix основной модели (LightGBM), а также
иллюстративная проверка baseline-агрегации (п. 3.2 дизайн-документа) на суррогатной
бинарной метке "высокий/низкий arousal" вместо недоступной реальной экспертной разметки.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Колонки-метаданные файла - не признаки для модели.
NON_FEATURE_COLS = {
    "file",
    "actor_id",
    "sentence_code",
    "emotion_code",
    "emotion",
    "level_code",
    "level",
}

# Группировка по arousal из EDA (docs/eda_report.md, раздел 2): anger/happy/fear -
# высокий pitch и энергия, мало пауз; sad/neutral - низкий pitch и энергия, много пауз.
# disgust исключён - его arousal по EDA неоднозначен.
HIGH_AROUSAL = {"anger", "fear", "happy"}
LOW_AROUSAL = {"sad", "neutral"}


def load_features(path):
    df = pd.read_csv(path)
    df["pitch_missing"] = df["pitch_mean_hz"].isna().astype(int)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    # Единичные сбои извлечения (слишком короткий/тихий клип) закрываем медианой
    # по всей выборке - сам факт сбоя для pitch уже отражён в pitch_missing.
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df, feature_cols


def evaluate(model, x, y, groups, n_splits):
    cv = GroupKFold(n_splits=n_splits)
    fold_scores = []
    oof_pred = np.empty(len(y), dtype=object)
    for train_idx, test_idx in cv.split(x, y, groups):
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(x.iloc[test_idx])
        oof_pred[test_idx] = pred
        fold_scores.append(f1_score(y.iloc[test_idx], pred, average="macro"))
    return np.array(fold_scores), oof_pred


def weighted_aggregation_score(df):
    """Baseline-агрегация из Этапа 2/4: взвешенное правило по нормализованным признакам."""

    def z(col):
        return (df[col] - df[col].mean()) / df[col].std()

    return z("pitch_mean_hz") + z("rms_mean") - z("silence_ratio")


def plot_confusion_matrix(y_true, y_pred, labels, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", values_format=".2f", colorbar=False
    )
    ax.set_title("Main-модель: confusion matrix (out-of-fold, доли по строке)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def illustrative_pilot_correlation(df):
    mask = df["emotion"].isin(HIGH_AROUSAL | LOW_AROUSAL)
    sub = df.loc[mask].copy()
    sub["arousal_binary"] = sub["emotion"].isin(HIGH_AROUSAL).astype(int)
    sub["agg_score"] = weighted_aggregation_score(sub)
    corr, p_value = spearmanr(sub["agg_score"], sub["arousal_binary"])
    return {
        "n": int(mask.sum()),
        "excluded_emotion": "disgust (arousal неоднозначен по EDA)",
        "spearman_r": float(corr),
        "p_value": float(p_value),
        "note": (
            "Иллюстративная проверка на CREMA-D: бинарная метка "
            "высокий/низкий arousal использована как суррогат экспертной оценки "
            "за отсутствием реальной разметки. Не является порогом для прод-пилота."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-csv",
        default="data/raw/crema_dev_server/crema_acoustic_features.csv",
    )
    parser.add_argument("--out-dir", default="reports/hw3")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_features(args.features_csv)
    x = df[feature_cols]
    y = df["emotion"]
    groups = df["actor_id"]

    models = {
        "baseline_logreg": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "main_lgbm": lgb.LGBMClassifier(
            random_state=0, class_weight="balanced", n_estimators=300, verbose=-1
        ),
    }

    labels = sorted(y.unique())
    per_class_f1 = {}
    results = {}
    main_model_fitted = None
    for name, model in models.items():
        scores, oof_pred = evaluate(model, x, y, groups, args.n_splits)
        results[name] = {
            "macro_f1_mean": float(scores.mean()),
            "macro_f1_std": float(scores.std()),
            "fold_scores": [float(s) for s in scores],
        }
        per_class_f1[name] = f1_score(y, oof_pred, labels=labels, average=None)
        if name == "main_lgbm":
            plot_confusion_matrix(y, oof_pred, labels, out_dir / "main_model_confusion_matrix.png")
            main_model_fitted = model.fit(x, y)

    per_class_df = pd.DataFrame(per_class_f1, index=labels)
    per_class_df.index.name = "emotion"
    per_class_df.to_csv(out_dir / "per_class_f1.csv")

    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": main_model_fitted.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(out_dir / "main_model_feature_importance.csv", index=False)

    results["illustrative_pilot_correlation"] = illustrative_pilot_correlation(df)
    results["n_samples"] = int(len(df))
    results["n_speakers"] = int(groups.nunique())
    results["n_features"] = len(feature_cols)
    results["n_splits"] = args.n_splits

    out_path = out_dir / "model_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
