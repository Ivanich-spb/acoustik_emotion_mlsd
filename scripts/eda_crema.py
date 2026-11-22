"""EDA по акустическим признакам CREMA-D, посчитанным на dev-окружении
(см. CLAUDE.md, не коммитится) и выгруженным сюда как агрегированный CSV.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "crema_dev_server"
OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "eda_hw2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_ORDER = ["neutral", "happy", "sad", "anger", "fear", "disgust"]


def main():
    df = pd.read_csv(DATA_DIR / "crema_acoustic_features.csv")
    df["emotion"] = pd.Categorical(df["emotion"], categories=EMOTION_ORDER, ordered=True)

    print("Всего записей:", len(df))
    print("\nПо эмоциям:\n", df.emotion.value_counts().sort_index())
    print("\nПо уровню интенсивности:\n", df.level.value_counts())
    print("\nДлительность (с):\n", df.duration_s.describe()[["mean", "std", "min", "50%", "max"]])
    print("\nДоля тишины (silence_ratio) по эмоциям:\n", df.groupby("emotion").silence_ratio.mean())
    print("\nСредний pitch (Hz) по эмоциям:\n", df.groupby("emotion").pitch_mean_hz.mean())
    print("\nСредняя энергия (rms_mean) по эмоциям:\n", df.groupby("emotion").rms_mean.mean())
    print("\nПропуски pitch_mean_hz (NaN, нет voiced-сегментов):", df.pitch_mean_hz.isna().sum())

    summary = df.groupby("emotion").agg(
        n=("file", "count"),
        duration_mean=("duration_s", "mean"),
        pitch_mean_hz=("pitch_mean_hz", "mean"),
        pitch_std_hz=("pitch_mean_hz", "std"),
        rms_mean=("rms_mean", "mean"),
        silence_ratio_mean=("silence_ratio", "mean"),
        voiced_fraction_mean=("voiced_fraction", "mean"),
    )
    summary.to_csv(OUT_DIR / "crema_summary_by_emotion.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    df.boxplot(column="pitch_mean_hz", by="emotion", ax=axes[0], rot=45)
    axes[0].set_title("Средний pitch, Гц")
    axes[0].set_xlabel("")
    df.boxplot(column="rms_mean", by="emotion", ax=axes[1], rot=45)
    axes[1].set_title("Средняя энергия (RMS)")
    axes[1].set_xlabel("")
    df.boxplot(column="silence_ratio", by="emotion", ax=axes[2], rot=45)
    axes[2].set_title("Доля тишины в записи")
    axes[2].set_xlabel("")
    fig.suptitle("CREMA-D: акустические признаки по эмоциям")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "crema_acoustic_by_emotion.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    df.duration_s.hist(bins=40, ax=ax)
    ax.set_xlabel("Длительность записи, с")
    ax.set_ylabel("Количество")
    ax.set_title("CREMA-D: распределение длительности записей")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "crema_duration_distribution.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df.emotion.value_counts().reindex(EMOTION_ORDER)
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Количество записей")
    ax.set_title("CREMA-D: баланс классов эмоций")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "crema_class_balance.png", dpi=150)


if __name__ == "__main__":
    main()
