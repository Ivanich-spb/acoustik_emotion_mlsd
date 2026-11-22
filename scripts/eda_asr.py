"""EDA по результатам ASR-транскрипции сэмпла CREMA-D: WER по эмоции и
интенсивности (см. scripts/transcribe_crema_sample.py).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "crema_dev_server"
OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "eda_hw2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_ORDER = ["neutral", "happy", "sad", "anger", "fear", "disgust"]


def main():
    with open(DATA_DIR / "asr_results.json", encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["emotion"] = pd.Categorical(df["emotion"], categories=EMOTION_ORDER, ordered=True)

    print("Всего транскрибировано:", len(df))
    by_emotion = df.groupby("emotion").wer.agg(["mean", "std", "count"])
    print("\nWER по эмоциям:\n", by_emotion)

    overall = df.wer.mean()
    print("\nСредний WER по всему сэмплу:", round(overall, 3))

    by_emotion.to_csv(OUT_DIR / "crema_asr_wer_by_emotion.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column="wer", by="emotion", ax=ax, rot=45)
    ax.set_title("CREMA-D (сэмпл 800): WER ASR по эмоциям")
    ax.set_xlabel("")
    ax.set_ylabel("WER")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "crema_asr_wer_by_emotion.png", dpi=150)

    cols = ["file", "emotion", "ref_text", "hyp_text", "wer"]
    worst = df.sort_values("wer", ascending=False).head(5)[cols]
    print("\nХудшие по WER примеры:\n", worst.to_string(index=False))


if __name__ == "__main__":
    main()
