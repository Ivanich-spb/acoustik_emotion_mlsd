"""EDA по метаданным Dusha (без аудио): распределение эмоций и длительностей
по доменам crowd/podcast. Используется как внешний прокси для разговорной
русской речи при сравнении с акцентированной англоязычной речью CREMA-D.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAW_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "dusha_paper_setups" / "paper_setups"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "eda_hw2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_FILES = {
    "crowd": "crowd_large.jsonl",
    "podcast": "podcast_large.jsonl",
}


def load_domain(name: str, path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append(
                {"domain": name, "emotion": rec["emotion"], "wav_length": rec["wav_length"]}
            )
    return pd.DataFrame(rows)


def main():
    dfs = []
    for name, fname in DOMAIN_FILES.items():
        path = RAW_DIR / fname
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        dfs.append(load_domain(name, path))
    df = pd.concat(dfs, ignore_index=True)

    cache_dir = RAW_DIR.parent.parent
    df.to_csv(cache_dir / "dusha_meta_full.csv", index=False)

    counts = df.groupby(["domain", "emotion"]).size().unstack("emotion")
    dist = counts.div(counts.sum(axis=1), axis=0)
    dist.to_csv(OUT_DIR / "dusha_emotion_distribution.csv")
    print(dist)

    fig, ax = plt.subplots(figsize=(7, 4))
    dist.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Доля записей")
    ax.set_title("Dusha: распределение эмоциональных меток по доменам")
    ax.legend(title="Эмоция", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dusha_emotion_distribution.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in DOMAIN_FILES:
        sub = df[df.domain == name]
        ax.hist(sub.wav_length, bins=40, alpha=0.6, label=name, range=(0, 15))
    ax.set_xlabel("Длительность записи, с")
    ax.set_ylabel("Количество")
    ax.set_title("Dusha: распределение длительности записей по доменам")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dusha_duration_distribution.png", dpi=150)

    print("\nДлительность (с), статистика по доменам:")
    print(df.groupby("domain").wav_length.describe()[["count", "mean", "std", "min", "50%", "max"]])


if __name__ == "__main__":
    main()
