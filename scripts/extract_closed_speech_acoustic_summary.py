"""Агрегированная акустическая статистика по закрытому корпусу речи.

Предназначен для сверки гипотез EDA (см. docs/eda_report.md) на
недоступных в репозитории данных без раскрытия самих данных: скрипт
читает аудио рекурсивно из `--raw-dir`, по каждому файлу считает те же
акустические признаки, что и `extract_crema_acoustic_features.py`, и
сохраняет ТОЛЬКО сводную статистику (среднее/медиану/перцентили) -
никаких путей, имён файлов или иных идентификаторов на выходе.

Пример запуска:
    python scripts/extract_closed_speech_acoustic_summary.py \\
        --raw-dir /path/to/closed/corpus --out-dir data/raw/closed_summary

Требует extras: pip install -e ".[audio]"
"""

import argparse
import concurrent.futures as cf
import io
import json
import subprocess
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def decode_wav(path: Path):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "wav", "-"],
        capture_output=True,
    )
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore")[:300])
    y, sr = sf.read(io.BytesIO(p.stdout))
    return y.astype(np.float32), sr


def extract_one(path: Path):
    try:
        y, sr = decode_wav(path)
    except Exception:  # noqa: BLE001
        return None

    duration = len(y) / sr
    if duration < 0.3:
        return None

    rms = librosa.feature.rms(y=y)[0]
    silence_ratio = float(np.mean(rms < 0.02))

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
        voiced = voiced[~np.isnan(voiced)] if voiced.size else voiced
        pitch_mean = float(np.mean(voiced)) if voiced.size else None
    except Exception:  # noqa: BLE001
        pitch_mean = None

    return {
        "duration_s": duration,
        "rms_mean": float(np.mean(rms)),
        "silence_ratio": silence_ratio,
        "pitch_mean_hz": pitch_mean,
    }


def summarize(values):
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.raw_dir.rglob("*.wav"))
    print(f"Found {len(files)} files, processing with {args.workers} workers...", flush=True)

    t0 = time.time()
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(extract_one, files)):
            if res is not None:
                results.append(res)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(files)}, elapsed={time.time() - t0:.0f}s", flush=True)

    summary = {
        "n_files_total": len(files),
        "n_files_ok": len(results),
        "duration_s": summarize([r["duration_s"] for r in results]),
        "rms_mean": summarize([r["rms_mean"] for r in results]),
        "silence_ratio": summarize([r["silence_ratio"] for r in results]),
        "pitch_mean_hz": summarize([r["pitch_mean_hz"] for r in results]),
    }

    with open(args.out_dir / "acoustic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Распределения для графика - только значения признаков, без путей/имён.
    pooled = {
        "pitch_mean_hz": [r["pitch_mean_hz"] for r in results if r["pitch_mean_hz"] is not None],
        "rms_mean": [r["rms_mean"] for r in results],
        "silence_ratio": [r["silence_ratio"] for r in results],
    }
    with open(args.out_dir / "pooled_values.json", "w") as f:
        json.dump(pooled, f)

    print(json.dumps(summary, indent=2))
    print(f"DONE. total_time={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
