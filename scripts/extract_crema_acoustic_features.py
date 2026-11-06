"""Извлечение акустических признаков из CREMA-D.

Ожидает локальный чек-аут CREMA-D (например, папку VideoFlash/AudioWAV
из официального репозитория) и считает признаки без промежуточного
сохранения WAV на диск (декодирование через ffmpeg-pipe).

Пример запуска:
    python scripts/extract_crema_acoustic_features.py \\
        --raw-dir /path/to/CREMA-D/VideoFlash \\
        --out-dir data/raw/crema_dev_server --limit 50

Требует extras: pip install -e ".[audio]"
"""

import argparse
import concurrent.futures as cf
import io
import json
import re
import subprocess
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

EMOTION_MAP = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
LEVEL_MAP = {"LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified"}
FNAME_RE = re.compile(r"^(\d+)_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})$")


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
    stem = path.stem
    m = FNAME_RE.match(stem)
    if not m:
        return None
    actor_id, sentence_code, emo_code, level_code = m.groups()
    try:
        y, sr = decode_wav(path)
    except Exception as e:  # noqa: BLE001
        return {"file": stem, "error": str(e)[:200]}

    duration = len(y) / sr
    if duration < 0.05:
        return {"file": stem, "error": "too_short"}

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    silence_ratio = float(np.mean(rms < 0.02))

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
        voiced = voiced[~np.isnan(voiced)] if voiced.size else voiced
        pitch_mean = float(np.mean(voiced)) if voiced.size else float("nan")
        pitch_std = float(np.std(voiced)) if voiced.size else float("nan")
        voiced_fraction = float(np.mean(voiced_flag)) if voiced_flag is not None else float("nan")
    except Exception:  # noqa: BLE001
        pitch_mean = pitch_std = voiced_fraction = float("nan")

    return {
        "file": stem,
        "actor_id": actor_id,
        "sentence_code": sentence_code,
        "emotion_code": emo_code,
        "emotion": EMOTION_MAP.get(emo_code, emo_code),
        "level_code": level_code,
        "level": LEVEL_MAP.get(level_code, level_code),
        "duration_s": duration,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "silence_ratio": silence_ratio,
        "pitch_mean_hz": pitch_mean,
        "pitch_std_hz": pitch_std,
        "voiced_fraction": voiced_fraction,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path, help="Папка с *.flv/*.mp4 CREMA-D")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.raw_dir.glob("*.flv")) or sorted(args.raw_dir.glob("*.mp4"))
    if args.limit:
        files = files[: args.limit]
    print(f"Processing {len(files)} files with {args.workers} workers...", flush=True)

    t0 = time.time()
    rows, errors = [], []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(extract_one, files)):
            if res is None:
                continue
            (errors if "error" in res else rows).append(res)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)} done, elapsed={time.time() - t0:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(args.out_dir / "crema_acoustic_features.csv", index=False)
    with open(args.out_dir / "errors.json", "w") as f:
        json.dump(errors, f, indent=2)
    print(f"DONE. rows={len(rows)} errors={len(errors)} total_time={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
