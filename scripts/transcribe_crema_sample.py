"""ASR-транскрипция сэмпла CREMA-D (faster-whisper) и сравнение с
эталонными фразами для оценки WER по эмоции/интенсивности.

Пример запуска:
    python scripts/transcribe_crema_sample.py \\
        --raw-dir /path/to/CREMA-D/VideoFlash \\
        --out-dir data/raw/crema_dev_server --sample-size 800

Требует extras: pip install -e ".[audio]"
"""

import argparse
import io
import json
import random
import re
import subprocess
import time
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

FNAME_RE = re.compile(r"^(\d+)_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})$")

SENTENCE_TEXT = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

EMOTION_MAP = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
PUNCT_RE = re.compile(r"[.,!?;:]")


def normalize_words(text: str):
    text = PUNCT_RE.sub("", text.lower()).replace("eleven", "11")
    return text.split()


def word_error_rate(ref: str, hyp: str) -> float:
    ref_words, hyp_words = normalize_words(ref), normalize_words(hyp)
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = (
                dp[i - 1][j - 1]
                if ref_words[i - 1] == hyp_words[j - 1]
                else 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
            )
    return dp[n][m] / max(1, n)


def decode_wav_bytes(path: Path) -> bytes:
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "wav", "-"],
        capture_output=True,
    )
    return p.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=800)
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.raw_dir.glob("*.flv")) or sorted(args.raw_dir.glob("*.mp4"))
    random.seed(args.seed)
    sample = random.sample(files, min(args.sample_size, len(files)))

    print(f"Loading model {args.model}...", flush=True)
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    rows = []
    t0 = time.time()
    for i, path in enumerate(sample):
        m = FNAME_RE.match(path.stem)
        if not m:
            continue
        actor_id, sentence_code, emo_code, level_code = m.groups()
        ref_text = SENTENCE_TEXT.get(sentence_code, "")

        wav_bytes = decode_wav_bytes(path)
        if not wav_bytes:
            continue
        y, sr = sf.read(io.BytesIO(wav_bytes))

        segments, _info = model.transcribe(y, language="en", beam_size=1)
        hyp_text = " ".join(seg.text.strip() for seg in segments).strip()
        wer = word_error_rate(ref_text, hyp_text) if ref_text else None

        rows.append(
            {
                "file": path.stem,
                "emotion": EMOTION_MAP.get(emo_code, emo_code),
                "level": level_code,
                "sentence_code": sentence_code,
                "ref_text": ref_text,
                "hyp_text": hyp_text,
                "wer": wer,
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(sample)} done, elapsed={time.time() - t0:.0f}s", flush=True)

    with open(args.out_dir / "asr_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"DONE. n={len(rows)} total_time={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
