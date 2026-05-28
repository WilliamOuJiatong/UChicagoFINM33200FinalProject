from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk

QA_START_PATTERNS = [
    r"\bfirst question\b",
    r"\bopen (?:the )?call to questions\b",
    r"\bquestions? and answers?\b",
    r"\bq&a\b",
]

EXCLUDE_SPEAKERS = {
    "operator",
    "unidentified analyst",
    "conference call participant",
    "unknown speaker",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract earnings-call Q&A pairs from HF transcript dataset."
    )
    p.add_argument(
        "--input",
        default="data/raw/sp500_subset_hf",
        help="Path to local HF dataset (save_to_disk format).",
    )
    p.add_argument(
        "--output",
        default="data/raw/qa_raw.csv",
        help="Output CSV path for extracted Q&A pairs.",
    )
    p.add_argument(
        "--min-question-chars",
        type=int,
        default=40,
        help="Minimum question length in characters.",
    )
    p.add_argument(
        "--min-answer-chars",
        type=int,
        default=60,
        help="Minimum answer length in characters.",
    )
    return p.parse_args()


def load_train(path: str) -> Dataset:
    ds = load_from_disk(path)
    if isinstance(ds, DatasetDict):
        return ds["train"]
    return ds


def normalize_speaker(s: str | None) -> str:
    return (s or "").strip()


def find_qa_start(segments: list[dict]) -> int:
    for i, seg in enumerate(segments):
        txt = str(seg.get("text", "")).lower()
        if any(re.search(pat, txt) for pat in QA_START_PATTERNS):
            return i
    return -1


def infer_management_speakers(segments: list[dict], qa_start: int) -> set[str]:
    mgmt: set[str] = set()
    for seg in segments[:qa_start]:
        speaker = normalize_speaker(seg.get("speaker"))
        text = str(seg.get("text", ""))
        if not speaker:
            continue
        if speaker.lower() in EXCLUDE_SPEAKERS:
            continue
        if len(text) >= 200:
            mgmt.add(speaker)
    return mgmt


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_pairs_from_transcript(
    segments: list[dict], mgmt_speakers: set[str], min_q: int, min_a: int
) -> list[dict]:
    pairs: list[dict] = []
    qa_start = find_qa_start(segments)
    if qa_start < 0:
        return pairs

    i = qa_start + 1
    while i < len(segments):
        seg = segments[i]
        speaker = normalize_speaker(seg.get("speaker"))
        text = clean_text(str(seg.get("text", "")))
        speaker_l = speaker.lower()

        # Skip operator/admin turns.
        if speaker_l in EXCLUDE_SPEAKERS or speaker_l == "operator":
            i += 1
            continue

        # Candidate analyst question: speaker not in management roster.
        if speaker and speaker not in mgmt_speakers and len(text) >= min_q:
            question = text
            question_speaker = speaker
            i += 1

            answer_parts: list[str] = []
            answer_speakers: list[str] = []
            while i < len(segments):
                nxt = segments[i]
                n_speaker = normalize_speaker(nxt.get("speaker"))
                n_text = clean_text(str(nxt.get("text", "")))
                n_l = n_speaker.lower()

                if n_l in EXCLUDE_SPEAKERS or n_l == "operator":
                    i += 1
                    continue

                # New analyst question starts: stop answer aggregation.
                if n_speaker and n_speaker not in mgmt_speakers:
                    break

                # Management response block.
                if n_speaker in mgmt_speakers and n_text:
                    answer_parts.append(n_text)
                    if not answer_speakers or answer_speakers[-1] != n_speaker:
                        answer_speakers.append(n_speaker)

                i += 1

            answer = clean_text(" ".join(answer_parts))
            if len(answer) >= min_a:
                pairs.append(
                    {
                        "question": question,
                        "answer": answer,
                        "question_speaker": question_speaker,
                        "answer_speakers": "|".join(answer_speakers),
                    }
                )
            continue

        i += 1

    return pairs


def main():
    args = parse_args()
    ds = load_train(args.input)

    rows: list[dict] = []
    for item in ds:
        segments = item.get("structured_content")
        if not isinstance(segments, list) or not segments:
            continue

        qa_start = find_qa_start(segments)
        if qa_start < 0:
            continue

        mgmt_speakers = infer_management_speakers(segments, qa_start)
        if not mgmt_speakers:
            continue

        pairs = extract_pairs_from_transcript(
            segments,
            mgmt_speakers,
            min_q=args.min_question_chars,
            min_a=args.min_answer_chars,
        )
        if not pairs:
            continue

        for p in pairs:
            rows.append(
                {
                    "company": item.get("company_name", ""),
                    "ticker": item.get("symbol", ""),
                    "date": item.get("date", ""),
                    "question": p["question"],
                    "answer": p["answer"],
                    "source_url": "https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts",
                    "question_speaker": p["question_speaker"],
                    "question_role": "analyst",
                    "answer_speakers": p["answer_speakers"],
                    "answer_role": "management",
                }
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    print(f"Wrote: {out}")
    print(f"Rows: {len(df)}")
    if not df.empty:
        print("Rows per ticker (top 15):")
        print(df["ticker"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
