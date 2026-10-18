"""Who each line in a time range sounds like, against the show's voice profiles.

Usage: python probe_voice.py work/<slug> work/<show>.address.json <from h:mm:ss> <to h:mm:ss>
"""
import json
import sys

import numpy as np

import translate


def seconds(clock):
    total = 0
    for part in clock.split(":"):
        total = total * 60 + float(part)
    return total


def main():
    prefix, series, start, end = sys.argv[1], sys.argv[2], seconds(sys.argv[3]), seconds(sys.argv[4])
    with open(f"{prefix}.subs.json", encoding="utf-8") as f:
        lines = json.load(f)
    embeddings = translate.load_embeddings(f"{prefix}.lines.json")
    profiles, _ = translate.load_profiles(translate.profiles_path(series))
    names = [n for n, (_, weight) in profiles.items() if weight >= translate.MIN_PROFILE_LINES]
    vectors = np.array([profiles[n][0] for n in names])
    for line in lines:
        if not start <= line["start"] <= end:
            continue
        if line["id"] in embeddings:
            scores = vectors @ embeddings[line["id"]]
            top = np.argsort(scores)[::-1][:2]
            voice = ", ".join(f"{names[i]} {scores[i]:.2f}" for i in top)
        else:
            voice = "too short to fingerprint"
        print(f"{translate.srt_time(line['start'])[:8]}  given to {line['speaker']:<16} sounds like {voice}"
              f"  | {line['text']}")


if __name__ == "__main__":
    main()
