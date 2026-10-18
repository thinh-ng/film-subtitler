"""Per-episode report: what the pipeline produced and where the LLM went against the evidence.

Usage: python report.py work/<slug> [work/<slug> ...]
"""
import json
import os
import sys

import translate


def main():
    for prefix in sys.argv[1:]:
        with open(f"{prefix}.bible.json", encoding="utf-8") as f:
            bible = json.load(f)
        with open(f"{prefix}.subs.json", encoding="utf-8") as f:
            lines = json.load(f)
        with open(f"{prefix}.report.txt", encoding="utf-8") as f:
            pronoun_changes = [r for r in f.read().splitlines() if " -> " in r]
        matched = [l for l in lines if l.get("voice_match")]
        overridden = [l for l in matched
                      if l["voice_match"].endswith("(strong)") and not l["voice_match"].startswith(l["speaker"])]
        print(f"== {os.path.basename(prefix)}")
        print(f"  {len(lines)} lines, {len(bible['characters'])} characters, {len(bible['notes'])} notes, "
              f"{len(matched)} lines matched to a voice profile")
        print(f"  pronoun-pair changes flagged: {len(pronoun_changes)}")
        for change in pronoun_changes:
            print(f"    {change}")
        print(f"  strong voice matches the LLM gave to someone else: {len(overridden)}")
        for l in overridden:
            print(f"    {translate.srt_time(l['start'])[:8]}  voice {l['voice_match']}, given to "
                  f"{l['speaker']}: {l['text']}")


if __name__ == "__main__":
    main()
