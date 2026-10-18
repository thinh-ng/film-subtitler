"""Audio -> timed lines with anonymous voice clusters.

Usage: python transcribe.py <audio.wav> <out.json> [--language ko] [--mix full_mix.wav]
Language is auto-detected when omitted. --mix is the full downmix of a surround track whose
<audio.wav> is the centre channel; speech mixed outside the centre is taken from it.
"""
import argparse
import functools
import json
import os
import re
import time

import numpy as np
import soundfile as sf
import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier

SAMPLE_RATE = 16000
MAX_CUE_SECONDS = 6.0
MAX_CUE_CHARS = {"default": 84, "cjk": 32}
CJK_LANGUAGES = {"zh", "yue", "ja", "ko"}
SENTENCE_ENDS = ".?!。？！"
CLAUSE_ENDS = ",;:、，；：—–"
# Lines shorter than this give unreliable voice embeddings; the LLM decides their speaker.
MIN_EMBED_SECONDS = 0.8
# Cosine distance; lower = more speakers. Untuned starting point.
CLUSTER_THRESHOLD = 0.75
# The same 1-8 characters four or more times in a row.
LOOP = re.compile(r"(.{1,8}?)\1{3,}")
# A whole phrase three times in a row.
PHRASE_LOOP = re.compile(r"(.{5,20}?)\1{2,}")
# 16 on the 32-thread machine it was measured on; fewer chunks in flight on smaller CPUs.
BATCH_SIZE = max(1, min(16, (os.cpu_count() or 2) // 2))
# More sensitive than faster-whisper's default (0.5) so quiet voices count as speech.
HOLE_VAD = VadOptions(threshold=0.3, min_silence_duration_ms=500, speech_pad_ms=200)
MIN_HOLE_SECONDS = 1.5
MAX_HOLE_SECONDS = 25.0
MIN_SWEEP_SECONDS = 2.0
SWEEP_CONFIDENCE = 0.5
# Whisper writes a word for a sniff or a door as readily as for a short reply; on wordless bursts
# real speech scored 0.83+, noise 0.56 or less.
MIN_SHORT_CONFIDENCE = 0.7
# Speech found only in the full mix sits under music and effects, so it needs firmer evidence.
MIN_MIX_CONFIDENCE = 0.6
# Mean word confidence below which a line may be another language rendered in the main one.
DOUBTFUL_CONFIDENCE = 0.65
# Chinese varieties are confused with each other on short clips; keep the film's own.
SAME_LANGUAGE = [{"zh", "yue"}]


def voices_path(lines_path):
    return lines_path.removesuffix(".lines.json") + ".voices.npz"


@functools.cache
def accurate_model():
    """Full large-v3: several times slower than turbo, so only for spans turbo got wrong or missed."""
    return WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=os.cpu_count())


def transcribe(model, audio, language):
    pipeline = BatchedInferencePipeline(model)
    segments, info = pipeline.transcribe(
        audio, language=language, word_timestamps=True,
        vad_filter=True, batch_size=BATCH_SIZE, beam_size=5,
    )
    words = []
    for seg in segments:
        print(f"  {seg.end:7.1f}s", flush=True)
        words.extend(as_word(w) for w in seg.words)
    return words, info.language


def as_word(w, offset=0.0, source="centre"):
    return {"start": w.start + offset, "end": w.end + offset, "text": w.word, "p": w.probability,
            "source": source}


def is_looped(text):
    """Whisper sometimes repeats a phrase until the window ends instead of transcribing the speech."""
    letters = re.sub(r"[\W_]+", "", text)
    return LOOP.search(letters) is not None or PHRASE_LOOP.search(letters) is not None


def repair_loops(audio, words, language):
    """Re-transcribe only the looped spans with full large-v3 and temperature fallback.

    Turbo loops mostly on less common languages (Cantonese lost ~20% of its lines); fixing just
    those spans keeps the fast path for the rest of the film.
    """
    windows = []
    lines = split_into_lines(words, language)
    for i, line in enumerate(lines):
        # A loop whose repeats each end a sentence is split into identical one-word lines.
        repeated = any(lines[j]["text"] == lines[j + 1]["text"] == lines[j + 2]["text"]
                       for j in range(max(0, i - 2), min(i + 1, len(lines) - 2)))
        if is_looped(line["text"]) or repeated:
            start, end = max(0, line["start"] - 1.0), line["end"] + 1.0
            if windows and start <= windows[-1][1]:
                windows[-1][1] = max(windows[-1][1], end)
            else:
                windows.append([start, end])
    if not windows:
        return words
    total = sum(e - s for s, e in windows)
    print(f"Repairing {len(windows)} looped spans ({total / 60:.1f} min) with large-v3...", flush=True)
    model = accurate_model()
    for start, end in windows:
        chunk = audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
        segments, _ = model.transcribe(chunk, language=language, word_timestamps=True,
                                       condition_on_previous_text=False, beam_size=5)
        # large-v3 loops too, just less often; a span it also loops on is left empty.
        fixed = [as_word(w, start) for seg in segments if not is_looped(seg.text) for w in seg.words]
        words = [w for w in words if w["end"] <= start or w["start"] >= end] + fixed
    words.sort(key=lambda w: w["start"])
    return words


def best_cut(words):
    """Where to cut an over-long line: after a comma or at the longest pause, in its later two
    thirds, so neither part is a stranded word or two."""

    def score(k):
        after_clause = words[k - 1]["text"].strip()[-1:] in CLAUSE_ENDS
        return after_clause + (words[k]["start"] - words[k - 1]["end"]) + k / len(words) / 10

    return max(range(max(1, len(words) // 3), len(words)), key=score)


def split_into_lines(words, language):
    """Cut at sentence ends, long pauses, or cue size limits so each line is one speaker turn."""
    max_chars = MAX_CUE_CHARS["cjk" if language in CJK_LANGUAGES else "default"]
    lines, current = [], []

    def flush(part):
        from_mix = sum(x["source"] == "mix" for x in part) * 2 > len(part)
        lines.append({"start": part[0]["start"], "end": part[-1]["end"],
                      "text": "".join(x["text"] for x in part).strip(),
                      "p": sum(x["p"] for x in part) / len(part),
                      "source": "mix" if from_mix else "centre"})

    for i, w in enumerate(words):
        current.append(w)
        text = "".join(x["text"] for x in current).strip()
        next_gap = words[i + 1]["start"] - w["end"] if i + 1 < len(words) else 99
        sentence_end = w["text"].strip()[-1:] in SENTENCE_ENDS
        too_big = current[-1]["end"] - current[0]["start"] > MAX_CUE_SECONDS or len(text) > max_chars
        if sentence_end or next_gap > 0.7:
            flush(current)
            current = []
        elif too_big and len(current) > 1:
            cut = best_cut(current)
            flush(current[:cut])
            current = current[cut:]
    if current:
        flush(current)
    return lines


def assign_voices(audio_by_source, lines):
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"},
        savedir="models/ecapa",
    )
    embeddable = [i for i, l in enumerate(lines) if l["end"] - l["start"] >= MIN_EMBED_SECONDS]
    embeddings = []
    with torch.no_grad():
        for i in embeddable:
            audio = audio_by_source[lines[i]["source"]]
            chunk = audio[int(lines[i]["start"] * SAMPLE_RATE): int(lines[i]["end"] * SAMPLE_RATE)]
            emb = encoder.encode_batch(torch.from_numpy(chunk).unsqueeze(0)).squeeze().numpy()
            embeddings.append(emb / np.linalg.norm(emb))
    # Clustering refuses fewer than two samples.
    labels = [0] * len(embeddings)
    if len(embeddings) > 1:
        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=CLUSTER_THRESHOLD
        ).fit_predict(np.array(embeddings))
    for i, label in zip(embeddable, labels):
        lines[i]["voice"] = f"V{label}"
    for line in lines:
        line.setdefault("voice", "V?")
    return dict(zip(embeddable, embeddings))


def find_holes(audio, words):
    """Stretches the speech detector hears as voice but the main pass wrote no words for.

    Returns (holes, bursts): gaps of MIN_HOLE_SECONDS or more, and shorter speech regions that
    have no words at all.
    """
    steps_per_second = 10
    covered = np.zeros(len(audio) * steps_per_second // SAMPLE_RATE + 1, bool)
    for w in words:
        covered[max(0, int(w["start"] * steps_per_second) - 3): int(w["end"] * steps_per_second) + 3] = True
    holes, bursts = [], []
    for region in get_speech_timestamps(audio, HOLE_VAD):
        a = region["start"] * steps_per_second // SAMPLE_RATE
        b = region["end"] * steps_per_second // SAMPLE_RATE
        if b > a and not covered[a:b].any() and (b - a) / steps_per_second < MIN_HOLE_SECONDS:
            bursts.append((region["start"] / SAMPLE_RATE, region["end"] / SAMPLE_RATE))
            continue
        gaps = np.flatnonzero(np.diff(np.concatenate(([1], covered[a:b], [1])).astype(int)))
        for gap_start, gap_end in zip(gaps[::2], gaps[1::2]):
            if (gap_end - gap_start) / steps_per_second >= MIN_HOLE_SECONDS:
                holes.extend(split_span((a + gap_start) / steps_per_second, (a + gap_end) / steps_per_second))
    return holes, bursts


def split_span(start, end):
    """Whisper hears 30 s at a time; a longer hole came back as 30 s of words all stamped at its
    start. Cut it into equal pieces that each fit."""
    pieces = int(np.ceil((end - start) / MAX_HOLE_SECONDS))
    step = (end - start) / pieces
    return [(start + k * step, start + (k + 1) * step) for k in range(pieces)]


def group_holes(holes, max_gap):
    """Holes that fit in one 30 s window, no more than max_gap seconds apart."""
    groups = []
    for hole in holes:
        if groups and hole[0] - groups[-1][-1][1] <= max_gap and hole[1] - groups[-1][0][0] <= 30:
            groups[-1].append(hole)
        else:
            groups.append([hole])
    return groups


def is_main(language, main_language):
    return language == main_language or any({language, main_language} <= same for same in SAME_LANGUAGE)


def detect_languages(model, audio, groups, main_language):
    """Language of each group of spans, or main_language when unsure.

    The small model settles most groups ~8x faster; only doubtful ones go to the main model.
    """
    scanner = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=os.cpu_count())
    languages = []
    for group in groups:
        chunk = np.concatenate([audio[int(s * SAMPLE_RATE): int(e * SAMPLE_RATE)] for s, e in group])
        language, probability, _ = scanner.detect_language(chunk)
        if not (is_main(language, main_language) and probability >= 0.5):
            language, probability, _ = model.detect_language(chunk)
        languages.append(main_language if probability < 0.5 or is_main(language, main_language) else language)
    return languages


def mean_probability(words):
    return sum(w.probability for w in words) / len(words) if words else 0.0


def sweep_gaps(audio, words, language):
    """Transcribe every stretch with no words with large-v3 and no speech detector.

    The detector gates every other pass and does not hear whispers or talk under a helicopter,
    while large-v3 transcribes both at 0.8+ confidence once pointed at them.
    """
    started = time.time()
    gaps, last = [], 0.0
    for w in sorted(words, key=lambda w: w["start"]):
        if w["start"] - last >= MIN_SWEEP_SECONDS:
            gaps.extend(split_span(last, w["start"]))
        last = max(last, w["end"])
    if len(audio) / SAMPLE_RATE - last >= MIN_SWEEP_SECONDS:
        gaps.extend(split_span(last, len(audio) / SAMPLE_RATE))
    found = transcribe_spans(accurate_model(), audio, [[g] for g in gaps], language,
                             min_confidence=SWEEP_CONFIDENCE)
    print(f"Swept {len(gaps)} stretches without words ({sum(e - s for s, e in gaps) / 60:.1f} min) "
          f"with large-v3 in {time.time() - started:.0f}s: kept {len(found)} words", flush=True)
    return sorted(words + found, key=lambda w: w["start"])


def transcribe_spans(model, audio, groups, language, source="centre", min_confidence=0.0):
    """Transcribe each group whole for context, keeping only words that land inside its spans.

    Packing spans into shared windows was faster but recovered a third as many words.
    A segment's words inside the spans are dropped together when their mean probability is
    below min_confidence.
    """
    clips = [{"start": g[0][0], "end": g[-1][1]} for g in groups]
    spans = [span for g in groups for span in g]
    segments, _ = BatchedInferencePipeline(model).transcribe(
        audio, language=language, clip_timestamps=clips,
        word_timestamps=True, batch_size=BATCH_SIZE, beam_size=5,
    )
    found = []
    for seg in segments:
        # Whisper's own test for "this was not speech"; holes are often music or breathing.
        if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0 or is_looped(seg.text):
            continue
        # "Thank you." and "Subtitles by ..." invented over silence scored 0.61 or less; real
        # speech that Whisper also doubted was speech scored 0.82+.
        if seg.no_speech_prob > 0.6 and mean_probability(seg.words) < 0.75:
            continue
        inside = [w for w in seg.words if any(s <= (w.start + w.end) / 2 <= e for s, e in spans)]
        if inside and sum(w.probability for w in inside) / len(inside) >= min_confidence:
            found.extend(as_word(w, source=source) for w in inside)
    return found


def fill_holes(model, audio, words, main_language, source="centre", min_confidence=0.0):
    """Transcribe speech the main pass skipped: another language, a soft voice, overlapping talk.

    Whisper told to expect one language often writes nothing for speech in another, so holes get
    their own language detection. Short wordless bursts are too short to tell a language from and
    are transcribed in the main one.
    """
    holes, bursts = find_holes(audio, words)
    if bursts:
        found = transcribe_spans(model, audio, group_holes(bursts, max_gap=3), main_language, source,
                                 max(min_confidence, MIN_SHORT_CONFIDENCE))
        print(f"Checked {len(bursts)} short wordless bursts ({source}): kept {len(found)} words", flush=True)
        words = sorted(words + found, key=lambda w: w["start"])
    if not holes:
        return words
    started = time.time()
    # Holes of 3 s or more get their own detection: a 30 s window can mix a foreign line with
    # soft speech in the main language. Shorter holes are too short to tell and are detected
    # on their whole 30 s window of holes instead.
    long_holes = [h for h in holes if h[1] - h[0] >= 3]
    language_of = dict(zip(long_holes, detect_languages(model, audio, [[h] for h in long_holes], main_language)))
    windows = [w for w in group_holes(holes, max_gap=30) if any(h not in language_of for h in w)]
    for window, language in zip(windows, detect_languages(model, audio, windows, main_language)):
        language_of.update((h, language) for h in window if h not in language_of)
    # Transcribe in tighter groups: clips that also span already-transcribed speech recovered
    # 30% fewer words, since Whisper again favours the clear speech over the missed parts.
    by_language = {}
    for hole in holes:
        by_language.setdefault(language_of[hole], []).append(hole)
    by_language = {lang: group_holes(hs, max_gap=3) for lang, hs in by_language.items()}
    summary = ", ".join(f"{lang} {sum(len(g) for g in groups)}" for lang, groups in by_language.items())
    print(f"Filling {len(holes)} holes ({source}, {sum(e - s for s, e in holes) / 60:.1f} min): {summary}",
          flush=True)
    for language, groups in by_language.items():
        words.extend(transcribe_spans(model, audio, groups, language, source, min_confidence))
    print(f"  filled in {time.time() - started:.0f}s", flush=True)
    words.sort(key=lambda w: w["start"])
    return words


def recheck_doubtful(model, audio, words, main_language):
    """Redo low-confidence lines that turn out to be in another language.

    Told to expect one language, Whisper sometimes writes an invented line in it for speech in
    another. Only low-confidence lines are checked, so a film's main language costs almost nothing.
    """
    doubtful = [(l["start"], l["end"]) for l in split_into_lines(words, main_language)
                if l["p"] < DOUBTFUL_CONFIDENCE and l["end"] - l["start"] >= 1.0]
    if not doubtful:
        return words
    started = time.time()
    groups = group_holes(doubtful, max_gap=3)
    foreign = [(group, language) for group, language
               in zip(groups, detect_languages(model, audio, groups, main_language))
               if language != main_language]
    print(f"Rechecked {len(doubtful)} low-confidence lines in {time.time() - started:.0f}s: "
          f"{sum(len(g) for g, _ in foreign)} in another language "
          f"({', '.join(sorted({lang for _, lang in foreign})) or 'none'})", flush=True)
    for group, language in foreign:
        words = [w for w in words if not any(s <= (w["start"] + w["end"]) / 2 <= e for s, e in group)]
        words.extend(transcribe_spans(model, audio, [group], language))
    words.sort(key=lambda w: w["start"])
    return words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("out")
    parser.add_argument("--language")
    parser.add_argument("--mix")
    args = parser.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32")
    assert sr == SAMPLE_RATE, "audio must be 16 kHz mono"
    audio_by_source = {"centre": audio}
    print("Transcribing...", flush=True)
    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=os.cpu_count())
    words, language = transcribe(model, audio, args.language)
    words = fill_holes(model, audio, words, language)
    words = sweep_gaps(audio, words, language)
    words = recheck_doubtful(model, audio, words, language)
    words = repair_loops(audio, words, language)
    if args.mix:
        mix, sr = sf.read(args.mix, dtype="float32")
        assert sr == SAMPLE_RATE, "mix must be 16 kHz mono"
        audio_by_source["mix"] = mix
        # After loop repair, which re-transcribes from the centre channel and would erase these.
        words = fill_holes(model, mix, words, language, "mix", MIN_MIX_CONFIDENCE)
    lines = split_into_lines(words, language)
    for line in lines:
        del line["p"]
    print(f"{len(lines)} lines ({language}, {sum(l['source'] == 'mix' for l in lines)} from the full mix). "
          "Clustering voices...", flush=True)
    embeddings = assign_voices(audio_by_source, lines)
    for line in lines:
        del line["source"]
    for n, line in enumerate(lines, 1):
        line["id"] = n
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"language": language, "lines": lines}, f, ensure_ascii=False, indent=1)
    # Kept so translation can match each line's voice to the characters of earlier episodes.
    np.savez(voices_path(args.out), ids=[lines[i]["id"] for i in embeddings],
             embeddings=np.array(list(embeddings.values())))
    print(f"Done: {len(lines)} lines, {len({l['voice'] for l in lines})} voice labels")


if __name__ == "__main__":
    main()
