"""Timed lines -> film bible -> context-aware Vietnamese (and English) subtitles.

Usage: python translate.py <lines.json> <out_prefix> [--context notes.txt] [--series address.json]
--context adds background the dialogue cannot give (e.g. a recap of earlier episodes).
--series carries each pair's default form of address from one episode to the next; it is read
before the bible is built and updated right after.
Writes <out_prefix>.bible.json, .subs.json, .vi.srt, .en.srt, .report.txt
Re-running resumes: the bible and finished chunks are cached next to the outputs.
"""
import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

MODEL = "opus"
MAX_CHUNK_LINES = 70
CONTEXT_LINES = 15
PARALLEL_CALLS = 8
# Usage limits on the subscription can last minutes to hours; wait rather than fail the film.
RETRY_WAITS = [30, 120, 300, 900, 1800, 3600]
# Netflix's Vietnamese timed-text guide: adult reading speed, line length, cue duration limits.
CPS = 17
LINE_CHARS = 42
MIN_CUE_SECONDS = 5 / 6
MAX_CUE_SECONDS = 7.0
# Two frames at 24 fps, so consecutive cues read as separate subtitles.
CUE_GAP = 2 / 24
# A line followed this quickly would only flash on screen; it shares a cue with the next one.
PAIR_SECONDS = 0.7
MAX_CUE_TEXTS = 2
NOTE_SECONDS = 6.0

LANGUAGE_NAMES = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Mandarin Chinese",
                  "yue": "Cantonese", "no": "Norwegian", "nn": "Norwegian", "de": "German",
                  "fr": "French", "it": "Italian", "es": "Spanish", "vi": "Vietnamese"}

SOURCE_GUIDANCE = {
    "ko": "Korean speech levels (banmal vs jondaetmal, -yo/-nida endings) and address terms "
          "(hyung, oppa, sunbae, titles + nim) are strong evidence for the Vietnamese pronoun pair. "
          "Keep Korean personal names romanized.",
    "ja": "Japanese keigo vs casual speech, name suffixes (-san, -kun, -chan, -sama) and kinship "
          "address (onii-chan, okaa-san) are strong evidence for the Vietnamese pronoun pair. "
          "Note regional dialects (e.g. Kansai-ben) and reflect them in register. Keep names romanized.",
    "zh": "Vietnamese viewers know Chinese names in Sino-Vietnamese (Hán-Việt) form; decide each "
          "name's Hán-Việt rendering once in the glossary and use it everywhere.",
    "de": "German du/Sie is strong evidence for intimacy and hierarchy; the switch between them is "
          "often a plot point.",
    "no": "Norwegian du is universal, so judge politeness from context, age and role.",
}
SOURCE_GUIDANCE["yue"] = SOURCE_GUIDANCE["zh"]
SOURCE_GUIDANCE["nn"] = SOURCE_GUIDANCE["no"]

SUBTITLE_CRAFT = """Subtitle craft rules:
- Translate meaning, intent and tone, never word-for-word. Each line to translate ends with
  {seconds, ≤N chars}: the most a viewer can read while it is on screen. Stay within N; when the
  speech is fast, condense — drop filler, repetition and hesitation, keep the meaning.
- Slang, idioms, swearing: use the natural Vietnamese equivalent at the same intensity.
- Figurative or joking phrasing is translated by what it means between the two characters.
  Vocabulary that asserts a relationship (romantic, sexual, kinship) is used only when the bible
  gives the pair that relationship; a borrowed one is rendered as a joke between them.
- Jokes and wordplay: rebuild the joke with a Vietnamese equivalent that gets the same reaction.
  If that is impossible, keep the meaning; keep a foreign word as-is when Vietnamese viewers would
  know it or it is a brand, title or name.
- Technical, legal, financial or cultural terms: use the established Vietnamese term; if there is
  none, keep the original term or translate literally, whichever a Vietnamese viewer understands.
- Songs: a song a character performs in the story that matters to it (a rap, a toast, a
  lullaby) is translated like dialogue, each line starting with "♪ ". Background or soundtrack
  songs: return "♪" for those lines.
- If a line is clearly not dialogue (transcription noise, a hallucinated "thanks for watching",
  credits text), return an empty string for it."""

SYSTEM = (
    "You are a senior film subtitler producing professional Vietnamese and English subtitles. "
    "You only know the film from the material you are given; do not assume it is any film you know."
)

BIBLE_SCHEMA = {
    "type": "object",
    "required": ["characters", "voice_map", "address", "scenes", "register", "glossary", "notes"],
    "properties": {
        "characters": {"type": "array", "items": {"type": "object",
            "required": ["name", "gender", "age", "role"],
            "properties": {"name": {"type": "string"}, "gender": {"type": "string"},
                           "age": {"type": "string"}, "role": {"type": "string"},
                           "evidence": {"type": "string"}}}},
        "voice_map": {"type": "array", "items": {"type": "object",
            "required": ["voice", "character", "confidence"],
            "properties": {"voice": {"type": "string"}, "character": {"type": "string"},
                           "confidence": {"type": "string"}}}},
        "address": {"type": "array", "items": {"type": "object",
            "required": ["from", "to", "self_term", "other_term", "from_line", "to_line", "why",
                         "default"],
            "properties": {"from": {"type": "string"}, "to": {"type": "string"},
                           "default": {"type": "boolean"},
                           "self_term": {"type": "string"}, "other_term": {"type": "string"},
                           "from_line": {"type": "integer"}, "to_line": {"type": "integer"},
                           "why": {"type": "string"}}}},
        "scenes": {"type": "array", "items": {"type": "object",
            "required": ["start_line", "end_line", "summary", "present"],
            "properties": {"start_line": {"type": "integer"}, "end_line": {"type": "integer"},
                           "summary": {"type": "string"},
                           "present": {"type": "array", "items": {"type": "string"}}}}},
        "register": {"type": "string"},
        "glossary": {"type": "array", "items": {"type": "object",
            "required": ["term", "vi", "en"],
            "properties": {"term": {"type": "string"}, "vi": {"type": "string"},
                           "en": {"type": "string"}, "why": {"type": "string"}}}},
        "notes": {"type": "array", "items": {"type": "object",
            "required": ["line", "vi", "en"],
            "properties": {"line": {"type": "integer"}, "vi": {"type": "string"},
                           "en": {"type": "string"}}}},
    },
}


def chunk_schema(with_english):
    fields = {"id": {"type": "integer"}, "speaker": {"type": "string"},
              "addressee": {"type": "string"}, "pair": {"type": "string"}, "vi": {"type": "string"}}
    if with_english:
        fields["en"] = {"type": "string"}
    return {"type": "object", "required": ["lines"],
            "properties": {"lines": {"type": "array", "items": {
                "type": "object", "required": list(fields), "properties": fields}}}}


def ask(prompt, schema):
    for wait in RETRY_WAITS + [None]:
        result = subprocess.run(
            ["claude", "-p", "--model", MODEL, "--output-format", "json", "--tools", "",
             "--no-session-persistence", "--setting-sources", "", "--system-prompt", SYSTEM,
             "--json-schema", json.dumps(schema)],
            input=prompt, capture_output=True, text=True, encoding="utf-8",
        )
        try:
            envelope = json.loads(result.stdout)
            if result.returncode == 0 and envelope.get("structured_output") is not None:
                return envelope["structured_output"]
            error = envelope.get("result") or result.stderr
        except json.JSONDecodeError:
            error = result.stderr or result.stdout
        if wait is None:
            raise RuntimeError(f"LLM call failed: {error[:500]}")
        print(f"  LLM call failed ({error[:120]!r}); retrying in {wait}s", flush=True)
        time.sleep(wait)


def fmt(line):
    m, s = divmod(int(line["start"]), 60)
    return f'{line["id"]} [{m:02d}:{s:02d}] {line["voice"]}: {line["text"]}'


def with_budget(line, following):
    """The line plus how long its cue can stay up and how many characters fit in that time."""
    limit = line["start"] + MAX_CUE_SECONDS
    until = min(following["start"], limit) if following else limit
    seconds = max(MIN_CUE_SECONDS, until - line["start"], line["end"] - line["start"])
    return f"{fmt(line)}  {{{seconds:.1f}s, ≤{int(seconds * CPS)} chars}}"


def build_bible(lines, language, context, defaults):
    source = LANGUAGE_NAMES.get(language, language)
    transcript = "\n".join(fmt(l) for l in lines)
    extra = f"\nBACKGROUND PROVIDED BY THE USER (trust it):\n{context}\n" if context else ""
    if defaults:
        extra += ("\nESTABLISHED FORMS OF ADDRESS from earlier episodes of this series. Each is that "
                  "direction's default; keep it unless this episode shows a lasting change in the "
                  "relationship, and say so in `why`:\n" + json.dumps(defaults, ensure_ascii=False) + "\n")
    prompt = f"""Below is an automatic transcript of a film's dialogue, detected as {source}. Each
line is `id [mm:ss] voice: text`. The transcript has recognition errors; infer what was meant. Some
lines may be in another language (a character switching languages); they get translated too.
Voice labels come from automatic voice clustering and are noisy — one label can merge similar
voices, one person can be split across labels, V? means unknown. Song lyrics, TV audio or
background speech may appear.
{extra}
Read the WHOLE transcript first, then build a "film bible" that subtitlers translating into
Vietnamese and English will rely on.

1. characters: every speaking character, by the name used in the dialogue or a descriptive label.
   Infer gender and approximate age from evidence; cite it briefly. If the story itself makes
   identities ambiguous, say so in `role`.
2. voice_map: best mapping of voice labels to characters, with confidence.
3. address: for every pair of characters who talk to each other, in EACH direction, the Vietnamese
   self-term and other-term the speaker uses (e.g. bố/con, anh/em, tôi/ông, cháu/bác, tao/mày).
   Choose by age, kinship, intimacy, politeness, power and hostility. When the relationship or the
   speaker's perception of it changes, add separate entries with line ranges and explain in `why`.
   Mark exactly one entry per direction `default`: the form the pair normally uses. A heated or
   one-off moment is its own non-default entry covering only those lines.
   anh/chị as the other-term means the speaker sees the addressee as older or senior; tôi/anh is
   not a neutral default. To an adult clearly younger than the speaker, use cậu/cô (em when close).
   {SOURCE_GUIDANCE.get(language, "")}
4. scenes: consecutive line ranges per scene, a one-sentence summary, and who is present.
   A conversation keeps the same participants until there is positive evidence of a change: the
   new speaker is named or addressed, someone greets or hands over, the scene cuts, or a voice
   label maps to them with high confidence. Topic, apparent location and low-confidence voice
   labels are not evidence; a phone call is one conversation.
5. register: tone and style guidance — genre, era, dialects, how characters speak (crude,
   formal, jargon-heavy), running jokes or catchphrases and how to render them consistently.
6. glossary: names, places, recurring terms, jargon, slang and catchphrases with the Vietnamese
   and English rendering to use everywhere, and why when the choice is not obvious.
7. notes: on-screen explanatory notes, shown in brackets under the subtitle. Use them SPARINGLY — only
   for a term a viewer would genuinely not understand that matters to the scene (technical,
   historical, cultural). Never for jokes, slang or anything the translation itself can carry.
   Each term at most once, on the line where it first matters; roughly no more than one note per
   five minutes. `line` is the line id; `vi` and `en` are the note text, under 60 characters.
   Write `en` only when the source is not English, otherwise leave it empty.

TRANSCRIPT:
{transcript}"""
    return ask(prompt, BIBLE_SCHEMA)


def make_chunks(lines, bible):
    """Group by scene so each call sees whole conversations, splitting overly long scenes."""
    by_id = {l["id"]: l for l in lines}
    chunks = []
    for scene in sorted(bible["scenes"], key=lambda s: s["start_line"]):
        ids = [i for i in range(scene["start_line"], scene["end_line"] + 1) if i in by_id]
        for k in range(0, len(ids), MAX_CHUNK_LINES):
            chunks.append({"ids": ids[k:k + MAX_CHUNK_LINES], "scene": scene})
    covered = {i for c in chunks for i in c["ids"]}
    missing = [l["id"] for l in lines if l["id"] not in covered]
    for k in range(0, len(missing), MAX_CHUNK_LINES):
        chunks.append({"ids": missing[k:k + MAX_CHUNK_LINES], "scene": None})
    return chunks


def translate_chunk(chunk, lines, bible, language, cache_dir):
    cache_path = os.path.join(cache_dir, f"chunk_{chunk['ids'][0]}_{chunk['ids'][-1]}.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    source = LANGUAGE_NAMES.get(language, language)
    with_english = language != "en"
    index = {l["id"]: n for n, l in enumerate(lines)}
    first, last = index[chunk["ids"][0]], index[chunk["ids"][-1]]
    before = lines[max(0, first - CONTEXT_LINES):first]
    after = lines[last + 1:last + 6]
    targets = [with_budget(lines[index[i]], lines[index[i] + 1] if index[i] + 1 < len(lines) else None)
               for i in chunk["ids"]]
    scene = chunk["scene"]["summary"] if chunk["scene"] else "(not assigned to a scene)"
    bible_text = json.dumps({k: bible[k] for k in ("characters", "voice_map", "address", "register", "glossary")},
                            ensure_ascii=False)
    english_rule = (f"- en: the English subtitle, translated directly from the {source} (not via "
                    "the Vietnamese), with the same craft rules.\n") if with_english else ""
    prompt = f"""Source language: {source}.

FILM BIBLE (built from the full transcript):
{bible_text}

CURRENT SCENE: {scene}

PRECEDING LINES (context only, do not translate):
{chr(10).join(fmt(l) for l in before) or "(none)"}

LINES TO TRANSLATE:
{chr(10).join(targets)}

FOLLOWING LINES (context only, do not translate):
{chr(10).join(fmt(l) for l in after) or "(none)"}

For EVERY line to translate, in order:
- speaker: the character speaking. Voice labels are hints only; trust the dialogue flow and bible.
  A conversation keeps the same participants until there is positive evidence of a change: the
  new speaker is named or addressed, someone greets or hands over, the scene cuts, or a voice
  label maps to them with high confidence. Topic, apparent location and low-confidence voice
  labels are not evidence; a phone call is one conversation.
- addressee: who they are speaking to (may differ from the previous speaker; "self", "group" or
  "unknown" are allowed).
- pair: the Vietnamese self-term/other-term used, from the bible's `address` for that direction
  and line range ("-" if the line needs no pronoun).
- vi: the Vietnamese subtitle in natural spoken Vietnamese matching the relationship and emotion.
  Drop pronouns where Vietnamese naturally would. Follow the glossary.
{english_rule}When a sentence spans several ids, translate so the fragments read naturally in sequence, and
keep each id to what that fragment says as far as Vietnamese word order allows: a subtitle that
runs ahead of the speech is gone before the words are heard.
The transcript has recognition errors; translate what was most likely said.

{SUBTITLE_CRAFT}

Return exactly one entry per id."""
    result = ask(prompt, chunk_schema(with_english))
    got = {r["id"] for r in result["lines"]}
    if not set(chunk["ids"]) <= got:
        raise RuntimeError(f"chunk {chunk['ids'][0]}: missing ids {sorted(set(chunk['ids']) - got)}")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result["lines"], f, ensure_ascii=False)
    return result["lines"]


def srt_time(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def reading_seconds(texts):
    return max(MIN_CUE_SECONDS, sum(len(t) for t in texts) / CPS)


def fits_one_cue(texts, speakers):
    """Two lines at most: one speaker's text wraps over both, two speakers get a line each."""
    if len(texts) > MAX_CUE_TEXTS:
        return False
    if len(set(speakers)) == 1:
        return len(" ".join(texts)) <= 2 * LINE_CHARS
    return all(len(t) <= LINE_CHARS for t in texts)


def build_cues(lines, key):
    """Non-overlapping dialogue cues: players stack overlapping cues and shuffle them as each ends.

    A line joins the cue before it when it follows within PAIR_SECONDS, or when that cue would
    otherwise vanish before it can be read; each cue then stays up for its reading time when the
    gap to the next allows.
    """
    cues = []
    for l in sorted(lines, key=lambda l: l["start"]):
        text = l.get(key, "").strip()
        if not text:
            continue
        if cues:
            cue = cues[-1]
            rushed = cue["start"] + reading_seconds(cue["texts"]) > l["start"] - CUE_GAP
            joined = (cue["texts"] + [text], cue["speakers"] + [l.get("speaker")])
            if ((l["start"] - cue["start"] < PAIR_SECONDS or rushed) and fits_one_cue(*joined)
                    and l["end"] - cue["start"] <= MAX_CUE_SECONDS):
                cue["texts"], cue["speakers"] = joined
                cue["end"] = max(cue["end"], l["end"])
                continue
        cues.append({"start": l["start"], "end": l["end"], "texts": [text], "speakers": [l.get("speaker")]})
    for cue, following in zip(cues, cues[1:] + [None]):
        wanted = max(cue["end"], cue["start"] + reading_seconds(cue["texts"]))
        cue["end"] = min(wanted, max(cue["end"], cue["start"] + MAX_CUE_SECONDS))
        if following:
            cue["end"] = min(cue["end"], max(following["start"] - CUE_GAP,
                                             (cue["start"] + following["start"]) / 2))
    return cues


def build_note_cues(lines, key, notes):
    """Notes get their own cues at the top of the screen, up as long as they take to read,
    independent of the dialogue cues below."""
    cues = []
    for l in sorted(lines, key=lambda l: l["start"]):
        note = notes.get(l["id"], {}).get(key, "").strip()
        if not note:
            continue
        if cues:
            cues[-1]["end"] = min(cues[-1]["end"], l["start"] - CUE_GAP)
        seconds = max(NOTE_SECONDS, len(note) / CPS)
        cues.append({"start": l["start"], "end": l["start"] + seconds, "text": f"{{\\an8}}<i>({note})</i>"})
    return cues


def wrap(text):
    """Break a long text over two lines at the most balanced space, bottom line longer, preferring
    a break after punctuation. A text the LLM already split (two speakers in one line) is kept."""
    if len(text) <= LINE_CHARS or "\n" in text:
        return text
    spaces = [i for i, c in enumerate(text) if c == " "]
    if not spaces:
        return text

    def cost(i):
        top, bottom = i, len(text) - i - 1
        return abs(top - bottom) + (4 if top > bottom else 0) - (8 if text[i - 1] in ",.;:?!…" else 0)

    i = min(spaces, key=cost)
    return f"{text[:i]}\n{text[i + 1:]}"


def is_lyric(text):
    return text.lstrip("- ").startswith("♪") and text.strip() != "♪"


def cue_text(cue):
    # Dialogue dashes mark a change of speaker; one speaker's fragments read as one text.
    if len(set(cue["speakers"])) == 1:
        text = " ".join(cue["texts"])
        lines = wrap(text).split("\n")
        if "\n" not in text and is_lyric(text):
            return "\n".join(f"<i>{line}</i>" for line in lines)
    else:
        lines = [f"- {t}" for t in cue["texts"]]
    return "\n".join(f"<i>{line}</i>" if is_lyric(line) else line for line in lines)


def write_srt(path, lines, key, notes):
    dialogue = [{"start": c["start"], "end": c["end"], "text": cue_text(c)} for c in build_cues(lines, key)]
    cues = sorted(dialogue + build_note_cues(lines, key, notes), key=lambda c: c["start"])
    # The byte-order mark stops players from misreading Vietnamese as a legacy code page.
    with open(path, "w", encoding="utf-8-sig") as f:
        for n, cue in enumerate(cues, 1):
            f.write(f"{n}\n{srt_time(cue['start'])} --> {srt_time(cue['end'])}\n{cue['text']}\n\n")


def consistency_report(lines):
    """Flag relationship directions whose pronoun pair changes, so a human can check them."""
    seen, report = {}, []
    for l in lines:
        # A half-pair like "con/-" means the line dropped one pronoun, not that the relationship changed.
        if "-" in l.get("pair", "-").split("/") or l["addressee"] in ("self", "group", "unknown"):
            continue
        key = (l["speaker"], l["addressee"])
        if key in seen and seen[key][0] != l["pair"]:
            report.append(f'{key[0]} -> {key[1]}: "{seen[key][0]}" (line {seen[key][1]}) '
                          f'then "{l["pair"]}" (line {l["id"]})')
        seen[key] = (l["pair"], l["id"])
    return report


def load_defaults(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_defaults(path, defaults, bible):
    """This episode's default for a direction replaces the series one; other directions are kept."""
    merged = {(d["from"], d["to"]): d for d in defaults}
    for a in bible["address"]:
        if a["default"]:
            merged[(a["from"], a["to"])] = {k: a[k] for k in ("from", "to", "self_term", "other_term", "why")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(merged.values()), f, ensure_ascii=False, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lines")
    parser.add_argument("prefix")
    parser.add_argument("--context")
    parser.add_argument("--series")
    args = parser.parse_args()

    with open(args.lines, encoding="utf-8") as f:
        data = json.load(f)
    language, lines = data["language"], data["lines"]
    context = open(args.context, encoding="utf-8").read() if args.context else ""
    cache_dir = f"{args.prefix}.cache"
    os.makedirs(cache_dir, exist_ok=True)

    bible_path = f"{args.prefix}.bible.json"
    if os.path.exists(bible_path):
        with open(bible_path, encoding="utf-8") as f:
            bible = json.load(f)
    else:
        print(f"Pass 1: building film bible from {len(lines)} {language} lines...", flush=True)
        defaults = load_defaults(args.series)
        bible = build_bible(lines, language, context, defaults)
        with open(bible_path, "w", encoding="utf-8") as f:
            json.dump(bible, f, ensure_ascii=False, indent=1)
        if args.series:
            save_defaults(args.series, defaults, bible)
    print(f"  {len(bible['characters'])} characters, {len(bible['scenes'])} scenes, "
          f"{len(bible['notes'])} notes", flush=True)

    chunks = make_chunks(lines, bible)
    print(f"Pass 2: translating {len(chunks)} chunks...", flush=True)
    with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
        results = list(pool.map(lambda c: translate_chunk(c, lines, bible, language, cache_dir), chunks))
    by_id = {r["id"]: r for chunk in results for r in chunk}
    for l in lines:
        l.update({k: v for k, v in by_id[l["id"]].items() if k != "id"})
    if language == "en":
        for l in lines:
            l["en"] = l["text"]

    with open(f"{args.prefix}.subs.json", "w", encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=1)
    notes = {n["line"]: n for n in bible["notes"]}
    write_srt(f"{args.prefix}.vi.srt", lines, "vi", notes)
    write_srt(f"{args.prefix}.en.srt", lines, "en", notes)
    report = consistency_report(lines)
    with open(f"{args.prefix}.report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report) or "No pronoun-pair changes.")
    print(f"Pass 3: {len(report)} pronoun-pair changes flagged. Done.")


if __name__ == "__main__":
    main()
