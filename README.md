# Film Subtitler

Makes Vietnamese and English subtitles for any film or episode, in any spoken language, by having
an LLM understand the whole story before it translates a single line.

## Why

Line-by-line machine translation has no idea who is talking to whom. "I love you" is *Mẹ yêu con*
from a mother to her child and *Anh yêu em* between lovers; Vietnamese forces that choice on every
line. This tool reads the film first, the way a human subtitler watches it first.

## How it works

```
video (on the NAS)
  │  1. Listen      speech → timed text in the original language, plus a rough
  │                 "which voice is this" label per line            (local, CPU)
  │                 • surround audio: the centre (dialogue) channel is transcribed first, so
  │                   music and songs don't bury soft voices; speech mixed elsewhere (recaps,
  │                   teasers) is then taken from the full mix
  │                 • speech the first pass skipped (soft voices, a line in another
  │                   language) is found and transcribed again, in its own language;
  │                   low-confidence lines get the same language check; a short burst
  │                   with no words at all is kept only when recognised with high confidence
  │                 • spots where speech recognition got stuck repeating itself are
  │                   redone with a slower, more accurate model
  ▼
  │  2. Understand  the LLM reads the WHOLE transcript once and writes a film bible:
  │                 who each character is, how they relate, which Vietnamese pronoun
  │                 pair each one uses toward each other one (and when that changes),
  │                 scenes, tone, a glossary for names/jargon/slang, and a few notes
  ▼
  │  3. Translate   scene by scene, with the bible attached: for every line decide
  │                 speaker → addressee → pronoun pair, then write the subtitle in
  │                 Vietnamese (and English, straight from the original language)
  ▼
  │  4. Check       flag every place a pronoun pair changes so a human can confirm it
  ▼
<video>.ai.vi.srt  <video>.ai.en.srt   next to the video, ready for Jellyfin / Plex
```

Translation style:

- Meaning and tone over literal wording. Slang, idioms and jokes get Vietnamese equivalents.
- Honorifics and speech levels (Korean, Japanese, German du/Sie...) feed the pronoun choice.
- Hard technical or cultural terms occasionally get a short note in brackets under the subtitle,
  kept rare so the screen stays clean. Cues never overlap; lines spoken in quick succession share
  one cue.
- The LLM is given no title or synopsis for films: context comes from the dialogue alone. For a TV
  episode you can pass a recap of earlier episodes (`--context`).

## Usage

```
export NAS_HOST=user@host  # the machine holding the videos, reachable over ssh
python run.py "/media/Movies/<folder>/<file>.mkv" [--language ko] [--context recap.txt]
./batch.sh jobs.txt        # many videos; one line per video: <path>|<run.py flags>
./status.sh jobs.txt       # one-line progress of a running batch
```

- `--language` is the film's main language, optional (auto-detected). Other languages are found on their own.
- A 2-hour film takes about 15–20 minutes on its own. In a batch, audio is pulled from the NAS
  ahead of time and one film translates while the next transcribes: 9 films took about an hour.
- Re-running resumes: finished LLM work is cached in `work/`.

Needs: the `claude` CLI logged in, ssh access to `NAS_HOST` (with ffmpeg installed there), and the Python env in `.venv`
(`uv venv -p 3.12 .venv && uv pip install -p .venv faster-whisper speechbrain scikit-learn soundfile torch torchaudio`).

## Outputs in `work/`

- `*.bible.json` — what the LLM understood about the film. Read this first when a translation looks off.
- `*.lines.json` — every line with speaker, addressee, pronoun pair and translations.
- `*.report.txt` — pronoun changes to double-check.

## Limits

- Audio only: relationships shown purely visually, and on-screen text (signs, letters), are missed.
- Voice labelling is rough; the LLM fixes most of it from dialogue flow, crowded scenes suffer.
- Speech recognition is weaker for Cantonese and heavy dialects, and invents phrases over music.
  The LLM drops lines that are clearly not dialogue, so check films with long music passages.
- Song lyrics are shown as ♪, not translated.
