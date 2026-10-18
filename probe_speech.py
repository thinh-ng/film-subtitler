"""Transcribe a stretch of a video with large-v3 and no speech detector, centre channel and full mix.

Usage: NAS_HOST=user@host python probe_speech.py "<NAS video path>" <from h:mm:ss> <to h:mm:ss>
What this hears and the subtitles lack, the pipeline missed; what it does not hear either is not
recoverable by recognition.
"""
import os
import subprocess
import sys

import numpy as np
from faster_whisper import WhisperModel

from probe_voice import seconds
from run import NAS_HOST

SAMPLE_RATE = 16000
CHANNELS = {"centre": "-af 'pan=mono|c0=c2'", "full mix": "-ac 1"}


def pull(video, start, length, channel_args):
    command = (f'ffmpeg -loglevel error -ss {start} -t {length} -i "{video}" -map 0:a:0 -vn '
               f'{channel_args} -ar {SAMPLE_RATE} -f s16le -')
    raw = subprocess.run(["ssh", NAS_HOST, command], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def main():
    if not NAS_HOST:
        sys.exit("Set NAS_HOST to the ssh target holding the videos, e.g. NAS_HOST=user@host")
    video, start, end = sys.argv[1], seconds(sys.argv[2]), seconds(sys.argv[3])
    model = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=os.cpu_count())
    for name, channel_args in CHANNELS.items():
        audio = pull(video, start, end - start, channel_args)
        segments, _ = model.transcribe(audio, language=None, vad_filter=False, word_timestamps=True,
                                       condition_on_previous_text=False, beam_size=5)
        print(f"--- {name}")
        for seg in segments:
            confidence = np.mean([w.probability for w in seg.words]) if seg.words else 0.0
            at = int(start + seg.start)
            print(f"  {at // 3600}:{at % 3600 // 60:02d}:{at % 60:02d}  confidence {confidence:.2f}  "
                  f"not-speech {seg.no_speech_prob:.2f}  {seg.text.strip()}")


if __name__ == "__main__":
    main()
