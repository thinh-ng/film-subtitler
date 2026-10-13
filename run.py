"""End-to-end: video on an SSH-reachable media server -> .ai.vi.srt / .ai.en.srt next to it.

Usage: NAS_HOST=user@host python run.py "/media/Movies/X/X.mkv" [--language ko]
                     [--context recap.txt] [--stage all|transcribe|translate]
Splitting stages lets the next film transcribe (CPU) while this one translates (LLM).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

# The machine holding the videos, as `user@host` for ssh; it needs ffmpeg.
NAS_HOST = os.environ.get("NAS_HOST")
WORK_DIR = "work"
PYTHON = sys.executable


def slug(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:60]


def extract_audio(video_path, wav_path):
    # Decodes only the first audio track on the Pi, so just 16 kHz mono crosses the network.
    probe = f'ffprobe -v error -select_streams a:0 -show_entries stream=channels -of csv=p=0 "{video_path}"'
    channels = int(subprocess.run(["ssh", NAS_HOST, probe], capture_output=True, text=True, check=True).stdout)
    # Surround mixes keep dialogue alone in the centre channel; music on the others was being
    # transcribed as dialogue and drowning out soft voices.
    mix = "-af 'pan=mono|c0=c2'" if channels >= 6 else "-ac 1"
    command = f'ffmpeg -loglevel error -i "{video_path}" -map 0:a:0 -vn {mix} -ar 16000 -f wav -'
    # Write to a temp name so a half-pulled file is never mistaken for a finished prefetch.
    partial = wav_path + ".part"
    with open(partial, "wb") as out:
        subprocess.run(["ssh", NAS_HOST, command], stdout=out, check=True)
    os.replace(partial, wav_path)


def prefetch(video_path):
    """Pull a later film's audio while the CPU is busy transcribing the current one."""
    wav = os.path.join(WORK_DIR, slug(video_path) + ".wav")
    if not os.path.exists(wav) and not os.path.exists(wav + ".part"):
        extract_audio(video_path, wav)


def upload(local_path, video_path, suffix):
    remote = os.path.splitext(video_path)[0] + suffix
    with open(local_path, "rb") as f:
        subprocess.run(["ssh", NAS_HOST, f'cat > "{remote}"'], stdin=f, check=True)
    print(f"  uploaded {remote}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--language")
    parser.add_argument("--context")
    parser.add_argument("--stage", default="all", choices=["all", "transcribe", "translate", "prefetch"])
    args = parser.parse_args()
    if not NAS_HOST:
        sys.exit("Set NAS_HOST to the ssh target holding the videos, e.g. NAS_HOST=user@host")

    os.makedirs(WORK_DIR, exist_ok=True)
    prefix = os.path.join(WORK_DIR, slug(args.video))
    wav, lines = f"{prefix}.wav", f"{prefix}.lines.json"

    if args.stage == "prefetch":
        prefetch(args.video)
        return

    if args.stage in ("all", "transcribe"):
        while os.path.exists(wav + ".part"):
            time.sleep(5)
        if not os.path.exists(wav):
            print(f"[{slug(args.video)}] extracting audio", flush=True)
            extract_audio(args.video, wav)
        command = [PYTHON, "transcribe.py", wav, lines]
        if args.language:
            command += ["--language", args.language]
        subprocess.run(command, check=True)
        os.remove(wav)
        # Cached translations are keyed by line numbers, which a new transcript reshuffles.
        shutil.rmtree(f"{prefix}.cache", ignore_errors=True)
        if os.path.exists(f"{prefix}.bible.json"):
            os.remove(f"{prefix}.bible.json")

    if args.stage in ("all", "translate"):
        command = [PYTHON, "translate.py", lines, prefix]
        if args.context:
            command += ["--context", args.context]
        subprocess.run(command, check=True)
        upload(f"{prefix}.vi.srt", args.video, ".ai.vi.srt")
        upload(f"{prefix}.en.srt", args.video, ".ai.en.srt")


if __name__ == "__main__":
    main()
