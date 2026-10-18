"""End-to-end: video on an SSH-reachable media server -> .vi.srt / .en.srt next to it.

Usage: NAS_HOST=user@host python run.py "/media/Movies/X/X.mkv" [--language ko]
                     [--context recap.txt] [--series address.json] [--stage all|transcribe|translate]
Splitting stages lets the next film transcribe (CPU) while this one translates (LLM).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

# The machine holding the videos, as `user@host` for ssh; it needs ffmpeg.
NAS_HOST = os.environ.get("NAS_HOST")
WORK_DIR = "work"
# Checksums of every subtitle uploaded, so an upload never replaces one from elsewhere.
UPLOADED = os.path.join(WORK_DIR, "uploaded.json")
PYTHON = sys.executable


def slug(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:60]


def pull_audio(video_path, filter_args, wav_path):
    # Decodes only the first audio track on the Pi, so just 16 kHz mono crosses the network.
    command = f'ffmpeg -loglevel error -i "{video_path}" -map 0:a:0 -vn {filter_args} -ar 16000 -f wav -'
    with open(wav_path, "wb") as out:
        subprocess.run(["ssh", NAS_HOST, command], stdout=out, check=True)


def extract_audio(video_path, wav_path, mix_path):
    """Centre channel to wav_path; for surround, also the full downmix to mix_path."""
    probe = f'ffprobe -v error -select_streams a:0 -show_entries stream=channels -of csv=p=0 "{video_path}"'
    channels = int(subprocess.run(["ssh", NAS_HOST, probe], capture_output=True, text=True, check=True).stdout)
    # Write to a temp name so a half-pulled file is never mistaken for a finished prefetch;
    # the centre file is renamed last, so its presence means both are complete.
    partial = wav_path + ".part"
    try:
        if channels >= 6:
            # Recaps, teasers and some scenes are mixed outside the centre channel.
            pull_audio(video_path, "-ac 1", partial)
            os.replace(partial, mix_path)
            # Surround mixes keep dialogue alone in the centre channel; music on the others was being
            # transcribed as dialogue and drowning out soft voices.
            pull_audio(video_path, "-af 'pan=mono|c0=c2'", partial)
        else:
            pull_audio(video_path, "-ac 1", partial)
        os.replace(partial, wav_path)
    finally:
        # A .part left behind makes every later run of this video wait on it forever.
        if os.path.exists(partial):
            os.remove(partial)


def prefetch(video_path):
    """Pull a later film's audio while the CPU is busy transcribing the current one."""
    prefix = os.path.join(WORK_DIR, slug(video_path))
    if not os.path.exists(f"{prefix}.wav") and not os.path.exists(f"{prefix}.wav.part"):
        extract_audio(video_path, f"{prefix}.wav", f"{prefix}.mix.wav")


def remote_md5(path):
    result = subprocess.run(["ssh", NAS_HOST, f'[ -f "{path}" ] && md5sum < "{path}" || true'],
                            capture_output=True, text=True, check=True)
    return result.stdout[:32] or None


def upload(local_path, video_path, suffix):
    """Writes <video><suffix>, refusing to replace a subtitle this pipeline did not upload."""
    remote = os.path.splitext(video_path)[0] + suffix
    uploaded = set()
    if os.path.exists(UPLOADED):
        with open(UPLOADED, encoding="utf-8") as f:
            uploaded = set(json.load(f))
    existing = remote_md5(remote)
    if existing and existing not in uploaded:
        sys.exit(f"Not overwriting {remote}: it was not made by this pipeline")
    with open(local_path, "rb") as f:
        subprocess.run(["ssh", NAS_HOST, f'cat > "{remote}"'], stdin=f, check=True)
    with open(local_path, "rb") as f:
        uploaded.add(hashlib.md5(f.read()).hexdigest())
    with open(UPLOADED, "w", encoding="utf-8") as f:
        json.dump(sorted(uploaded), f, indent=1)
    print(f"  uploaded {remote}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--language")
    parser.add_argument("--context")
    parser.add_argument("--series")
    parser.add_argument("--stage", default="all", choices=["all", "transcribe", "translate", "prefetch"])
    args = parser.parse_args()
    if not NAS_HOST:
        sys.exit("Set NAS_HOST to the ssh target holding the videos, e.g. NAS_HOST=user@host")

    os.makedirs(WORK_DIR, exist_ok=True)
    prefix = os.path.join(WORK_DIR, slug(args.video))
    wav, mix, lines = f"{prefix}.wav", f"{prefix}.mix.wav", f"{prefix}.lines.json"

    if args.stage == "prefetch":
        prefetch(args.video)
        return

    if args.stage in ("all", "transcribe"):
        while os.path.exists(wav + ".part"):
            time.sleep(5)
        if not os.path.exists(wav):
            print(f"[{slug(args.video)}] extracting audio", flush=True)
            extract_audio(args.video, wav, mix)
        command = [PYTHON, "transcribe.py", wav, lines]
        if args.language:
            command += ["--language", args.language]
        if os.path.exists(mix):
            command += ["--mix", mix]
        subprocess.run(command, check=True)
        os.remove(wav)
        if os.path.exists(mix):
            os.remove(mix)
        # Cached translations are keyed by line numbers, which a new transcript reshuffles.
        shutil.rmtree(f"{prefix}.cache", ignore_errors=True)
        if os.path.exists(f"{prefix}.bible.json"):
            os.remove(f"{prefix}.bible.json")

    if args.stage in ("all", "translate"):
        command = [PYTHON, "translate.py", lines, prefix]
        if args.context:
            command += ["--context", args.context]
        if args.series:
            command += ["--series", args.series]
        subprocess.run(command, check=True)
        upload(f"{prefix}.vi.srt", args.video, ".vi.srt")
        upload(f"{prefix}.en.srt", args.video, ".en.srt")


if __name__ == "__main__":
    main()
