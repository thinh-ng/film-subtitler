#!/usr/bin/env bash
# Usage: ./batch.sh jobs.txt   — one job per line: <video path>|<run.py flags>
# Transcription (CPU-bound) runs one film at a time; each film's translation (LLM-bound)
# runs in the background so the next film can start transcribing immediately.
cd "$(dirname "$0")"
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
# Stop Git Bash rewriting /mnt/... NAS paths into Windows paths.
export MSYS_NO_PATHCONV=1
mkdir -p work/logs
# Pull later films' audio ahead of time; the CPU sat idle ~2 min per film waiting on the Pi.
# The first film is left to the main loop so two pulls never race for the same file.
( grep -v '^\s*\(#\|$\)' "$1" | tail -n +2 | while IFS='|' read -r video flags; do
    $PY run.py "$video" --stage prefetch < /dev/null
  done > work/logs/prefetch.log 2>&1 ) &
while IFS='|' read -r video flags; do
  [ -z "$video" ] && continue
  case "$video" in \#*) continue;; esac
  name=$(basename "${video%.*}")
  log="work/logs/${name:0:60}.log"
  echo "$(date +%T) START $name"
  if $PY run.py "$video" $flags --stage transcribe < /dev/null > "$log" 2>&1; then
    ( $PY run.py "$video" $flags --stage translate < /dev/null >> "$log" 2>&1 \
        && echo "$(date +%T) DONE $name" || echo "$(date +%T) FAILED $name (see $log)" ) &
  else
    echo "$(date +%T) FAILED transcribe $name (see $log)"
  fi
done < "$1"
wait
echo "$(date +%T) ALL FINISHED"
