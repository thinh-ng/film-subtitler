#!/usr/bin/env bash
# One-line batch progress: what is transcribing, what is translating, how many are done.
cd "$(dirname "$0")"
total=$(grep -cv '^\s*\(#\|$\)' ${1:-work/jobs.txt})
batch_log=${2:-work/logs/batch.log}
done=$(grep -c ' DONE ' "$batch_log")
failed=$(grep -c 'FAILED' "$batch_log")
line="$(date +%H:%M) done $done/$total"
[ "$failed" -gt 0 ] && line="$line, FAILED $failed"
for log in work/logs/*.log; do
  case "$log" in */batch*.log|*/prefetch*.log) continue;; esac
  name=$(basename "$log" .log | cut -c1-28)
  if grep -q "uploaded .*\.en\.srt" "$log"; then continue; fi
  if grep -q "Pass 2" "$log"; then
    slug=$(ls -d work/*.cache 2>/dev/null | grep -F "$(echo "$name" | sed 's/[^A-Za-z0-9]\+/_/g' | cut -c1-20)" | head -1)
    n=$(grep -o "translating [0-9]* chunks" "$log" | grep -o "[0-9]*")
    line="$line | translating $name ($(ls "$slug" 2>/dev/null | wc -l)/$n chunks)"
  elif grep -q "Pass 1" "$log"; then
    line="$line | building bible: $name"
  elif grep -q "Clustering" "$log"; then
    line="$line | voice labels: $name"
  else
    sec=$(grep -oE "^\s+[0-9]+\.[0-9]s" "$log" | tail -1 | tr -d ' s')
    [ -n "$sec" ] && line="$line | transcribing $name at $(( ${sec%.*} / 60 )) min"
  fi
done
echo "$line"
grep -q "ALL FINISHED" "$batch_log" && echo "ALL FINISHED" && exit 1
exit 0
