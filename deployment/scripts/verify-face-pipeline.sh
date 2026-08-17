#!/usr/bin/env sh
set -eu

container_name="${1:-frigate}"

docker inspect "$container_name" --format '{{.State.Status}} {{.State.Health.Status}}'
docker logs --since 10m "$container_name" 2>&1 | grep -Ei \
  'watchdog\.(entrence|GF).*ERROR|Failed to sync surface|Unable to read frames' && exit 1 || true
docker top "$container_name" -eo pid,args | grep -E 'qsv|ffmpeg.*(entrence|GF)'

printf '%s\n' 'Face pipeline health checks passed'
