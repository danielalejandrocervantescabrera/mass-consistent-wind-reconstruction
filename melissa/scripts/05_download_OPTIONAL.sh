#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
echo 'The 61 raw files are bundled. Fresh requests may reach 48,800.'
if [[ "${ALLOW_DOWNLOAD:-0}" != 1 ]]; then
 echo 'Opt in with ALLOW_DOWNLOAD=1 bash scripts/05_download_OPTIONAL.sh'; exit 2
fi
"$PYTHON" "$ROOT/src/download_melissa_power_hourly.py" \
 --outdir "$RAW" --start 20251026T09 --end 20251028T21 \
 --exclude '' --n 10 --half-span-lat 2 --half-span-lon 2 \
 --min-elev 10 --max-elev 290 --step-elev 40 --workers 2
