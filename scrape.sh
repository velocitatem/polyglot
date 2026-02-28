#!/usr/bin/env bash
set -euo pipefail

base="https://datacollective.mozillafoundation.org/datasets"
api_base="https://datacollective.mozillafoundation.org/api"
# MDC_API_KEY should be set in the environment, not hardcoded here.
# export MDC_API_KEY=... before running this script.

page=1
while :; do
    ids=$(
        curl -fsSL "$base?page=$page" \
            | grep -oE '/datasets/cm[a-z0-9]+' \
            | sed 's|^/datasets/||' \
            | sort -u
       )

    [ -z "$ids" ] && break

    while IFS= read -r id; do
        # NOTE: If this endpoint requires auth for you, export MDC_API_KEY and uncomment header line.
        curl -fsSL \
             -H 'accept: application/json' \
             ${MDC_API_KEY:+ -H "authorization: Bearer $MDC_API_KEY"} \
             "$api_base/datasets/$id" \
            | jq -r '
        [
          .id,
          .name // "",
          .shortDescription // "",
          .longDescription // "",
          ( .tags? // [] | join(";") )
        ] | @tsv
      ' || true
    done <<<"$ids"

    page=$((page+1))
done | sort -u
