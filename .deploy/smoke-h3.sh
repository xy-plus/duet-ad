#!/usr/bin/env bash
set -euo pipefail
umask 077

usage() {
  echo "Usage: RUN_PAID_SMOKE=1 ACCESS_TOKEN=<hidden> $0 <video-path>" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi
if [[ "${RUN_PAID_SMOKE:-}" != "1" ]]; then
  echo "Refusing paid smoke: set RUN_PAID_SMOKE=1 after explicit approval." >&2
  exit 2
fi
if [[ -z "${ACCESS_TOKEN:-}" ]]; then
  echo "ACCESS_TOKEN must be present in the environment." >&2
  exit 2
fi
if [[ "$ACCESS_TOKEN" == *$'\n'* || "$ACCESS_TOKEN" == *$'\r'* ]]; then
  echo "ACCESS_TOKEN must not contain a newline." >&2
  exit 2
fi

video=$1
if [[ ! -f "$video" || ! -r "$video" ]]; then
  echo "Video is missing or unreadable: $video" >&2
  exit 2
fi

base_url=${BASE_URL:-http://127.0.0.1:3212}
base_url=${base_url%/}
poll_interval=${POLL_INTERVAL_S:-2}
prepare_timeout=${PREPARE_TIMEOUT_S:-2400}
generation_timeout=${GENERATION_TIMEOUT_S:-3600}
fit_choice=${FIT_MODE:-pad}
dialogue_mode=${DIALOGUE_MODE:-auto}

if [[ "$fit_choice" != "crop" && "$fit_choice" != "pad" ]]; then
  echo "FIT_MODE must be crop or pad." >&2
  exit 2
fi
if [[ "$dialogue_mode" != "auto" && "$dialogue_mode" != "none" ]]; then
  echo "DIALOGUE_MODE in this smoke must be auto or none." >&2
  exit 2
fi
if ! [[ "$poll_interval" =~ ^[0-9]+$ && "$prepare_timeout" =~ ^[0-9]+$ && "$generation_timeout" =~ ^[0-9]+$ ]]; then
  echo "Timeout and polling values must be non-negative integers." >&2
  exit 2
fi

python_bin=.venv/bin/python
if [[ ! -x "$python_bin" ]]; then
  python_bin=$(command -v python3 || true)
fi
if [[ -z "$python_bin" ]]; then
  echo "python3 is required to parse API JSON safely." >&2
  exit 2
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required." >&2
  exit 2
fi

tmp_dir=$(mktemp -d)
auth_header=$tmp_dir/auth.header
cleanup() {
  rm -rf -- "$tmp_dir"
}
trap cleanup EXIT INT TERM
printf 'Authorization: Bearer %s\n' "$ACCESS_TOKEN" >"$auth_header"

json_get() {
  local path=$1
  "$python_bin" -c '
import json, sys
value = json.load(sys.stdin)
for key in sys.argv[1].split("."):
    if not isinstance(value, dict) or key not in value:
        print("")
        raise SystemExit(0)
    value = value[key]
if value is True:
    print("true")
elif value is False:
    print("false")
elif value is None:
    print("")
else:
    print(value)
' "$path"
}

new_request_id() {
  "$python_bin" -c 'import uuid; print(uuid.uuid4())'
}

get_detail() {
  curl --fail-with-body --silent --show-error \
    --header "@$auth_header" \
    "$base_url/api/conversations/$1"
}

poll_prepare() {
  local cid=$1 deadline detail status error
  deadline=$(( $(date +%s) + prepare_timeout ))
  while (( $(date +%s) <= deadline )); do
    detail=$(get_detail "$cid")
    status=$(json_get status <<<"$detail")
    case "$status" in
      done)
        printf '%s' "$detail"
        return 0
        ;;
      failed)
        error=$(json_get error <<<"$detail")
        echo "Input preparation failed for $cid: ${error:-unknown_error}" >&2
        return 1
        ;;
      queued|processing)
        sleep "$poll_interval"
        ;;
      *)
        echo "Unexpected preparation status for $cid: ${status:-missing}" >&2
        return 1
        ;;
    esac
  done
  echo "Input preparation timed out for $cid." >&2
  return 1
}

poll_generation() {
  local cid=$1 deadline detail status error has_video attempt
  deadline=$(( $(date +%s) + generation_timeout ))
  while (( $(date +%s) <= deadline )); do
    detail=$(get_detail "$cid")
    status=$(json_get generation.status <<<"$detail")
    attempt=$(json_get generation.attempt <<<"$detail")
    case "$status" in
      succeeded)
        has_video=$(json_get has_video <<<"$detail")
        if [[ "$has_video" != "true" ]]; then
          echo "Generation succeeded without generated.mp4 for $cid." >&2
          return 1
        fi
        echo "H3 smoke succeeded: cid=$cid attempt=${attempt:-unknown}"
        return 0
        ;;
      failed)
        error=$(json_get generation.error <<<"$detail")
        echo "H3 smoke stopped: cid=$cid status=failed error=${error:-unknown_error}; only an explicit manual retry may use a new client_request_id." >&2
        return 1
        ;;
      submission_unknown)
        error=$(json_get generation.error <<<"$detail")
        echo "H3 smoke locked: cid=$cid status=submission_unknown error=${error:-unknown_error}; verify the original provider POST before any operator action." >&2
        return 1
        ;;
      resume_required)
        error=$(json_get generation.error <<<"$detail")
        echo "H3 smoke paused: cid=$cid status=resume_required error=${error:-unknown_error}; manually continue in the UI with the original client_request_id and frozen dialogue/fit. This script will not POST again." >&2
        return 1
        ;;
      queued|running)
        sleep "$poll_interval"
        ;;
      *)
        echo "Unexpected generation status for $cid: ${status:-missing}" >&2
        return 1
        ;;
    esac
  done
  echo "H3 generation timed out for $cid; do not resubmit automatically." >&2
  return 1
}

upload_id=$(new_request_id)
echo "Creating smoke conversation through the formal API..."
create_response=$(curl --fail-with-body --silent --show-error \
  --header "@$auth_header" \
  --form "file=@$video" \
  --form "note=H3 production smoke" \
  --form "client_request_id=$upload_id" \
  --form "voice_mode=keep" \
  "$base_url/api/conversations")
cid=$(json_get id <<<"$create_response")
if ! [[ "$cid" =~ ^[0-9a-f]{32}$ ]]; then
  echo "Create response did not contain a valid conversation id." >&2
  exit 1
fi
echo "Created cid=$cid; waiting for frozen input."

detail=$(poll_prepare "$cid")
read_only=$(json_get read_only <<<"$detail")
receipt_version=$(json_get receipt_version <<<"$detail")
fit_required=$(json_get fit_required <<<"$detail")
if [[ "$read_only" != "false" || "$receipt_version" != "1" ]]; then
  echo "Conversation is not an operable schema-v2 frozen input: cid=$cid" >&2
  exit 1
fi
if [[ "$fit_required" == "true" ]]; then
  selected_fit=$fit_choice
else
  selected_fit=none
fi

generation_id=$(new_request_id)
payload=$("$python_bin" -c '
import json, sys
print(json.dumps({
    "confirm": True,
    "client_request_id": sys.argv[1],
    "dialogue_mode": sys.argv[2],
    "fit_mode": sys.argv[3],
}, ensure_ascii=False, separators=(",", ":")))
' "$generation_id" "$dialogue_mode" "$selected_fit")

echo "Submitting H3: cid=$cid dialogue=$dialogue_mode fit=$selected_fit"
submit_response=$(curl --fail-with-body --silent --show-error \
  --header "@$auth_header" \
  --header 'Content-Type: application/json' \
  --data "$payload" \
  "$base_url/api/conversations/$cid/submit")
submit_status=$(json_get status <<<"$submit_response")
if [[ "$submit_status" != "queued" ]]; then
  echo "Submit was not accepted as queued for cid=$cid." >&2
  exit 1
fi

poll_generation "$cid"
