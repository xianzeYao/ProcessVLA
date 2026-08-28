#!/usr/bin/env bash

# Track child processes by PID and Linux start time so cleanup never signals a
# different process after numeric PID reuse.
declare -gA OWNED_PROCESS_START_TIMES=()

_owned_process_start_time() {
  local pid="$1" stat remainder
  [[ -r "/proc/${pid}/stat" ]] || return 1
  IFS= read -r stat < "/proc/${pid}/stat" || return 1
  remainder="${stat##*) }"
  read -r -a fields <<< "${remainder}"
  ((${#fields[@]} >= 20)) || return 1
  printf '%s\n' "${fields[19]}"
}

remember_owned_process() {
  local pid="$1" start_time
  start_time="$(_owned_process_start_time "${pid}")" || return 1
  OWNED_PROCESS_START_TIMES["${pid}"]="${start_time}"
}

forget_owned_process() {
  local pid="$1"
  unset 'OWNED_PROCESS_START_TIMES['"${pid}"']'
}

terminate_owned_process() {
  local pid="$1" expected_start current_start
  expected_start="${OWNED_PROCESS_START_TIMES[${pid}]-}"
  [[ -n "${expected_start}" ]] || return 0
  current_start="$(_owned_process_start_time "${pid}")" || {
    forget_owned_process "${pid}"
    return 0
  }
  if [[ "${current_start}" == "${expected_start}" ]]; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
  forget_owned_process "${pid}"
}

cleanup_owned_processes() {
  local pid
  for pid in "${!OWNED_PROCESS_START_TIMES[@]}"; do
    terminate_owned_process "${pid}"
  done
}
