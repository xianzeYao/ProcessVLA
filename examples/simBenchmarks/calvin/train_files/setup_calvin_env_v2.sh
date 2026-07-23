#!/usr/bin/env bash
set -euo pipefail

CALVIN_ROOT="${CALVIN_ROOT:-/root/data/yxz/benchmarks/calvin}"
CONDA_BIN="${CONDA_BIN:-/root/data/yxz/miniforge3/bin/conda}"
ENV_NAME="${CALVIN_ENV_NAME:-calvin}"

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  "$CONDA_BIN" create -y -n "$ENV_NAME" python=3.8
fi

run_in_env() {
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" "$@"
}

# The official pin is 3.18.4; this index exposes the equivalent post-release.
run_in_env python -m pip install wheel "cmake==3.18.4.post1"
run_in_env python -m pip install -e "$CALVIN_ROOT/calvin_env/tacto"
run_in_env python -m pip install -e "$CALVIN_ROOT/calvin_env"
run_in_env python -m pip install -e "$CALVIN_ROOT/calvin_models"
run_in_env python - <<'PY'
import gym, hydra, numpy, pybullet
print('CALVIN environment ready:', gym.__version__, hydra.__version__, numpy.__version__)
PY
