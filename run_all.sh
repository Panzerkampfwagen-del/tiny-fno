#!/usr/bin/env bash
# tinyfno task runner.
#
#   ./run_all.sh build        pre-compile the CUDA kernel (one-time, a few min;
#                             do this first so later commands aren't blocked)
#   ./run_all.sh verify       tests + evaluate + benchmark + profile (uses
#                             existing checkpoints, no retraining; ~1-2 min)
#   ./run_all.sh data         generate Burgers (both variants) and NS datasets
#   ./run_all.sh train        train all three models
#   ./run_all.sh reproduce    data + train + verify, from scratch (~15 min)
#   ./run_all.sh clean        remove generated data/checkpoints/plots
#
# Override the interpreter or toolkit if your paths differ:
#   PY=... CUDA_HOME=... ./run_all.sh verify

set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/home/aryan/anaconda3/envs/qiskit_clean/bin/python}
export CUDA_HOME=${CUDA_HOME:-/home/aryan/anaconda3/envs/tinyinfer}

hr() { printf '\n=== %s ===\n' "$1"; }

build() {
  hr "pre-build CUDA kernel"
  $PY -c "from kernels.spectral_mm import _load_ext; _load_ext(); print('[build] kernel ready')"
}

data() {
  hr "Burgers nu=0.1 (headline)"; $PY -m data.burgers       --config configs/burgers.yaml --plot
  hr "Burgers nu=0.01 (hard)";    $PY -m data.burgers       --config configs/burgers_nu0.01.yaml
  hr "Navier-Stokes 2D";          $PY -m data.navier_stokes --config configs/navier_stokes.yaml
}

train() {
  hr "train Burgers nu=0.1";  $PY train.py --config configs/burgers.yaml
  hr "train Burgers nu=0.01"; $PY train.py --config configs/burgers_nu0.01.yaml
  hr "train Navier-Stokes";   $PY train.py --config configs/navier_stokes.yaml
}

verify() {
  hr "pytest (incl. kernel gradcheck)"; $PY -m pytest
  hr "evaluate Burgers";   $PY evaluate.py --config configs/burgers.yaml
  hr "evaluate NS";        $PY evaluate.py --config configs/navier_stokes.yaml
  hr "benchmark kernel";   $PY benchmark.py --cin 32 --cout 32 --kmax 12
  hr "profile NS step";    $PY profiling.py --config configs/navier_stokes.yaml --batch 64
}

reproduce() { data; train; verify; }

clean() {
  rm -f results/*.h5 results/*.pt results/*.png results/*.json results/*.log
  echo "cleaned generated artifacts in results/"
}

case "${1:-}" in
  build) build ;;
  data) data ;;
  train) train ;;
  verify) verify ;;
  reproduce) reproduce ;;
  clean) clean ;;
  *) grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
