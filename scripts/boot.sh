#!/usr/bin/env bash
set -e

choose_torch_wheel() {
  if ! command -v nvidia-smi &>/dev/null; then
    echo "cpu" ; return
  fi

  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
  # keep only major.minor → 8.6, 9.0, 12.0 …
  MAJOR=${CC%%.*}
  if   (( MAJOR <= 8 ));  then echo "cu118"
  elif (( MAJOR == 9 )); then echo "cu121"
  elif (( MAJOR >=12 )); then echo "cpu"   # no wheels yet
  else                     echo "cu121"
  fi
}

CUDA_TAG=$(choose_torch_wheel)
echo ">>> [boot] Selected wheel: $CUDA_TAG"

if [[ "$CUDA_TAG" == "cpu" ]]; then
  pip install --no-cache-dir torch==2.3.0+cpu torchvision==0.18.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu
  export USE_CUDA=false
else
  pip install --no-cache-dir torch==2.3.0+$CUDA_TAG torchvision==0.18.0+$CUDA_TAG \
        --index-url https://download.pytorch.org/whl/$CUDA_TAG
fi

# Install accelerate and bitsandbytes (if desired) after torch so they pick correct backend
pip install --no-cache-dir accelerate==1.6.0 \
                       --extra-index-url https://download.pytorch.org/whl/$CUDA_TAG || true

# Show torch info
python - <<'PY'
import torch
print("Torch version:", torch.__version__, "CUDA available:", torch.cuda.is_available())
PY

exec python -m neurosync.server.app