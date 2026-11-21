#!/usr/bin/env bash
# Shared helpers for ORPO LlamaFactory train resume after interrupted runs.
# Source from sweep scripts; do not execute directly.

orpo_latest_checkpoint() {
  local out_dir="$1"
  local latest="" ckpt step max_step=-1
  shopt -s nullglob
  for ckpt in "$out_dir"/checkpoint-*; do
    [[ -d "$ckpt" ]] || continue
    step="${ckpt##*/checkpoint-}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    if (( step > max_step )) && [[ -f "$ckpt/adapter_model.safetensors" ]]; then
      max_step=$step
      latest="$ckpt"
    fi
  done
  shopt -u nullglob
  printf '%s' "$latest"
}

# Patch yaml for train: resume from latest checkpoint-* if present, else fresh run.
# Prints "resume" or "fresh".
orpo_patch_yaml_resume() {
  local yaml="$1" out_dir="$2"
  local ckpt
  [[ -f "$yaml" ]] || return 1
  ckpt="$(orpo_latest_checkpoint "$out_dir")"
  if [[ -n "$ckpt" ]]; then
    sed -i \
      -e 's/^overwrite_output_dir:.*/overwrite_output_dir: false/' \
      -e "s|^resume_from_checkpoint:.*|resume_from_checkpoint: ${ckpt}|" \
      "$yaml"
    echo "resume"
  else
    sed -i \
      -e 's/^overwrite_output_dir:.*/overwrite_output_dir: true/' \
      -e 's/^resume_from_checkpoint:.*/resume_from_checkpoint: null/' \
      "$yaml"
    echo "fresh"
  fi
}
#!/usr/bin/env bash
# Shared helpers for ORPO LlamaFactory train resume after interrupted runs.
# Source from sweep scripts; do not execute directly.

orpo_latest_checkpoint() {
  local out_dir="$1"
  local latest="" ckpt step max_step=-1
  shopt -s nullglob
  for ckpt in "$out_dir"/checkpoint-*; do
    [[ -d "$ckpt" ]] || continue
    step="${ckpt##*/checkpoint-}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    if (( step > max_step )) && [[ -f "$ckpt/adapter_model.safetensors" ]]; then
      max_step=$step
      latest="$ckpt"
    fi
  done
  shopt -u nullglob
  printf '%s' "$latest"
}

# Patch yaml for train: resume from latest checkpoint-* if present, else fresh run.
# Prints "resume" or "fresh".
orpo_patch_yaml_resume() {
  local yaml="$1" out_dir="$2"
  local ckpt
  [[ -f "$yaml" ]] || return 1
  ckpt="$(orpo_latest_checkpoint "$out_dir")"
  if [[ -n "$ckpt" ]]; then
    sed -i \
      -e 's/^overwrite_output_dir:.*/overwrite_output_dir: false/' \
      -e "s|^resume_from_checkpoint:.*|resume_from_checkpoint: ${ckpt}|" \
      "$yaml"
    echo "resume"
  else
    sed -i \
      -e 's/^overwrite_output_dir:.*/overwrite_output_dir: true/' \
      -e 's/^resume_from_checkpoint:.*/resume_from_checkpoint: null/' \
      "$yaml"
    echo "fresh"
  fi
}
