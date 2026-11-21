#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${OPENAI_BASE_URL:-http://localhost:8080/v1}"
MODELS_URL="${BASE_URL%/}/models"
CHAT_URL="${BASE_URL%/}/chat/completions"

echo "=== Llama.cpp Router Mode Preflight ==="
echo "Base URL: ${BASE_URL}"

echo
echo "1) Check if anything is listening on port 8080 (inside this Linux namespace)..."
if command -v ss >/dev/null 2>&1; then
  ss -ltnp | awk 'NR==1 || /:8080/' || echo "  -> no listener on :8080 (from this namespace)"
else
  netstat -tlnp 2>/dev/null | awk 'NR==1 || /:8080/' || echo "  -> no listener on :8080 (from this namespace)"
fi

echo
echo "2) Try calling ${MODELS_URL} (3s timeout)..."
if command -v curl >/dev/null 2>&1; then
  if curl -sS --max-time 3 "${MODELS_URL}" | sed 's/\"id\":/\\n  \"id\":/g' | head -60; then
    echo
    echo "  -> /models reachable. Check that model IDs match your registry/env overrides."
  else
    echo "  -> /models request failed (connection refused / no JSON / timeout)."
    echo "     If using Windows host from WSL, set:"
    echo "     export OPENAI_BASE_URL=\"http://\$(ip route | awk '/default/ {print \$3}'):8080/v1\""
    echo "     Then retry this script."
  fi
else
  echo "  -> curl not installed; install curl to run this check."
fi

echo
echo "3) Show local model filename overrides used by this project (if set):"
echo "  LOCAL_MODEL_LLAMA_3_2_3B  = ${LOCAL_MODEL_LLAMA_3_2_3B:-<not set>}"
echo "  LOCAL_MODEL_LLAMA_3_2_11B = ${LOCAL_MODEL_LLAMA_3_2_11B:-<not set>}"
echo "  LOCAL_MODEL_QWEN_3_5_4B   = ${LOCAL_MODEL_QWEN_3_5_4B:-<not set>}"
echo "  LOCAL_MODEL_QWEN_3_5_9B   = ${LOCAL_MODEL_QWEN_3_5_9B:-<not set>}"

echo
echo "4) (Optional) Test a small chat completion against one local model."
echo "   This will only work if llama-server is running and the model file exists."
TEST_MODEL="${LOCAL_MODEL_LLAMA_3_2_11B:-Llama-3.2-11B-Vision-Instruct.Q4_K_M}"
read -rp "   Run quick chat-completion test against '${TEST_MODEL}'? [y/N] " ans
if [[ "${ans:-N}" =~ ^[Yy]$ ]]; then
  if command -v curl >/dev/null 2>&1; then
    echo
    echo "   -> Sending test request..."
    curl -sS --max-time 20 -X POST "${CHAT_URL}" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"${TEST_MODEL}\",
        \"messages\": [{\"role\":\"user\",\"content\":\"Say hi\"}],
        \"max_tokens\": 10
      }" || true
    echo
    echo "   -> If response is empty or timed out, check:"
    echo "      (a) BASE_URL is correct from WSL,"
    echo "      (b) model ID exists in ${MODELS_URL},"
    echo "      (c) router server process is still running on host."
    echo
  else
    echo "   -> curl not installed; skipping test request."
  fi
else
  echo "   -> Skipping test request."
fi

echo
echo "=== Preflight complete ==="

