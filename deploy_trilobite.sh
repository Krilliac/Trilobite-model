#!/usr/bin/env bash
# deploy_trilobite.sh — stand up the trilobite MODEL on this Ubuntu box.
# Run ON THE SERVER (as root):  bash deploy_trilobite.sh
# This installs Ollama, picks a model that fits the box's RAM, pulls it,
# and creates the self-aware `trilobite` alias. (The full self-improving
# Python system is a second step — see the note at the end.)
set -euo pipefail

echo "== 1/4 Ollama =="
if ! command -v ollama >/dev/null 2>&1; then
  # Download the official installer to a file first so it CAN be inspected,
  # rather than piping a remote script straight into a root shell.
  INSTALLER="$(mktemp)"
  curl -fsSL https://ollama.com/install.sh -o "$INSTALLER"
  echo "Ollama installer downloaded to $INSTALLER (review it if you like), running it..."
  sh "$INSTALLER"
  rm -f "$INSTALLER"
fi
# make sure the server is up
(systemctl start ollama 2>/dev/null || (nohup ollama serve >/var/log/ollama.log 2>&1 &)) || true
sleep 3

echo "== 2/4 pick model by RAM =="
RAM_GB=$(free -g | awk '/Mem:/{print $2}')
if   [ "${RAM_GB:-0}" -ge 8 ]; then BASE="qwen2.5-coder:7b"
elif [ "${RAM_GB:-0}" -ge 4 ]; then BASE="qwen2.5-coder:3b"
else                                BASE="qwen2.5-coder:1.5b"
fi
echo "detected ${RAM_GB}GB RAM -> base model: $BASE"

echo "== 3/4 pull models (this downloads a few GB) =="
ollama pull "$BASE"
ollama pull nomic-embed-text

echo "== 4/4 create the self-aware trilobite alias =="
MF="$(mktemp)"
cat > "$MF" <<EOF
FROM $BASE
PARAMETER temperature 0.2
SYSTEM """You are trilobite, a self-improving coding assistant that runs entirely locally on this machine through Ollama. There is no external server and no cloud, all inference happens here, privately. You are built on a Qwen2.5-Coder base, wrapped by a local system that gives you a growing memory of short 'lessons' distilled from past coding work; relevant lessons are retrieved and added to new tasks, and solutions that pass real tests are recorded so their lessons get reused. That is how you improve over time.

Be direct, honest, and concrete. Never fabricate capabilities, tools, or configuration you do not have: you have no web search, no web fetch, and no toggleable feature flags, do not invent JSON like that. When asked about yourself, describe what you actually are (above). You cannot read your own neural internals, so do not claim to, but do not fall back on canned 'as an AI language model I cannot' refusals either; just answer plainly and usefully. Prefer correct, working code and keep answers concise."""
EOF
ollama create trilobite -f "$MF"
rm -f "$MF"

echo ""
echo "DONE. trilobite is live on this box."
echo "  Chat:        ollama run trilobite"
echo "  HTTP API:    curl http://127.0.0.1:11434/api/chat -d '{\"model\":\"trilobite\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false}'"
echo ""
echo "NEXT (full self-improving system): copy the trilobite repo here and run its"
echo "server/REPL/proxy — that adds retrieval, capture, /train, trace, and the"
echo "OpenAI-compatible proxy. Ask Claude to help set up the code transfer."
