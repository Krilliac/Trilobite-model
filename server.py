"""
local-llm MCP server
---------------------
Bridges Claude Code to a local Ollama instance running on the RTX 4050 (6 GB VRAM).

Design goals:
  * Claude decides WHEN to offload (tools are opt-in), so VRAM is idle until used.
  * Tiered models: only one sits in VRAM at a time; short keep_alive frees it fast.
  * Zero third-party HTTP deps (stdlib urllib) -> only `mcp` is required.

Tiers (escalation ladder, cheapest first):
  LOCAL  (private, free, offline, runs on the 6 GB 4050):
    fast        -> qwen2.5:3b            (~2 GB, fully GPU-resident, snappy)
    code        -> qwen2.5-coder:7b      (~4.7 GB Q4, strong coding model)
    general     -> qwen2.5:7b-instruct   (~4.7 GB Q4, general text grunt-work)
  CLOUD  (Ollama-hosted, huge, metered; prompt leaves this machine):
    cloud-code  -> qwen3-coder:480b-cloud (frontier coding, no local VRAM cost)
    cloud-general -> gpt-oss:120b-cloud   (heavy reasoning over text)
"""

import json
import os
import urllib.request
import urllib.error

from mcp.server.fastmcp import FastMCP

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").replace("http://", "")
BASE = f"http://{OLLAMA_HOST}"
# How long a model stays in VRAM after its last call. Short = frees GPU quickly.
KEEP_ALIVE = os.environ.get("LOCAL_LLM_KEEP_ALIVE", "2m")
TIMEOUT = int(os.environ.get("LOCAL_LLM_TIMEOUT", "300"))

TIERS = {
    "fast": os.environ.get("LOCAL_LLM_FAST", "qwen2.5:3b"),
    "code": os.environ.get("LOCAL_LLM_CODE", "qwen2.5-coder:7b"),
    "general": os.environ.get("LOCAL_LLM_GENERAL", "qwen2.5:7b-instruct"),
    "cloud-code": os.environ.get("LOCAL_LLM_CLOUD_CODE", "qwen3-coder:480b-cloud"),
    "cloud-general": os.environ.get("LOCAL_LLM_CLOUD_GENERAL", "gpt-oss:120b-cloud"),
}
# Tiers whose ":...-cloud" model runs on Ollama's servers (data leaves the machine).
CLOUD_TIERS = {"cloud-code", "cloud-general"}

mcp = FastMCP("local-llm")


def _post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(path: str) -> dict:
    req = urllib.request.Request(f"{BASE}{path}")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


@mcp.tool()
def offload(
    prompt: str,
    tier: str = "fast",
    system: str = "",
    temperature: float = 0.2,
    num_predict: int = 1024,
    num_ctx: int = 4096,
) -> str:
    """Offload a self-contained subtask to a local-GPU or Ollama-cloud model instead of doing it yourself.

    Use this to save your own context/budget on bulk or mechanical work: summarizing,
    classifying, extracting, drafting boilerplate, code generation, reformatting.

    Escalation ladder — pick the CHEAPEST tier that can do the job:
      LOCAL (private, free, runs on the 6 GB 4050; loads into VRAM on call, unloads after):
        tier="fast"          -> 3B. Default. Summaries, classification, simple text, quick answers.
        tier="code"          -> 7B coder. Code gen, refactor snippets, code explanation.
        tier="general"       -> 7B instruct. Harder text reasoning, longer drafts, rewriting.
      CLOUD (frontier-size, hosted by Ollama, METERED, and the PROMPT LEAVES THIS MACHINE):
        tier="cloud-code"    -> qwen3-coder:480b. Hard/large coding the 7B can't handle well.
        tier="cloud-general" -> gpt-oss:120b. Hard reasoning / large-context text tasks.

    PRIVACY: prefer a LOCAL tier for anything touching the user's private code or data
    (e.g. their mangos-unified / Cambrian / MMORPG repos). Only use a cloud-* tier when the
    task is genuinely beyond the 7B AND the content isn't sensitive, or the user asked for it.

    Give the model a FULLY self-contained prompt (it has no memory of this conversation
    and cannot see files). Paste in any context it needs.

    Args:
        prompt: The complete, self-contained instruction + context for the model.
        tier: One of "fast", "code", "general", "cloud-code", "cloud-general".
        system: Optional system instruction to steer behavior/format.
        temperature: 0.0-1.0. Low = deterministic (default 0.2).
        num_predict: Max tokens to generate (default 1024).
        num_ctx: Context window for LOCAL tiers (default 4096). On this 6 GB card (~4.9 GB free)
            the 7B always runs ~80% GPU / ~20% CPU; the 3B fits 100% GPU. Bigger num_ctx = more
            CPU spill + system RAM. Raise only when a task needs more context. Ignored for cloud.

    Returns:
        The model's text response.
    """
    model = TIERS.get(tier)
    if model is None:
        return f"ERROR: unknown tier '{tier}'. Valid tiers: {', '.join(TIERS)}."

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    options = {"temperature": temperature, "num_predict": num_predict}
    payload = {"model": model, "messages": messages, "stream": False, "options": options}
    # keep_alive + a capped context only matter for local models held in VRAM; cloud runs remote.
    if tier not in CLOUD_TIERS:
        payload["keep_alive"] = KEEP_ALIVE
        options["num_ctx"] = num_ctx
    try:
        out = _post("/api/chat", payload)
    except urllib.error.URLError as e:
        return (
            f"ERROR contacting Ollama at {BASE}: {e}. "
            "Is the Ollama server running? (the tray app / `ollama serve`)"
        )
    msg = out.get("message", {}).get("content", "")
    return msg if msg else f"(empty response) raw={json.dumps(out)[:500]}"


@mcp.tool()
def status() -> str:
    """Report local-LLM state: which models are installed, and which are currently in VRAM.

    Use this to check whether the GPU is busy before offloading, or to confirm models pulled.
    """
    try:
        tags = _get("/api/tags").get("models", [])
        ps = _get("/api/ps").get("models", [])
    except urllib.error.URLError as e:
        return f"ERROR contacting Ollama at {BASE}: {e}"

    installed = sorted(m.get("name", "?") for m in tags)
    loaded = [
        f"{m.get('name')} (VRAM ~{round(m.get('size_vram', 0)/1e9, 1)} GB)" for m in ps
    ]
    tier_lines = [
        f"  {k}={v}" + ("  [CLOUD — leaves machine]" if k in CLOUD_TIERS else "  [local GPU]")
        for k, v in TIERS.items()
    ]
    lines = [
        f"Ollama @ {BASE}",
        "Tiers:",
        *tier_lines,
        f"Installed/registered models: {', '.join(installed) if installed else '(none)'}",
        f"In VRAM now: {', '.join(loaded) if loaded else '(none — GPU idle)'}",
        f"local keep_alive: {KEEP_ALIVE}",
    ]
    return "\n".join(lines)


@mcp.tool()
def unload(tier: str = "all") -> str:
    """Immediately free GPU VRAM by unloading a model (or all of them).

    Args:
        tier: "all" (default), or one of "fast", "code", "general".
    """
    if tier == "all":
        # Only local tiers occupy VRAM; cloud tiers run remote.
        targets = [v for k, v in TIERS.items() if k not in CLOUD_TIERS]
    elif tier in CLOUD_TIERS:
        return f"'{tier}' is a cloud tier — it uses no local VRAM, nothing to unload."
    else:
        targets = [TIERS.get(tier)]
    if None in targets:
        return f"ERROR: unknown tier '{tier}'. Valid: all, {', '.join(TIERS)}."
    freed = []
    for model in targets:
        try:
            _post("/api/generate", {"model": model, "keep_alive": 0})
            freed.append(model)
        except urllib.error.URLError:
            pass
    return f"Unload requested for: {', '.join(freed) if freed else '(none)'}."


if __name__ == "__main__":
    mcp.run()
