"""trilobite_client — standalone thin remote client for a hosted trilobite.

Drop this single file on any PC with Python (stdlib only — no server/memory_store/
mcp/ollama imports) to talk to a trilobite instance hosted elsewhere (e.g. a VPS
running trilobite_serve.py with TRILOBITE_HOST=0.0.0.0).

Config (env or argv):
    TRILOBITE_SERVER   e.g. http://your-vps:11435   (required)
    TRILOBITE_API_KEY  optional bearer key, if the server has auth enabled
    --server URL       argv override for TRILOBITE_SERVER
    --key K            argv override for TRILOBITE_API_KEY

Run:
    python trilobite_client.py
    python trilobite_client.py --server http://your-vps:11435 --key s3cret
"""
import json
import os
import sys
import urllib.error
import urllib.request

USAGE = """usage: trilobite_client.py [--server URL] [--key API_KEY]

Set TRILOBITE_SERVER (and optionally TRILOBITE_API_KEY) in the environment,
or pass --server/--key on the command line.

Example:
    set TRILOBITE_SERVER=http://your-vps:11435
    set TRILOBITE_API_KEY=s3cret
    python trilobite_client.py
"""


def _parse_argv(argv):
    """Parse --server/--key overrides out of argv. Returns (server, key)."""
    server = None
    key = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--server" and i + 1 < len(argv):
            server = argv[i + 1]
            i += 2
        elif arg == "--key" and i + 1 < len(argv):
            key = argv[i + 1]
            i += 2
        else:
            i += 1
    return server, key


def build_request(server, api_key, prompt):
    """Pure builder: returns (url, headers_dict, body_bytes) for a chat completion
    POST to `server`, with the given prompt as the sole user message."""
    url = server.rstrip("/") + "/v1/chat/completions"
    body = json.dumps({
        "model": "trilobite",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return url, headers, body


def send_prompt(server, api_key, prompt):
    """Send prompt to the hosted trilobite; returns the assistant's reply text,
    or raises on a network/HTTP error (caller handles presentation)."""
    url, headers, body = build_request(server, api_key, prompt)
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8")
    obj = json.loads(raw)
    return obj["choices"][0]["message"]["content"]


def resolve_config(argv):
    """Resolve (server, api_key) from argv overrides then env. Returns (server, key)."""
    argv_server, argv_key = _parse_argv(argv)
    server = argv_server or os.environ.get("TRILOBITE_SERVER", "")
    key = argv_key or os.environ.get("TRILOBITE_API_KEY", "")
    return server, key


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    server, api_key = resolve_config(argv)

    if not server:
        print(USAGE)
        return 1

    print("trilobite (remote) — connected to %s" % server)

    while True:
        try:
            line = input("trilobite> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        stripped = line.strip()
        if stripped in ("/exit", "/quit"):
            return 0
        if not stripped:
            continue

        try:
            reply = send_prompt(server, api_key, line)
            print(reply)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = str(e)
            print("HTTP %s: %s" % (e.code, err_body))
        except urllib.error.URLError as e:
            print("connection error: %s" % e)
        except Exception as e:
            print("error: %s" % e)


if __name__ == "__main__":
    sys.exit(main())
