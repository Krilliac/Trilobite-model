"""trilobite — interactive terminal REPL for the local self-improving coding assistant.

Boots straight into the real learning loop (server.trilobite), the way `claude`
drops you into an interactive session. Slash-commands control trace/strict mode,
teach outcomes back, and surface stats/lessons. Stdlib only + server/memory_store.
"""
import sys

import server
import memory_store

BANNER = """trilobite - fully local self-improving coder
type /help for commands, or just start typing to ask trilobite something.
"""

HELP = """commands:
  /help              show this help
  /trace [on|off]    toggle trace mode (bare = on); shows retrieval + prompt
  /strict [on|off]   toggle strict mode (bare = on); pins to the trilobite alias
  /stats             show trilobite's learning stats
  /lessons           show the 10 most recent distilled lessons
  /pass, /good       record the last answer as tests_passed
  /fail, /bad        record the last answer as failed
  /exit, /quit, /q   leave
"""


def _strip_footer(text):
    idx = text.find(server.FOOTER_PREFIX)
    if idx == -1:
        return text
    return text[:idx]


def _print_lessons():
    conn = server._open_db()
    try:
        lessons = memory_store.recent_lessons(conn, 10)
    finally:
        conn.close()
    if not lessons:
        print("(no lessons yet)")
        return
    for l in lessons:
        print("- %s" % l["text"])


def _on_off(arg, current):
    arg = (arg or "").strip().lower()
    if arg in ("", "on"):
        return True
    if arg == "off":
        return False
    print("usage: on|off (bare = on)")
    return current


def main():
    trace = False
    strict = None  # None = env default
    last_iid = None

    print(BANNER)

    while True:
        try:
            line = input("trilobite> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/help":
                print(HELP)
            elif cmd == "/trace":
                trace = _on_off(arg, trace)
                print("trace: %s" % ("on" if trace else "off"))
            elif cmd == "/strict":
                strict = _on_off(arg, strict)
                print("strict: %s" % ("on" if strict else "off"))
            elif cmd == "/stats":
                print(server.trilobite_stats())
            elif cmd == "/lessons":
                _print_lessons()
            elif cmd in ("/pass", "/good"):
                if last_iid:
                    print(server.record_outcome(last_iid, "tests_passed"))
                else:
                    print("(nothing to record yet)")
            elif cmd in ("/fail", "/bad"):
                if last_iid:
                    print(server.record_outcome(last_iid, "failed"))
                else:
                    print("(nothing to record yet)")
            elif cmd in ("/exit", "/quit", "/q"):
                break
            else:
                print("unknown command %s — try /help" % cmd)
            continue

        out = server.trilobite(line, trace=trace, strict=strict)
        if out.startswith("ERROR"):
            print(out)
            continue

        last_iid = server.parse_interaction_id(out)
        cleaned = _strip_footer(out)
        print(cleaned)
        if last_iid:
            print("(/pass or /fail to teach trilobite)")


if __name__ == "__main__":
    main()
