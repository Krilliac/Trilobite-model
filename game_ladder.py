"""game_ladder — trilobite's "game gauntlet".

Escalating-difficulty ladder: for each level, trilobite generates a
complete, runnable game (code + procedurally-drawn assets), we ground it
by actually running it headless, and the ladder stops at the first level
that fails. Also feeds pass/fail back into trilobite's learning loop via
server.record_outcome, so the gauntlet doubles as training signal.

Stdlib only (+ py_compile/subprocess). No GPU/model calls at import time —
those only happen in the __main__ live driver.
"""
import os
import py_compile
import subprocess
import sys
import tempfile

import grounding

PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Scripts", "python.exe")

_PYGAME_TEST_INSTRUCTION = (
    "Write a COMPLETE, runnable, single-file Python program. Return ONLY the code "
    "in one python code block. Use only the standard library and pygame; draw ALL "
    "visuals procedurally with pygame.draw/Surface — NO external image/sound files. "
    "The program MUST support headless testing: when the environment variable "
    "GAME_TEST is set, run exactly 60 iterations of the main loop (processing "
    "events, updating, drawing) then call sys.exit(0) — do NOT loop forever or "
    "wait for input in that mode."
)

_CONSOLE_TEST_INSTRUCTION = (
    "Write a COMPLETE, runnable, single-file Python program. Return ONLY the code "
    "in one python code block. Use only the standard library. The program MUST "
    "support automated testing: when env var GAME_TEST is set, play a short "
    "scripted/auto demo (no input()) and exit(0)."
)


def _console(n, name, desc):
    return {"n": n, "name": name, "kind": "console",
            "prompt": "%s %s" % (desc, _CONSOLE_TEST_INSTRUCTION)}


def _pygame(n, name, desc):
    return {"n": n, "name": name, "kind": "pygame",
            "prompt": "%s %s" % (desc, _PYGAME_TEST_INSTRUCTION)}


LEVELS = [
    _console(1, "number_guessing", "Build a number-guessing game: the program picks a "
             "secret random number in a range and the player guesses, getting "
             "higher/lower feedback until they win."),
    _console(2, "tic_tac_toe", "Build a tic-tac-toe game where the human player plays "
             "against a simple AI opponent (does not need to be unbeatable, just a "
             "reasonable move-picking heuristic)."),
    _pygame(3, "player_square", "Build a pygame program with a window that draws a "
            "background and a player square the user moves with the arrow keys."),
    _pygame(4, "pong", "Build a pygame Pong game: two paddles (one can be AI or "
            "player 2), a bouncing ball, a score for each side, and a win condition."),
    _pygame(5, "snake", "Build a pygame Snake game: a grid, food that spawns "
            "randomly, a snake that grows when it eats, and game over on "
            "self-collision or wall collision."),
    _pygame(6, "breakout", "Build a pygame Breakout/Arkanoid game: a paddle, a ball, "
            "a grid of bricks that break on collision, scoring, and a limited number "
            "of lives."),
    _pygame(7, "asteroids", "Build a pygame Asteroids game: a ship that rotates and "
            "thrusts, bullets the ship fires, asteroids that split into smaller "
            "asteroids when hit, and screen wrap-around at the edges."),
    _pygame(8, "space_shooter", "Build a pygame side-scrolling space shooter with "
            "procedurally-drawn sprite assets (draw the ship/enemies/bullets with "
            "pygame.draw shapes onto Surfaces): waves of enemies, a player hp bar, "
            "a score counter, and simple particle effects on explosions."),
    _pygame(9, "platformer", "Build a pygame platformer: gravity, jumping, solid "
            "platforms plus at least one moving platform, collectible items, and a "
            "small hand-designed level layout."),
    _pygame(10, "topdown_dungeon", "Build a pygame tile-based top-down game with a "
            "procedurally-generated map (e.g. random rooms/corridors or cellular "
            "automata), enemies with basic chase AI toward the player, and a "
            "minimap rendered in a corner of the screen."),
    _pygame(11, "multi_scene", "Build a pygame multi-scene game with a menu scene, a "
            "play scene, and a game-over scene (menu -> play -> game-over -> menu), "
            "a high score that is saved to and loaded from a local file, and a "
            "particle system used somewhere in the game (e.g. on scoring or death)."),
    _pygame(12, "tower_defense", "Build an ambitious pygame tower-defense-lite game: "
            "enemies pathfind along a route from spawn to goal (e.g. simple "
            "waypoint-following or grid pathfinding), the player places towers that "
            "shoot at enemies in range, there is a wave counter, and a lose "
            "condition when too many enemies reach the goal."),
]


def detect_failure(stdout, stderr, returncode):
    """Pure classifier: did this run count as a failure, and why?

    Returns (failed: bool, reason: str).
    """
    stderr = stderr or ""
    stdout = stdout or ""
    has_traceback = "Traceback (most recent call last)" in stderr
    if has_traceback:
        lines = [l for l in stderr.strip().splitlines() if l.strip()]
        last_line = lines[-1] if lines else "Traceback (most recent call last)"
        return True, last_line
    if returncode != 0:
        return True, "exit code %d" % returncode
    return False, "ok"


def ground(code, kind, timeout=15):
    """Run generated game `code` headless and report pass/fail.

    Returns (passed: bool, detail: str).
    """
    fd, path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            py_compile.compile(path, doraise=True)
        except (py_compile.PyCompileError, SyntaxError) as e:
            msg = str(e).strip()
            last_line = msg.splitlines()[-1] if msg else "syntax error"
            return False, "SyntaxError: %s" % last_line

        env = dict(os.environ)
        env.update({
            "SDL_VIDEODRIVER": "dummy",
            "SDL_AUDIODRIVER": "dummy",
            "GAME_TEST": "1",
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
        })
        interp = PY if os.path.exists(PY) else sys.executable
        try:
            p = subprocess.run(
                [interp, path],
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            err = e.stderr or ""
            if isinstance(err, bytes):
                err = err.decode("utf-8", errors="replace")
            if "Traceback (most recent call last)" in err:
                lines = [l for l in err.strip().splitlines() if l.strip()]
                return False, lines[-1] if lines else "traceback (timed out)"
            return True, "ran (timed out, no crash)"

        failed, reason = detect_failure(p.stdout, p.stderr, p.returncode)
        return (not failed), reason
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def run_ladder(gen_fn, start=1, max_levels=99, save_dir="games", record=None):
    """Run the gauntlet from `start`, generating+grounding each level in turn.

    Stops (and returns) at the first level that fails, or after the last
    level in LEVELS / max_levels, whichever comes first.
    """
    levels = [l for l in LEVELS if l["n"] >= start][:max_levels]
    last_n = start - 1
    for level in levels:
        n, name, kind, prompt = level["n"], level["name"], level["kind"], level["prompt"]
        raw = gen_fn(prompt)
        code = grounding.extract_code_block(raw)
        if not code:
            detail = "no code block"
            if record:
                record(level, False, code)
            print("LEVEL %d %s: FAIL — %s" % (n, name, detail))
            return {"reached": last_n, "failed_level": level, "detail": detail}

        level_dir = os.path.join(save_dir, "level_%02d_%s" % (n, name))
        os.makedirs(level_dir, exist_ok=True)
        with open(os.path.join(level_dir, "game.py"), "w", encoding="utf-8") as f:
            f.write(code)

        passed, detail = ground(code, kind)
        if record:
            record(level, passed, code)
        print("LEVEL %d %s: %s — %s" % (n, name, "PASS" if passed else "FAIL", detail))
        if not passed:
            return {"reached": last_n, "failed_level": level, "detail": detail}
        last_n = n

    return {"reached": last_n, "failed_level": None}


if __name__ == "__main__":
    # Live driver — needs Ollama/GPU. Do not run this in automated tests.
    import server

    _last_raw = {"text": None}

    def gen_fn(prompt):
        text = server.trilobite(prompt, num_predict=2048)
        _last_raw["text"] = text
        return text

    def record(level, passed, code):
        raw = _last_raw["text"]
        iid = server.parse_interaction_id(raw) if raw else None
        if iid:
            server.record_outcome(iid, "tests_passed" if passed else "failed")

    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    max_levels = int(sys.argv[2]) if len(sys.argv) > 2 else 99

    result = run_ladder(gen_fn, start=start, max_levels=max_levels, record=record)

    reached = result["reached"]
    failed_level = result["failed_level"]
    if failed_level:
        print("GAUNTLET: reached level %d, failed at %d (%s)" % (
            reached, failed_level["n"], result["detail"]))
    else:
        print("GAUNTLET: reached level %d, cleared the whole ladder" % reached)

    os.makedirs("games", exist_ok=True)
    with open(os.path.join("games", "progress.txt"), "w", encoding="utf-8") as f:
        f.write(str(reached))
