# ---------------------------------------------------------------------------
# experiments/run_sweep.py
# ---------------------------------------------------------------------------
# Sweep the LLM controller across multiple (Ollama) models, target shapes, and
# repetitions (REPETITIONS > 1 gives repeated runs of the same model/target pair,
# for statistics on stochastic LLM behavior).
#
# For each (model, target, repetition), builds one chained shell command:
#   cli.py --model ... --provider ollama --target ... --run-name <stem>
#   && analysis_plots.py --run <stem> --error-curve --output-dir viz/
# where <stem> = <model>_<target>_run_<NN> (NN = 01, 02, ...).
# The `&&` means the plot step is skipped if the run itself fails, so a
# crashed model/target combo never produces a stray or stale plot.
#
# The full command list is always written to experiments/commands.txt (one
# command per line) before running, as a record of exactly what was launched.
#
# Each run's full stdout/stderr is captured (not streamed to the terminal) and
# saved to experiments/sweep_logs/<stem>.log. The terminal only gets one
# concise status line per run, so an N-model sweep failing (e.g. Ollama server
# down) doesn't dump N full tracebacks/banners — check the per-run log file
# for full detail on any individual failure.
#
# The resulting command list is executed with a multiprocessing Pool so N
# runs go in parallel — N is a CLI flag, not hardcoded.
#
# Run from anywhere — this script resolves the project root itself, and all
# subprocess commands execute with that root as their working directory:
#   python3 experiments/run_sweep.py                  # full sweep, 1 processes at a time
#   python3 experiments/run_sweep.py --processes 8    # override parallelism
#   python3 experiments/run_sweep.py --dry-run        # print/save commands without running them
# ---------------------------------------------------------------------------

import argparse
import shlex
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# This file lives in experiments/, one level below the project root — add the
# root to sys.path so `from src...` resolves regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.paths import PROJECT_ROOT, VISUALIZATION_DIR

COMMANDS_FILE  = Path(__file__).resolve().parent / "commands.txt"
SWEEP_LOGS_DIR = Path(__file__).resolve().parent / "sweep_logs"

# ---------------------------------------------------------------------------
# SWEEP CONFIGURATION — edit these to change what gets run.
# ---------------------------------------------------------------------------

# (model, provider) pairs — provider is passed straight to cli.py's --provider flag.
MODELS = [
    # Anthropic
    ("claude-fable-5",                "anthropic"),
    ("claude-opus-4-8",               "anthropic"),
    ("claude-opus-4-7",               "anthropic"),
    ("claude-opus-4-6",               "anthropic"),
    ("claude-opus-4-5-20251101",      "anthropic"),

    ("claude-sonnet-5",                "anthropic"),
    ("claude-sonnet-4-6",              "anthropic"),
    ("claude-sonnet-4-5-20250929",     "anthropic"),

    ("claude-haiku-4-5",               "anthropic"),

    # OpenAI
    ("gpt-5.6",               "openai"),
    ("gpt-5.5",               "openai"),
    ("gpt-5.4",               "openai"),
    ("gpt-5.3-codex",         "openai"),
    ("gpt-5.2",               "openai"),
    ("gpt-5",                 "openai"),
    ("gpt-4.1",               "openai"),
    ("o3",                    "openai"),
    ("o3-pro",                "openai"),

    # Google Gemini
    ("gemini-3.5-flash",      "gemini"),
    ("gemini-3.1-pro-preview","gemini"),
    ("gemini-2.5-pro",        "gemini"),

    # Ollama
    ("qwen2.5:14b",           "ollama"),
    ("qwen2.5-coder:14b",     "ollama"),
    ("phi4",                  "ollama"),
    ("mistral-nemo",          "ollama"),
    ("gemma3:12b",            "ollama"),
    ("olmo2:13b",             "ollama"),
]

TARGETS = [
    "tip_reach",
    # "C",
    # "U",
    # "S",
    # "L",
    # "arc",
    # "sine",
]

REPETITIONS = 1  # repeats per (model, target) pair — bump for stochastic-behavior statistics

# Providers cli.py/llm_wrapper actually support. Checked upfront (validate_models())
# so a typo'd provider fails in the first second, not partway through an overnight sweep.
_KNOWN_PROVIDERS = {"anthropic", "openai", "gemini", "ollama", "fake"}


# ---------------------------------------------------------------------------
# COMMAND BUILDING
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Make a model name safe for use as a filename stem (":" is common in Ollama tags)."""
    return name.replace(":", "-").replace("/", "-")


def validate_models() -> None:
    """
    Fail fast on malformed MODELS entries, before any subprocess is launched.

    This only catches structural mistakes (wrong shape, unknown provider) --
    it can't catch a genuinely misspelled model name (e.g. "qwen2.5:14"), since
    that's only detectable by actually calling the provider. That case is still
    safe: run_one() isolates each (model, target, rep) in its own subprocess, so
    one bad model only fails its own runs -- the rest of an overnight sweep
    continues untouched (see run_one() and main()'s use of Pool.map).
    """
    problems = []
    for i, entry in enumerate(MODELS):
        if not (isinstance(entry, tuple) and len(entry) == 2):
            problems.append(f"MODELS[{i}] = {entry!r} -- expected a (model, provider) tuple")
            continue
        _, provider = entry
        if provider not in _KNOWN_PROVIDERS:
            problems.append(
                f"MODELS[{i}] = {entry!r} -- unknown provider {provider!r}, "
                f"expected one of {sorted(_KNOWN_PROVIDERS)}"
            )
    if problems:
        print("MODELS is misconfigured -- refusing to start the sweep:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)


def build_commands() -> list[tuple[str, str]]:
    """One (stem, chained shell command) pair per (model, target, repetition)."""
    python_exe = shlex.quote(sys.executable)
    viz_dir = shlex.quote(str(VISUALIZATION_DIR))

    commands = []
    for model, provider in MODELS:
        for target in TARGETS:
            for rep in range(1, REPETITIONS + 1):
                stem = f"{_sanitize(model)}_{target}_run_{rep:02d}"
                run_cmd = (
                    f"{python_exe} cli.py --provider {shlex.quote(provider)} "
                    f"--model {shlex.quote(model)} "
                    f"--target {shlex.quote(target)} --run-name {shlex.quote(stem)}"
                )
                plot_cmd = (
                    f"{python_exe} analysis_plots.py --run {shlex.quote(stem)} "
                    f"--error-curve --output-dir {viz_dir}"
                )
                commands.append((stem, f"{run_cmd} && {plot_cmd}"))
    return commands


def save_commands(commands: list[tuple[str, str]]) -> None:
    """Write the full command list to commands.txt as a record of what was launched."""
    COMMANDS_FILE.write_text("\n".join(cmd for _, cmd in commands) + "\n")
    print(f"Saved {len(commands)} commands to {COMMANDS_FILE}")


# ---------------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------------

def _last_meaningful_line(output: str) -> str:
    """Pick a short, human-relevant line out of a failed run's captured output."""
    lines = [l for l in output.splitlines() if l.strip()]
    for line in lines:
        if line.startswith("[ERROR]"):
            return line
    return lines[-1] if lines else "(no output captured)"


def run_one(item: tuple[str, str]) -> int:
    """Execute one chained command; capture its output to a log file instead of
    streaming to the terminal, and print one concise status line. Returns exit code."""
    stem, cmdline = item
    SWEEP_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SWEEP_LOGS_DIR / f"{stem}.log"

    print(f"[START] {stem}")
    result = subprocess.run(
        cmdline, shell=True, cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    log_path.write_text(output)

    if result.returncode == 0:
        print(f"[OK] {stem}")
    else:
        print(f"[FAILED exit={result.returncode}] {stem}: {_last_meaningful_line(output)}")
        print(f"         full output: {log_path}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Sweep the LLM controller across models and targets.")
    parser.add_argument(
        "--processes", type=int, default=1, metavar="N",
        help="Number of parallel processes (default: 1). Ollama-routed models share one "
             "Ollama server, so raising this doesn't parallelize those calls beyond what "
             "the server can actually serve concurrently -- cloud-provider entries "
             "(anthropic/openai/gemini) are independently rate-limited by that provider.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print and save the command list without running anything.",
    )
    args = parser.parse_args()

    validate_models()
    commands = build_commands()
    print(
        f"Built {len(commands)} commands "
        f"({len(MODELS)} models x {len(TARGETS)} targets x {REPETITIONS} repetition(s)).\n"
    )
    save_commands(commands)

    if args.dry_run:
        print()
        for _, cmdline in commands:
            print(cmdline)
        return

    start = time.time()
    with Pool(processes=args.processes) as pool:
        results = pool.map(run_one, commands)

    elapsed_min = (time.time() - start) / 60
    n_failed = sum(1 for r in results if r != 0)
    print(f"\nDone: {len(commands) - n_failed}/{len(commands)} succeeded in {elapsed_min:.1f} minutes.")
    if n_failed:
        print(f"{n_failed} command(s) failed -- see [FAILED] lines above.")


if __name__ == "__main__":
    main()
