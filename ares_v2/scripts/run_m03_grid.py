"""M03 Text Verifiers Execution Orchestrator."""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd_list):
    print(f"\n>>> Running: {' '.join(cmd_list)}", flush=True)
    t0 = time.time()
    env = dict(PYTHONPATH="ares_v2/src", **subprocess.os.environ)
    proc = subprocess.run(cmd_list, env=env)
    dur = time.time() - t0
    if proc.returncode != 0:
        print(f">>> FAILED with returncode {proc.returncode} in {dur:.1f}s", flush=True)
        sys.exit(proc.returncode)
    print(f">>> SUCCEEDED in {dur:.1f}s", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Run M03 Transformer Grid")
    parser.add_argument("--model", type=str, default="roberta-base")
    args = parser.parse_args()

    model = args.model
    # 1. CoAID (4,403 train items, binary)
    run_cmd([sys.executable, "-m", "ares.text.finetune", "--model", model, "--dataset", "coaid", "--epochs", "3", "--seed", "42"])

    # 2. Constraint (6,420 train items, binary)
    run_cmd([sys.executable, "-m", "ares.text.finetune", "--model", model, "--dataset", "constraint", "--epochs", "3", "--seed", "42"])

    # 3. LIAR (10,269 train items, binary)
    run_cmd([sys.executable, "-m", "ares.text.finetune", "--model", model, "--dataset", "liar", "--epochs", "4", "--seed", "42"])

    # 4. X-FACT (23,525 train items, 7-way)
    run_cmd([sys.executable, "-m", "ares.text.finetune", "--model", model, "--dataset", "xfact", "--epochs", "2", "--seed", "42", "--max_len", "128"])


if __name__ == "__main__":
    main()
