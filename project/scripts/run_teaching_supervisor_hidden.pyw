"""Run the scheduled supervisor without a console, preserving Python/logs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    python = Path(sys.executable).with_name('python.exe')
    if not python.is_file():
        raise FileNotFoundError(python)
    logs = Path(plan['runs_root']) / 'supervisor-logs'
    logs.mkdir(parents=True, exist_ok=True)
    temporary = Path(plan['data_root']) / 'tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8', TEMP=str(temporary), TMP=str(temporary))
    with (logs / 'tick.stdout.log').open('ab') as out, (logs / 'tick.stderr.log').open('ab') as err:
        result = subprocess.run(
            [str(python), '-B', '-u', '-m', 'scripts.teaching_supervisor', '--plan', str(plan_path)],
            cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
