"""Remote AutoDL orchestration via Paramiko."""
from __future__ import annotations

import os
import sys
import time

import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 14066
USER = "root"
REMOTE_ROOT = "/root/autodl-tmp/SECI-Net"


def connect() -> paramiko.SSHClient:
    password = os.environ.get("AUTODL_SSH_PASSWORD")
    if not password:
        raise SystemExit("Set AUTODL_SSH_PASSWORD")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=password, timeout=60)
    return client


def run(client: paramiko.SSHClient, command: str, timeout: int | None = None) -> int:
    print(f"\n>>> {command[:200]}{'...' if len(command) > 200 else ''}")
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err:
        print(err, end="" if err.endswith("\n") else "\n", file=sys.stderr)
    return code


def run_script(client: paramiko.SSHClient, script: str, timeout: int | None = None) -> int:
    remote_path = "/root/autodl-tmp/_remote_cmd.sh"
    sftp = client.open_sftp()
    with sftp.file(remote_path, "w") as remote_file:
        remote_file.write(script)
    sftp.chmod(remote_path, 0o755)
    sftp.close()
    return run(client, f"bash {remote_path}", timeout=timeout)


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "setup"
    client = connect()
    try:
        if action == "setup":
            setup_script = """#!/usr/bin/env bash
set -euo pipefail
source /etc/network_turbo 2>/dev/null || true
cd /root/autodl-tmp
rm -rf SECI-Net
git clone https://github.com/ShuoMeng66/SECI-Net.git || git clone https://ghproxy.net/https://github.com/ShuoMeng66/SECI-Net.git
cd /root/autodl-tmp/SECI-Net
/root/miniconda3/bin/python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install scikit-learn datasets gdown tqdm
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"""
            code = run_script(client, setup_script, timeout=900)
            if code != 0:
                raise SystemExit(code)

        elif action == "start":
            start_script = """#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/SECI-Net
source /etc/network_turbo 2>/dev/null || true
mkdir -p logs
chmod +x scripts/autodl_run_full.sh
nohup bash scripts/autodl_run_full.sh > logs/autodl_nohup.log 2>&1 &
echo STARTED_PID=$!
"""
            code = run_script(client, start_script, timeout=120)
            if code != 0:
                raise SystemExit(code)

        elif action == "status":
            run(client, f"tail -n 30 {REMOTE_ROOT}/logs/autodl_nohup.log 2>/dev/null || tail -n 30 {REMOTE_ROOT}/logs/run_full.log 2>/dev/null || echo 'no log yet'")
            run(client, f"pgrep -af run_minimal_experiments || pgrep -af autodl_run_full || echo 'no training process'")
            run(client, f"test -f {REMOTE_ROOT}/tables/results_rollup.json && cat {REMOTE_ROOT}/tables/results_rollup.json || echo 'results not ready'")

        elif action == "fetch":
            # download via sftp
            sftp = client.open_sftp()
            local_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            os.makedirs(os.path.join(local_root, "tables"), exist_ok=True)
            remote_tables = f"{REMOTE_ROOT}/tables"
            for name in sftp.listdir(remote_tables):
                if name.endswith((".tex", ".json")):
                    remote = f"{remote_tables}/{name}"
                    local = os.path.join(local_root, "tables", name)
                    print(f"Downloading {remote} -> {local}")
                    sftp.get(remote, local)
            sftp.close()

        else:
            raise SystemExit(f"Unknown action: {action}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
