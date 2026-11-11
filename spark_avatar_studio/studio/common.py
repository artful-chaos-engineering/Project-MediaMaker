"""common.py - shared configuration, paths, logging and the local-or-remote runner for Spark Avatar Studio.

Every step script imports this. Two ideas live here:
  * cfg()        - studio.yaml as a dict, found relative to this file.
  * Spark        - runs a shell command on the Spark. When studio.yaml says host "local" (or we are
                   already on the Spark) it runs in-process; otherwise it runs over SSH with paramiko.
"""
from __future__ import annotations
import datetime, json, os, shlex, socket, subprocess, sys, time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent          # the spark_avatar_studio folder
CFG_PATH = ROOT / "studio.yaml"
STATE_PATH = ROOT / "state.json"


def cfg() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def say(msg: str) -> None:
    """One-line log with a timestamp; every step uses this so the notebook and menu read the same way."""
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def set_state(**kv) -> None:
    s = state(); s.update(kv); s["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    STATE_PATH.write_text(json.dumps(s, indent=2))


def require_owner(args) -> None:
    """The studio animates one face: the owner's. Capture, build and render refuse anything else."""
    owner = cfg().get("owner", "")
    if not getattr(args, "i_am_the_subject", False):
        sys.exit("refused: add --i-am-the-subject to confirm the photos/video are of you (or of someone who gave you written consent)")
    if not owner:
        sys.exit("refused: set owner in studio.yaml")


def on_spark() -> bool:
    """True when the configured Spark host is this machine: "local", our hostname, or any of our own IP addresses."""
    c = cfg()["spark"]; host = str(c["host"])
    if host in ("local", "localhost", "127.0.0.1") or socket.gethostname().split(".")[0] == host.split(".")[0]:
        return True
    try:
        mine = {"127.0.0.1"} | {i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None)}
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
        return socket.gethostbyname(host) in (mine | set(out))
    except Exception:
        return False


class Spark:
    """Run commands on the Spark, locally or over SSH, with the same call either way."""

    def __init__(self):
        c = cfg()["spark"]
        self.local = on_spark()
        # Local: run inside the installed studio (remote_home) when it exists, since that is where the models and avatars live.
        self.home = c["remote_home"] if (not self.local or Path(c["remote_home"]).is_dir()) else str(ROOT)
        # Local mode runs the code from this folder but uses the venv installed at remote_home when there is none here.
        self.venv = f"{self.home}/.venv" if (Path(self.home) / ".venv").exists() else str(ROOT / ".venv")
        self.host, self.user, self.key = c["host"], c["user"], os.path.expanduser(c.get("ssh_key", "~/.ssh/id_ed25519"))
        self._ssh = None

    # -- connection ---------------------------------------------------------------------
    def ssh(self):
        if self.local:
            return None
        if self._ssh is None:
            import paramiko
            cl = paramiko.SSHClient(); cl.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cl.connect(self.host, username=self.user, key_filename=self.key if os.path.exists(self.key) else None, password=os.environ.get("SPARK_PASSWORD"), timeout=15)
            self._ssh = cl
        return self._ssh

    def run(self, cmd: str, check: bool = True, stream: bool = True, venv: bool = True) -> str:
        """Run a shell command on the Spark inside the studio venv; stream output; return it as text."""
        pre = f"cd {self.home} && source {self.venv}/bin/activate 2>/dev/null; " if venv else f"cd {self.home} 2>/dev/null; "
        full = f"bash -lc {shlex.quote(pre + cmd)}"
        out = []
        if self.local:
            p = subprocess.Popen(full, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                out.append(line)
                if stream: print(line, end="", flush=True)
            rc = p.wait()
        else:
            _, so, _ = self.ssh().exec_command(full, get_pty=True)
            for line in so:
                out.append(line)
                if stream: print(line, end="", flush=True)
            rc = so.channel.recv_exit_status()
        text = "".join(out)
        if check and rc != 0:
            raise RuntimeError(f"command failed ({rc}) on the Spark: {cmd}\n{text[-2000:]}")
        return text

    # -- detached processes ---------------------------------------------------------------
    def detach(self, cmd: str, log: str | None = None) -> None:
        """Start a long-running process that outlives this call (the model server), without holding the SSH channel."""
        pre = f"cd {self.home} && source {self.venv}/bin/activate 2>/dev/null; "
        redir = f"> {log} 2>&1" if log else "> /dev/null 2>&1"
        full = f"{pre}setsid {cmd} {redir} < /dev/null &"
        if self.local:
            subprocess.Popen(["bash", "-lc", full], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        chan = self.ssh().get_transport().open_session()
        chan.exec_command(full)
        time.sleep(0.5)
        chan.close()

    # -- files ----------------------------------------------------------------------------
    def put(self, local: Path, remote_rel: str) -> None:
        """Copy a file to <remote_home>/<remote_rel>."""
        if self.local:
            dst = Path(self.home) / remote_rel; dst.parent.mkdir(parents=True, exist_ok=True)
            if Path(local).resolve() != dst.resolve():
                subprocess.run(["cp", str(local), str(dst)], check=True)
            return
        remote = f"{self.home}/{remote_rel}"
        made = self.__dict__.setdefault("_made", set())
        if os.path.dirname(remote) not in made:
            self.run(f"mkdir -p {os.path.dirname(remote)}", venv=False, stream=False); made.add(os.path.dirname(remote))
        if self.__dict__.get("_sftp") is None:
            self._sftp = self.ssh().open_sftp()
        self._sftp.put(str(local), remote)

    def get(self, remote_rel: str, local: Path) -> None:
        if self.local:
            if Path(f"{self.home}/{remote_rel}").resolve() != Path(local).resolve():
                Path(local).parent.mkdir(parents=True, exist_ok=True); subprocess.run(["cp", f"{self.home}/{remote_rel}", str(local)], check=True)
            return
        sftp = self.ssh().open_sftp(); Path(local).parent.mkdir(parents=True, exist_ok=True)
        sftp.get(f"{self.home}/{remote_rel}", str(local)); sftp.close()

    def sync_code(self) -> None:
        """Push the studio folder (code, config, prompts) to the Spark over SFTP (no rsync needed)."""
        if self.local:
            return
        say(f"syncing studio code to {self.user}@{self.host}:{self.home}")
        self.run(f"mkdir -p {self.home}", venv=False, stream=False)
        skip = {".venv", "models", "data", "out", "state.json", "__pycache__", ".vscode", "graft", ".git", "LatentSync", "avatars"}
        files = [f for f in ROOT.rglob("*") if f.is_file() and not (set(f.relative_to(ROOT).parts) & skip)]
        dirs = sorted({str(Path(self.home) / f.relative_to(ROOT).parent) for f in files})
        self.run("mkdir -p " + " ".join(dirs), venv=False, stream=False)
        sftp = self.ssh().open_sftp()
        for f in files:
            sftp.put(str(f), f"{self.home}/{f.relative_to(ROOT).as_posix()}")
        sftp.close()

    def voice_url(self) -> str:
        """Address of the Spark Voice Studio model server (the voice that drives the lips).

        voice.port must equal server.port in Spark Voice Studio's studio.yaml. From the Mac the Spark's firewall blocks
        the port, so it is reached through an SSH tunnel to localhost, as in the voice studio. voice.url overrides all of this.
        """
        v = cfg()["voice"]
        if v.get("url"):
            return v["url"]
        if not self.local:
            self._tunnel(int(v["port"]))
        return f"http://localhost:{v['port']}"

    def _tunnel(self, port: int) -> None:
        """Forward localhost:<port> to the Spark's server over SSH; the Spark's firewall blocks the port directly."""
        if self.__dict__.get("_tunnel_up"):
            return
        import socketserver, threading
        transport = self.ssh().get_transport()

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                ch = transport.open_channel("direct-tcpip", ("127.0.0.1", port), self.request.getpeername())

                def pump(src, dst):
                    try:
                        while True:
                            d = src.recv(65536)
                            if not d:
                                break
                            dst.sendall(d)
                    except OSError:
                        pass
                    finally:
                        for x in (src, dst):
                            try:
                                x.shutdown(2)
                            except Exception:
                                pass

                t = threading.Thread(target=pump, args=(ch, self.request), daemon=True)
                t.start()
                pump(self.request, ch)
                t.join(1)

        class Srv(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        srv = Srv(("127.0.0.1", port), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self._tunnel_up = True
