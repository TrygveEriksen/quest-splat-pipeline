#!/usr/bin/env python3
"""Capture server: a phone uploads a video, the Quest picks the subject, and the
Quest downloads the finished gaussians.

    ./splat_server.py                    # HTTP on 0.0.0.0:8765, jobs in ./jobs

Phone:  open http://<this PC>:8765/ and upload a video of the object.
Quest:  QuestSplatViewer finds this server by UDP broadcast (port 8766), shows
        frame 0 of the newest upload, sends back a click or box, and loads the
        result when it is ready.

AnySplat starts as soon as the frames are extracted, so it runs while the
subject is being picked in the headset. A job can be re-prompted after it
finishes; only SAM 2, masking and fixup re-run.

Stdlib only, so it runs on the system python; every stage runs in its own conda
env (ANYSPLAT_ENV, SAM2_ENV), as in pipeline.sh.

LAN use only: there is no authentication, and anyone who can reach the port can
upload video and spend GPU time.
"""
import argparse
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".3gp"}
MAX_UPLOAD = 4 * 1024 ** 3
DISCOVER_REQUEST = b"QSV_DISCOVER"

GPU = threading.Lock()   # AnySplat and SAM 2 each want most of the card


def log(msg):
    print(f"[server] {msg}", flush=True)


def find_conda():
    for c in (os.environ.get("CONDA_EXE"), shutil.which("conda"),
              str(Path.home() / "anaconda3/bin/conda"),
              str(Path.home() / "miniconda3/bin/conda")):
        if c and Path(c).is_file():
            return c
    raise SystemExit("conda not found: put it on PATH or set CONDA_EXE")


class StepFailed(Exception):
    pass


class Job:
    def __init__(self, root, job_id, name):
        self.id = job_id
        self.name = name
        self.dir = root / job_id
        self.created = time.time()
        self.state = "uploaded"     # extracting, awaiting_prompt, working, done, failed
        self.message = ""
        self.error = None
        self.inference = "pending"  # running, done, failed
        self.prompt = None
        self.splats = None
        self.cond = threading.Condition()

    @property
    def frames(self):
        return sorted((self.dir / "frames").glob("*.png"))

    @property
    def result(self):
        return self.dir / "quest.ply"

    def update(self, **fields):
        with self.cond:
            for k, v in fields.items():
                setattr(self, k, v)
            self.cond.notify_all()
        (self.dir / "status.json").write_text(json.dumps(self.status()))

    def status(self):
        return {
            "id": self.id, "name": self.name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created)),
            "state": self.state, "message": self.message, "error": self.error,
            "frames": len(self.frames), "inference": self.inference,
            # 0 rather than null: Unity's JsonUtility can't read null into an int.
            "prompt": self.prompt, "splats": self.splats or 0,
            "frame_url": f"/api/jobs/{self.id}/frame.png",
            "result_url": f"/api/jobs/{self.id}/result.ply" if self.result.is_file() else None,
        }


class Pipeline:
    """Runs one job's stages with the repo's scripts, each in its conda env."""

    def __init__(self, args):
        self.args = args
        self.conda = find_conda()
        self.env = dict(os.environ)
        hf = HERE.parent.parent / "hf_cache"
        if "HF_HOME" not in self.env and hf.is_dir():
            self.env["HF_HOME"] = str(hf)

    def step(self, job, env, label, *script_args):
        cmd = [self.conda, "run", "-n", env, "--no-capture-output", "python", *map(str, script_args)]
        logfile = job.dir / "log.txt"
        with open(logfile, "a") as f:
            f.write(f"\n=== {label}: {' '.join(cmd)}\n")
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=self.env).returncode
        if rc != 0:
            tail = logfile.read_text(errors="replace").strip().splitlines()[-3:]
            raise StepFailed(f"{label} failed (exit {rc}): " + " | ".join(tail))
        return logfile

    def run(self, job, resume=False):
        a, d = self.args, job.dir
        try:
            if not resume:
                job.update(state="extracting", message="extracting frames")
                self.step(job, a.anysplat_env, "extract frames", HERE / "extract_frames.py",
                          d / job.video, "-o", d / "frames", "--max-frames", a.frames)
                job.update(state="awaiting_prompt",
                           message="pick the object in the headset (reconstruction is running)")
                job.update(inference="running")
                with GPU:
                    self.step(job, a.anysplat_env, "AnySplat", HERE / "run_anysplat.py",
                              "--frames", d / "frames", "--out", d / "raw",
                              "--anysplat-root", a.anysplat_root)
                job.update(inference="done")

            while True:   # one pass per prompt; a finished job can be re-prompted
                with job.cond:
                    job.cond.wait_for(lambda: job.prompt is not None and job.state != "done")
                    prompt = job.prompt
                self.finish(job, prompt)
        except StepFailed as e:
            if job.inference == "running":
                job.update(inference="failed")
            job.update(state="failed", message="", error=str(e))
            log(f"{job.id}: {e}")

    def finish(self, job, prompt):
        a, d = self.args, job.dir
        if "box" in prompt:
            sam_prompt = ["--box", *prompt["box"]]
        else:
            sam_prompt = ["--points", ";".join(f"{x},{y}" for x, y in prompt["points"])]

        job.update(state="working", message="segmenting the object (SAM 2)", error=None)
        masks = d / "masks"
        shutil.rmtree(masks, ignore_errors=True)
        with GPU:
            self.step(job, a.sam2_env, "SAM 2", HERE / "make_masks_sam2.py",
                      "--frames", d / "frames", "--out", masks,
                      "--checkpoint", a.sam2_checkpoint, *sam_prompt)

        with job.cond:
            job.cond.wait_for(lambda: job.inference in ("done", "failed"))
        if job.inference == "failed":
            raise StepFailed("AnySplat failed earlier")

        job.update(message="keeping the gaussians inside the masks")
        self.step(job, a.anysplat_env, "mask splats", HERE / "mask_splats.py",
                  "--ply", d / "raw/gaussians.ply", "--poses", d / "raw/poses.npz",
                  "--masks", masks, "-o", d / "masked.ply")

        # Stand it on the surface it was filmed on, origin at its base, so the
        # headset can set it down on a real table; centre it if that fails.
        job.update(message="standing it upright on its surface")
        try:
            self.step(job, a.anysplat_env, "level", HERE / "level_ply.py",
                      "--raw", d / "raw/gaussians.ply", "--subject", d / "masked.ply",
                      "--poses", d / "raw/poses.npz", "-o", d / "leveled.ply")
            fixup_in, recenter = d / "leveled.ply", []
        except StepFailed:
            fixup_in, recenter = d / "masked.ply", ["--recenter"]

        job.update(message="fixing opacity and scale")
        # --axes none: the Quest app tells UnitySplats the file is RDF (see pipeline.sh).
        logfile = self.step(job, a.anysplat_env, "fixup", HERE / "fixup_ply.py",
                            fixup_in, "-o", job.result,
                            "--axes", "none", "--scale", a.scale, *recenter)
        counts = re.findall(r"wrote ([\d,]+) splats", logfile.read_text(errors="replace"))
        splats = int(counts[-1].replace(",", "")) if counts else None
        job.update(state="done", message="ready", splats=splats)
        log(f"{job.id}: done, {splats} splats")


class Server:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.jobs).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.pipeline = Pipeline(args)
        self.jobs = {}
        self.lock = threading.Lock()
        self.reload()

    def start(self, job, resume=False):
        threading.Thread(target=self.pipeline.run, args=(job, resume), daemon=True).start()

    def reload(self):
        """Pick up jobs from an earlier run, so the headset can re-download or
        re-prompt them; anything interrupted mid-inference is marked failed."""
        for d in sorted(self.root.iterdir()):
            st = d / "status.json"
            if not st.is_file():
                continue
            s = json.loads(st.read_text())
            job = Job(self.root, s["id"], s["name"])
            job.created = time.mktime(time.strptime(s["created"], "%Y-%m-%d %H:%M:%S"))
            job.video = next((p.name for p in d.glob("video.*")), "video.mp4")
            job.prompt, job.splats = s.get("prompt"), s.get("splats")
            if (d / "raw/gaussians.ply").is_file():
                job.inference = "done"
                if job.result.is_file():
                    job.update(state="done", message="ready")
                else:
                    job.update(state="awaiting_prompt", message="pick the object in the headset",
                               prompt=None)
                self.start(job, resume=True)
            else:
                job.update(state="failed", error="interrupted by a server restart")
            self.jobs[job.id] = job
        if self.jobs:
            log(f"reloaded {len(self.jobs)} job(s) from {self.root}")

    def create(self, name, ext):
        job_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        job = Job(self.root, job_id, name)
        job.dir.mkdir(parents=True)
        job.video = "video" + ext
        with self.lock:
            self.jobs[job_id] = job
        return job

    def listing(self):
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)
        return [j.status() for j in jobs]


def make_handler(server):
    upload_page = (HERE / "splat_server_upload.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if not self.path.startswith("/api/jobs/") or self.command != "GET":
                log(f"{self.client_address[0]} {fmt % args}")

        def send(self, code, body=b"", ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if code >= 400:
                # An error may leave an upload body unread; don't reuse the
                # connection, or those bytes would be parsed as the next request.
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)

        def error(self, code, msg):
            self.send(code, {"error": msg})

        def send_file(self, path, ctype):
            size = path.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile, 1 << 20)

        def job(self, job_id):
            job = server.jobs.get(job_id)
            if job is None:
                self.error(HTTPStatus.NOT_FOUND, f"no job {job_id}")
            return job

        def do_GET(self):
            parts = urlparse(self.path).path.strip("/").split("/")
            if parts == [""]:
                return self.send(HTTPStatus.OK, upload_page, "text/html; charset=utf-8")
            if parts == ["api", "jobs"]:
                return self.send(HTTPStatus.OK, server.listing())
            if len(parts) >= 3 and parts[:2] == ["api", "jobs"]:
                job = self.job(parts[2])
                if job is None:
                    return
                if len(parts) == 3:
                    return self.send(HTTPStatus.OK, job.status())
                if parts[3:] == ["frame.png"]:
                    frames = job.frames
                    if not frames:
                        return self.error(HTTPStatus.CONFLICT, "frames not extracted yet")
                    return self.send_file(frames[0], "image/png")
                if parts[3:] == ["result.ply"]:
                    if not job.result.is_file():
                        return self.error(HTTPStatus.CONFLICT, "no result yet")
                    return self.send_file(job.result, "application/octet-stream")
            self.error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self):
            url = urlparse(self.path)
            parts = url.path.strip("/").split("/")
            length = int(self.headers.get("Content-Length") or 0)

            if parts == ["api", "jobs"]:
                name = (parse_qs(url.query).get("name") or ["video.mp4"])[0]
                ext = Path(name).suffix.lower()
                if ext not in VIDEO_EXTS:
                    return self.error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                                      f"expected a video ({', '.join(sorted(VIDEO_EXTS))})")
                if not 0 < length <= MAX_UPLOAD:
                    return self.error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                                      "empty or over 4 GB")
                job = server.create(Path(name).name[:120], ext)
                with open(job.dir / job.video, "wb") as f:
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(remaining, 1 << 20))
                        if not chunk:
                            break
                        f.write(chunk)
                        remaining -= len(chunk)
                if remaining:
                    job.update(state="failed", error="upload interrupted")
                    return self.error(HTTPStatus.BAD_REQUEST, "upload interrupted")
                job.update(message="queued")
                log(f"{job.id}: received {name} ({length / 1e6:.1f} MB)")
                server.start(job)
                return self.send(HTTPStatus.CREATED, job.status())

            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "prompt":
                job = self.job(parts[2])
                if job is None:
                    return
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    prompt = parse_prompt(body)
                except (ValueError, TypeError, KeyError) as e:
                    return self.error(HTTPStatus.BAD_REQUEST, f"bad prompt: {e}")
                with job.cond:
                    if job.state not in ("awaiting_prompt", "done"):
                        return self.error(HTTPStatus.CONFLICT, f"job is {job.state}")
                    job.prompt = prompt
                    job.state = "working"
                    job.message = ("waiting for the reconstruction to finish"
                                   if job.inference == "running" else "queued")
                    job.cond.notify_all()
                job.update()
                log(f"{job.id}: prompt {prompt}")
                return self.send(HTTPStatus.ACCEPTED, job.status())

            self.error(HTTPStatus.NOT_FOUND, "not found")

    return Handler


def parse_prompt(body):
    """{"box": [x0,y0,x1,y1]} or {"points": [[x,y], ...]}, frame-0 pixels."""
    if "box" in body:
        box = [float(v) for v in body["box"]]
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("box must be [x0, y0, x1, y1] with x1 > x0, y1 > y0")
        return {"box": [round(v, 1) for v in box]}
    points = [[round(float(x), 1), round(float(y), 1)] for x, y in body["points"]]
    if not points:
        raise ValueError("no points")
    return {"points": points}


def discovery(port, http_port):
    """Answer the headset's UDP broadcast with the HTTP port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    while True:
        data, addr = sock.recvfrom(256)
        if data.strip() == DISCOVER_REQUEST:
            sock.sendto(f"QSV_SERVER {http_port}".encode(), addr)
            log(f"discovered by {addr[0]}")


def lan_addresses():
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "scope", "global"],
                             capture_output=True, text=True).stdout
        return [l.split()[3].split("/")[0] for l in out.splitlines()
                if not l.split()[1].startswith(("docker", "br-", "veth"))]
    except OSError:
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--discovery-port", type=int, default=8766)
    ap.add_argument("--jobs", default="jobs", help="where uploads and results go")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--anysplat-root", default=str(HERE.parent.parent / "AnySplat"))
    ap.add_argument("--anysplat-env", default=os.environ.get("ANYSPLAT_ENV", "anysplat"))
    ap.add_argument("--sam2-env", default=os.environ.get("SAM2_ENV", "sam2"))
    ap.add_argument("--sam2-checkpoint",
                    default=str(HERE.parent.parent / "sam2/checkpoints/sam2.1_hiera_large.pt"))
    args = ap.parse_args()

    server = Server(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server))
    httpd.daemon_threads = True
    threading.Thread(target=discovery, args=(args.discovery_port, args.port), daemon=True).start()

    log(f"jobs in {server.root}")
    for ip in lan_addresses():
        log(f"phone upload page: http://{ip}:{args.port}/")
    log(f"headset discovery on UDP {args.discovery_port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
