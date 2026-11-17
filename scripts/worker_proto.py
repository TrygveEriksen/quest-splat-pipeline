#!/usr/bin/env python3
"""Line protocol shared by the resident model workers (run_anysplat.py --serve,
make_masks_sam2.py --serve) and splat_server.py.

A worker loads its model once, then serves one job per request. The server and
worker talk over the worker's stdio:

  - requests  : one JSON object per line on the worker's STDIN
  - results   : one line on the worker's STDERR, prefixed with MARK, carrying a
                JSON object ({"ready": true} at startup, then {"ok": ...} per job)
  - everything else on stdout/stderr is human/library log text, which the server
    appends verbatim to the active job's log.txt

Keeping results and logs on the same (stderr) stream makes them strictly
ordered, so the server never has to race two pipes: it reads lines until it sees
a MARK line, routing the plain ones to the log.
"""
import json
import sys

MARK = "\x01QSVMSG\x01"      # str form (worker writes it); server matches the bytes


def emit(obj):
    """Send one protocol message to the server, after flushing pending logs."""
    sys.stderr.flush()
    sys.stderr.write("\n" + MARK + json.dumps(obj) + "\n")
    sys.stderr.flush()


def serve_loop(handle):
    """Signal readiness, then run handle(request_dict) once per stdin line.

    handle returns a dict (merged into the result, "ok" defaulting to True) or
    raises; either way a single result message goes back for each request.
    """
    emit({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            emit({"ok": False, "error": f"bad request: {e}"})
            continue
        try:
            result = dict(handle(req) or {})
            result.setdefault("ok", True)
        except Exception as e:
            import traceback
            print(traceback.format_exc(), file=sys.stderr)
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        emit(result)
