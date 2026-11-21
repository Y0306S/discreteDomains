"""Bounded loopback JSON transport for private ML runtime services."""

import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import urllib.error
import urllib.request


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(encoded(value)).hexdigest()


class RPCError(RuntimeError):
    pass


def call(url, payload=None, *, timeout=30, retries=2):
    if not url.startswith("http://127.0.0.1:"):
        raise ValueError("Runtime endpoints must use loopback or a loopback SSH tunnel")
    data = encoded(payload) if payload is not None else None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.load(response)
            if not value.get("ok"):
                raise RPCError(value.get("error", "Runtime request failed"))
            return value["result"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            if exc.code not in (429, 502, 503, 504) or attempt == retries:
                raise RPCError("Runtime HTTP %s: %s" % (exc.code, detail)) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError,
                http.client.HTTPException, json.JSONDecodeError) as exc:
            if attempt == retries:
                raise RPCError("Runtime transport failed; the same request may be resumed") from exc
        time.sleep(min(2 ** attempt, 8))


def server(application, port, *, max_pending=8):
    admission = threading.BoundedSemaphore(max_pending)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, value):
            body = encoded(value)
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != "/health":
                return self.reply(404, {"ok": False, "error": "Unknown path"})
            self.reply(200, {"ok": True, "result": application.health()})

        def do_POST(self):
            if self.path != "/rpc":
                return self.reply(404, {"ok": False, "error": "Unknown path"})
            if not admission.acquire(blocking=False):
                return self.reply(429, {"ok": False, "error": "Runtime queue is full"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError("Invalid request length")
                request = json.loads(self.rfile.read(length))
                result = application.handle(request)
                self.reply(200, {"ok": True, "result": result})
            except Exception as exc:
                self.reply(409, {"ok": False, "error": type(exc).__name__ + ": " + str(exc)})
            finally:
                admission.release()

    instance = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    instance.daemon_threads = True
    return instance
