"""Serve the demo locally, like the Hugging Face Space, with the models from demo/hf_export.

    python demo/serve.py            # then open http://localhost:8000/?models=/models/

Sends the same cross-origin-isolation headers as the Space (README.md custom_headers), so
ONNX Runtime can use several threads.
"""
import argparse
import functools
import http.server
from pathlib import Path

DEMO = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        if path.startswith("/models/"):
            return str(DEMO / "hf_export" / path[len("/models/"):])
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    handler = functools.partial(Handler, directory=str(DEMO / "space"))
    Handler.extensions_map[".mjs"] = "text/javascript"
    print(f"http://localhost:{args.port}/?models=/models/")
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()
