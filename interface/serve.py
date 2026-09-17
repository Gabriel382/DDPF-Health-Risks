"""Serve the static DDPF Studio mockup with Python's standard library."""

from __future__ import annotations

import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8080"))


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    server = ThreadingHTTPServer((HOST, PORT), SimpleHTTPRequestHandler)
    print(f"DDPF Studio is available at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping DDPF Studio.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
