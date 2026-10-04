"""Validate the bind address before starting models or changing job state."""
import socket
import sys

from iris_pipeline.config import settings


def main():
    family = socket.AF_INET6 if ":" in settings.host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((settings.host, settings.port))
        except OSError as exc:
            print(
                f"Cannot start IRIS Pipeline on {settings.host}:{settings.port}: {exc}\n"
                "If IRIS Pipeline is already running, open its URL. To restart a "
                "background instance, run ./scripts/stop.ps1 first; for a foreground "
                "instance, press Ctrl+C in its terminal. Otherwise choose a free "
                "IRIS_PORT in .env.",
                file=sys.stderr,
            )
            return 1
    host = f"[{settings.host}]" if family == socket.AF_INET6 else settings.host
    print(f"IRIS Pipeline: http://{host}:{settings.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
