"""Run `sibyl init` with retries around transient TLS failures.

Why this exists: sibyl-memory-cli 0.4.0 crashes the whole 30-minute activation
window on a single bad TLS record. Its polling loop catches only HttpError:

    except HttpError as e:
        if e.status in (404, 503, 0):
            pass                      # transient, keep polling

and `http_request` maps urllib.error.URLError to HttpError(0). But an
ssl.SSLError raised inside getresponse() is not a URLError, so it escapes
unwrapped and kills the process. Observed twice on this machine:

    ssl.SSLError: [SSL: SSLV3_ALERT_BAD_RECORD_MAC] sslv3 alert bad record mac

This wraps `http_request` so TLS and socket errors are retried a few times and
then reported as HttpError(0) - the transient status their loop already knows
how to survive. Nothing about the authentication flow is reimplemented: the
browser handoff, the pairing code and the credential write are all still
theirs. Only the transport is made to stop giving up.

Run it the same way you would the real command:

    python scripts/sibyl_init_resilient.py
    python scripts/sibyl_init_resilient.py --force
"""

from __future__ import annotations

import argparse
import os
import ssl
import sys
import time

try:
    from sibyl_memory_cli import cli
except ImportError:  # pragma: no cover
    sys.exit("sibyl-memory-cli is not installed:  pip install sibyl-memory-cli")

#: Attempts per call before reporting a transient failure upstream.
RETRIES = 4
BACKOFF_SECONDS = 1.5

_original = cli.http_request


def resilient_http_request(method: str, path: str, **kwargs):
    """Retry transport-level failures; let real HTTP errors through untouched.

    An HttpError means the server answered - a 404 or a 401 is information and
    must not be retried away. Only connection-level faults are retried.
    """
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            return _original(method, path, **kwargs)
        except cli.HttpError:
            raise
        except (ssl.SSLError, OSError, ConnectionError) as exc:
            last = exc
            if attempt < RETRIES - 1:
                time.sleep(BACKOFF_SECONDS * (attempt + 1))

    # Report as the transient status their polling loop already handles, so a
    # rough patch of network costs a few seconds rather than the whole window.
    raise cli.HttpError(0, {"error": f"transport failure: {last}"}, path)


def patch_fchmod() -> bool:
    """Make credential writing possible on Windows.

    ``write_credentials_atomic`` calls ``os.fchmod(fd, 0o600)`` on the temp
    file before writing. ``os.fchmod`` is POSIX-only and simply does not exist
    on Windows, so the call raises AttributeError *after* the server has
    already returned credentials - the bind succeeds and the save crashes.
    With cli 0.4.0 no Windows user can complete activation by any path.

    Windows has no POSIX mode bits: NTFS uses ACLs, and Python's os.chmod
    there only toggles the read-only flag. The file lands under the user's
    own profile directory, which is already ACL-restricted to that user and
    administrators, so skipping the call loses nothing on this platform. It
    is what the CLI should do itself.
    """
    if hasattr(os, "fchmod"):
        return False

    def _noop_fchmod(fd: int, mode: int) -> None:  # pragma: no cover - Windows
        return None

    os.fchmod = _noop_fchmod  # type: ignore[attr-defined]
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="sibyl init, with TLS retries")
    ap.add_argument(
        "--force", action="store_true", help="re-activate even if credentials exist"
    )
    # cmd_init reads both of these off the namespace; the real parser supplies
    # `credentials` as a global option, so it has to be provided here too.
    ap.add_argument("--credentials", default=str(cli.DEFAULT_CRED_PATH))
    args = ap.parse_args()

    cli.http_request = resilient_http_request
    print(f"[patched] retrying transport errors up to {RETRIES}x")
    if patch_fchmod():
        print("[patched] os.fchmod shimmed (Windows) so credentials can be saved")
    print()
    return cli.cmd_init(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
