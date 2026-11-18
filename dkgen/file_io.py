"""File replacement that tolerates temporary Windows or synced-folder locks."""

import math
import os
import time


def replace_with_retry(source, destination, *, timeout=10.0):
    """Atomically replace a file, retrying permission errors for a bounded time.

    The caller must close the completed source file first. Keep the old
    destination in place until replacement succeeds; never unlink it first.
    Other I/O errors propagate immediately, and persistent permission errors
    still fail the save rather than silently skipping it.
    """
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("Replacement timeout must be finite and nonnegative")
    deadline = time.monotonic() + timeout
    delay = 0.05
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(2 * delay, 1.0)
