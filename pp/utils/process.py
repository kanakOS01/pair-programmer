import asyncio
import os
import signal
import sys


def signal_process_group(process: asyncio.subprocess.Process, sig: int = signal.SIGKILL) -> None:
    """Send sig to the process and to all its children.

    The process must start with start_new_session=True, so that its process group contains only it and its
    children. On Windows there are no process groups, so only the process itself gets the signal.
    """
    try:
        if sys.platform == "win32":
            process.kill() if sig == signal.SIGKILL else process.terminate()
        else:
            os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


async def stop_process_group(process: asyncio.subprocess.Process, grace_seconds: float = 2.0) -> None:
    """Ask the process group to stop (SIGTERM). If it is still running after grace_seconds, kill it (SIGKILL)."""
    signal_process_group(process, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
    except asyncio.TimeoutError:
        signal_process_group(process, signal.SIGKILL)
        await process.wait()
    else:
        # The main process stopped. Kill children that ignored SIGTERM.
        signal_process_group(process, signal.SIGKILL)
