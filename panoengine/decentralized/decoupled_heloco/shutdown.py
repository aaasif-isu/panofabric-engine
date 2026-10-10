"""Wait for learner teardown under the existing shared training deadline."""
import time
from pathlib import Path


def wait_for_learner_shutdown(processes, deadline, folder):
    """Protocol completion can precede trainer/CUDA/distributed teardown.

    Processes maps learner ID to Popen. Poll all children so a nonzero exit
    is reported promptly even when another learner is still tearing down.
    Failure cleanup remains the calling coordinator's responsibility.
    """
    pending = dict(processes)
    next_report = time.monotonic() + 10.0
    while pending:
        for learner_id, process in list(pending.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            if returncode != 0:
                raise RuntimeError(
                    f"learner {learner_id} failed during shutdown: pid={process.pid}, "
                    f"exit_code={returncode}; inspect {Path(folder)/f'learner_{learner_id}'/'trainer.log'}"
                )
            del pending[learner_id]
        if not pending:
            return
        now = time.monotonic()
        if now >= deadline:
            details = ', '.join(f"learner {i} pid={p.pid} log={Path(folder)/f'learner_{i}'/'trainer.log'}" for i,p in pending.items())
            raise TimeoutError(f"learner shutdown exceeded --training-timeout; still running: {details}")
        if now >= next_report:
            print(f"Waiting for learner process teardown: ids={list(pending)}, remaining_run_seconds={deadline-now:.1f}", flush=True)
            next_report = now + 10.0
        time.sleep(min(.1, deadline-now))
