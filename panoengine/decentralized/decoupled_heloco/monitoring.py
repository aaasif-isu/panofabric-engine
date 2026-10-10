"""Measured coordinator resources and immutable convergence exports.

No evaluation kernels run during training. RSS includes the coordinator's Python,
model, optimizer, socket threads, and snapshot overhead; it excludes child learners.
"""
import csv
import json
from pathlib import Path
import threading
import time
import torch


def observed_tokens(folder):
    """Latest published training tokens, not an atomic cross-worker barrier."""
    total = 0
    for path in Path(folder).glob('learner_*/steps.csv'):
        latest = 0
        with path.open() as stream:
            for row in csv.DictReader(stream):
                try:
                    latest = int(row['total_tokens'])
                except (ValueError, TypeError, KeyError):
                    continue  # concurrent writer may not have completed last row
        total += latest
    return total


class Trajectory:
    def __init__(self, folder, settings, *, kind, origin=None):
        self.folder = Path(folder)
        self.enabled = settings.get('enabled', False)
        self.every = settings.get('global_every_updates' if kind == 'global' else 'learner_every_steps', 10)
        self.kind = kind
        self.origin = time.monotonic() if origin is None else origin
        self.last = None
        self.overhead_s = 0.0
        self.local = threading.local()
        if self.enabled:
            (self.folder / 'trajectory').mkdir(exist_ok=True)

    def save_from(self, step, provider, *, tokens=0, force=False, final=False):
        if not self.enabled or (not (force or final) and (self.last == step or step % self.every)):
            return
        wall, cpu = time.perf_counter(), time.thread_time()
        parameters = provider()
        duration, cpu_duration = time.perf_counter()-wall, time.thread_time()-cpu
        self.overhead_s += duration
        self.local.wall = getattr(self.local, "wall", 0.0) + duration
        self.local.cpu = getattr(self.local, "cpu", 0.0) + cpu_duration
        self.save(step, parameters, tokens=tokens, force=force, final=final)

    def save(self, step, parameters, *, tokens=0, force=False, final=False):
        if not self.enabled or (not (force or final) and (self.last == step or step % self.every)):
            return
        started = time.perf_counter()
        cpu_started = time.thread_time()
        final = final or (force and (step != 0 or tokens != 0))
        elapsed = 0.0 if step == 0 and not final else time.monotonic() - self.origin
        payload = {'parameters': {n:p.detach().cpu().clone() for n,p in parameters.items()},
                   'step':step, 'processed_tokens':tokens, 'elapsed_s':elapsed, 'kind':self.kind,
                   'snapshot_event': 'final' if final else ('initial' if step == 0 else 'interval')}
        clock_provider = getattr(self, "clock_provider", None)
        if clock_provider is not None:
            payload['syncer_step'] = clock_provider()
        # Keep the last update observation as well as the completion endpoint.
        # A final save may have the same weights/update index but later tokens.
        suffix = '-final' if final else ''
        torch.save(payload, self.folder / 'trajectory' / f'{step:08d}{suffix}.pt')
        self.last = step
        duration = time.perf_counter() - started
        self.overhead_s += duration
        self.local.wall = getattr(self.local, "wall", 0.0) + duration
        self.local.cpu = getattr(self.local, "cpu", 0.0) + time.thread_time() - cpu_started


class CentralMonitor:
    def __init__(self, folder, settings, *, replica_pids=(), replica_resources=()):
        self.folder = Path(folder)
        self.enabled = settings.get('enabled', False)
        self.origin = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.rows = []
        self.trajectory = Trajectory(folder, settings, kind='global', origin=self.origin)
        self.replica_pids = tuple(replica_pids)
        self.replica_resources = tuple(replica_resources)
        self.thread = None
        if self.enabled:
            self.stream = (self.folder / 'central_memory.csv').open('w', newline='')
            self.writer = csv.DictWriter(self.stream, fieldnames=('elapsed_s','rss_bytes','process_cpu_s'))
            self.writer.writeheader()
            self.replica_stream = (self.folder / 'syncer_memory.csv').open('w', newline='')
            self.replica_writer = csv.DictWriter(self.replica_stream, fieldnames=('elapsed_s', 'role', 'pid', 'rss_bytes', 'process_cpu_s'))
            self.replica_writer.writeheader()
            def sample():
                while True:
                    # Linux current RSS, unlike ru_maxrss which is lifetime peak.
                    import os
                    rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
                    self.writer.writerow({'elapsed_s':time.monotonic()-self.origin,
                                          'rss_bytes':rss,'process_cpu_s':time.process_time()})
                    self.stream.flush()
                    elapsed = time.monotonic()-self.origin
                    total_rss, total_cpu = rss, time.process_time()
                    self.replica_writer.writerow(dict(elapsed_s=elapsed, role='coordinator', pid=os.getpid(), rss_bytes=rss, process_cpu_s=total_cpu))
                    sampled = 0
                    for pid, resource_path in zip(self.replica_pids, self.replica_resources, strict=True):
                        try:
                            resource = json.loads(Path(resource_path).read_text())
                            replica_rss, replica_cpu = resource['rss_bytes'], resource['process_cpu_s']
                        except (OSError, ValueError, KeyError):
                            continue
                        sampled += 1
                        total_rss += replica_rss
                        total_cpu += replica_cpu
                        self.replica_writer.writerow(dict(elapsed_s=elapsed, role='replica', pid=pid, rss_bytes=replica_rss, process_cpu_s=replica_cpu))
                    if sampled == len(self.replica_pids):
                        self.replica_writer.writerow(dict(elapsed_s=elapsed, role='aggregate', pid='', rss_bytes=total_rss, process_cpu_s=total_cpu))
                    self.replica_stream.flush()
                    if self.stop.wait(settings.get('memory_sample_seconds',0.1)):
                        break
            self.thread = threading.Thread(target=sample, daemon=True)
            self.thread.start()

    def record(self, phase, wall_s, cpu_s, step):
        if self.enabled:
            with self.lock:
                self.rows.append({'phase':phase,'step':step,'wall_s':wall_s,'thread_cpu_s':cpu_s,
                                  'elapsed_s':time.monotonic()-self.origin})

    def close(self):
        if not self.enabled:
            return
        self.stop.set()
        self.thread.join()
        self.stream.close()
        self.replica_stream.close()
        with (self.folder/'central_processing.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=('phase','step','wall_s','thread_cpu_s','elapsed_s'))
            writer.writeheader();writer.writerows(self.rows)
        (self.folder/'monitoring.json').write_text(json.dumps({
            'rss_scope':'central_memory.csv: coordinator only; syncer_memory.csv: per-process and sum of coordinator plus latest CPU replica self-samples (up to 0.1s old), excludes learners; RSS sum may double-count shared pages',
            'syncer_replica_pids':list(self.replica_pids),
            'processing_scope':'outer apply/merge/correction/update and dispatch construction/enqueue; wall may include lock contention; includes replica IPC and collective waiting for sharded syncers; excludes learner-capture waiting and snapshot IO; coordinator CPU covers the control thread; replica CPU is separate; not full HTTP serialization' ,
            'token_axis':'latest published cumulative learner tokens; asynchronous observation',
            'snapshot_overhead_s':self.trajectory.overhead_s},indent=2))


def attach_baseline(server, model, monitor):
    """Snapshot at committed weights while the original server lock is held."""
    original_commit = server._commit_step_locked
    def commit(*args, **kwargs):
        result = original_commit(*args, **kwargs)
        if server._revision % monitor.trajectory.every == 0:
            monitor.trajectory.save(server._revision, dict(model.named_parameters()),
                                    tokens=observed_tokens(monitor.folder))
        return result
    if monitor.enabled:
        server._commit_step_locked = commit
        original_apply = server._apply_one
        def apply(*args, **kwargs):
            wall, cpu = time.perf_counter(), time.thread_time()
            io = getattr(monitor.trajectory.local, "wall", 0.0)
            cpu_io = getattr(monitor.trajectory.local, "cpu", 0.0)
            result = original_apply(*args, **kwargs)
            monitor.record('outer_apply', max(0,time.perf_counter()-wall-(getattr(monitor.trajectory.local,"wall",0)-io)),
                           max(0,time.thread_time()-cpu-(getattr(monitor.trajectory.local,"cpu",0)-cpu_io)), server._revision)
            return result
        server._apply_one = apply
        original_dispatch = server._build_snapshot_locked
        def dispatch(*args, **kwargs):
            wall, cpu = time.perf_counter(), time.thread_time()
            result = original_dispatch(*args, **kwargs)
            monitor.record("dispatch_snapshot", time.perf_counter()-wall, time.thread_time()-cpu, server._revision)
            return result
        server._build_snapshot_locked = dispatch
