"""Persistent CPU syncer replicas with fragment-only IPC all-reduce.

The coordinator owns capture admission and ordered clocks. Each replica owns
full global/outer state and merges the contributors assigned to its rank.
Local IPC is trusted; a replica failure aborts the run (no partial recovery).
"""
import io
import multiprocessing as mp
import time
import os
import json
import tempfile
import threading
from pathlib import Path

import torch

from .fragment_manager import FragmentManager
from .optimizer import FragmentSGD, FragmentDiLoCo, FragmentHeLoCo


def _pack(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def _unpack(value):
    return torch.load(io.BytesIO(value), map_location='cpu', weights_only=False)


class _Collective:
    """Reduce to replica zero then broadcast, entirely outside coordinator."""
    def __init__(self, rank, peers, timeout):
        self.rank, self.peers, self.timeout = rank, peers, timeout

    def all_reduce(self, tensor, *, minimum=False):
        deadline = time.monotonic()+self.timeout
        def receive(connection):
            if not connection.poll(max(0, deadline-time.monotonic())):
                raise TimeoutError('syncer collective timed out')
            value = _unpack(connection.recv_bytes())
            if value.shape != tensor.shape or value.dtype != tensor.dtype:
                raise ValueError('syncer collective tensor mismatch')
            return value
        if self.rank == 0:
            for peer in self.peers:
                value = receive(peer)
                if minimum:
                    torch.minimum(tensor, value, out=tensor)
                else:
                    tensor.add_(value)
            packed = _pack(tensor)
            for peer in self.peers:
                peer.send_bytes(packed)
        else:
            self.peers[0].send_bytes(_pack(tensor))
            tensor.copy_(receive(self.peers[0]))

    def barrier(self):
        self.all_reduce(torch.zeros((), dtype=torch.int64))


def _worker(rank, peers, connection, initialization, timeout, resource_path):
    torch.set_num_threads(1)
    stopped = threading.Event()
    def sample_resources():
        while True:
            rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
            temporary = Path(resource_path + '.tmp')
            temporary.write_text(json.dumps({'rss_bytes':rss, 'process_cpu_s':time.process_time()}))
            temporary.replace(resource_path)
            if stopped.wait(.1):
                return
    sampler = threading.Thread(target=sample_resources, daemon=True)
    sampler.start()
    try:
        parameters, fragments, method, options = _unpack(initialization)
        manager = FragmentManager(parameters.items(), fragments)
        cls = {'sgd': FragmentSGD, 'diloco': FragmentDiLoCo, 'heloco': FragmentHeLoCo}[method]
        optimizer = cls(manager, parameters, **options)
        del parameters, initialization
        collective = _Collective(rank, peers, timeout)
        connection.send_bytes(_pack(('ok', None)))
        while True:
            command, payload = _unpack(connection.recv_bytes())
            if command == 'close':
                break
            if command == 'step':
                fragment_id, contributions, merge = payload
                wall, cpu = time.perf_counter(), time.process_time()
                merged = {}
                before = method == 'heloco' and optimizer.config.correction_order == 'correct_then_merge'
                # Every rank executes identical collectives, including empty
                # ranks belonging to unavailable learners (zero contribution).
                error = None
                local = {}
                try:
                    for name in manager.fragment(fragment_id).parameter_names:
                        template = optimizer._parameters[name]
                        direction = torch.zeros_like(template)
                        radius = torch.zeros((), dtype=template.dtype)
                        mode = merge
                        if mode == 'paper_rda':
                            mode = 'weighted_average' if {'embedding', 'tok_embeddings', 'embed_tokens'}.intersection(name.split('.')) else 'rda'
                        for baseline, current, weight in contributions:
                            gradient = baseline[name].float() - current[name].float()
                            if before:
                                gradient = optimizer.correct({name: gradient})[name]
                            if not bool(torch.isfinite(gradient).all()):
                                raise ValueError('nonfinite contributor gradient')
                            if mode == 'rda':
                                norm = gradient.norm()
                                radius.add_(norm, alpha=weight)
                                if bool(norm > 0):
                                    direction.add_(gradient / norm, alpha=weight)
                            else:
                                direction.add_(gradient, alpha=weight)
                        local[name] = (direction, radius, mode)
                except Exception as exc:
                    error = str(exc)
                ok = torch.tensor(0 if error else 1)
                collective.all_reduce(ok, minimum=True)
                if not ok.item():
                    raise RuntimeError(error or 'another syncer replica rejected the contribution')
                collective_bytes = 16  # validation and completion int64 scalars
                for name, (direction, radius, mode) in local.items():
                    collective.all_reduce(direction)
                    collective_bytes += direction.numel() * direction.element_size()
                    if mode == 'rda':
                        collective.all_reduce(radius)
                        collective_bytes += radius.element_size()
                        norm = direction.norm()
                        direction = direction.mul_(radius / norm) if bool(norm > torch.finfo(direction.dtype).eps) else torch.zeros_like(direction)
                    merged[name] = direction
                norm = sum(float(t.double().square().sum()) for t in merged.values()) ** 0.5
                dispatch = optimizer.step(fragment_id, merged, already_corrected=before) if method == 'heloco' else optimizer.step(fragment_id, merged)
                # Barrier ensures all replicas finished before reporting commit.
                collective.barrier()
                stats = {'rank':rank, 'wall_s':time.perf_counter()-wall,
                         'process_cpu_s':time.process_time()-cpu,
                         'collective_tensor_bytes':collective_bytes}
                connection.send_bytes(_pack(('ok', (dispatch if rank == 0 else None, norm, stats))))
            elif command in {'snapshot', 'dispatch_snapshot', 'momentum_snapshot', 'model_snapshot'}:
                result = getattr(optimizer, command)(*payload)
                connection.send_bytes(_pack(('ok', result)))
            else:
                raise ValueError('unknown syncer replica command')
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send_bytes(_pack(('error', f'replica {rank}: {exc}')))
        except (OSError, EOFError):
            pass
    finally:
        stopped.set()
        sampler.join()
        connection.close()
        for peer in peers:
            peer.close()


class ShardedOptimizer:
    """Optimizer facade; global and momentum state reside in child replicas."""
    def __init__(self, parameters, manager, method, options, shards, timeout=60.0):
        self.manager = manager
        self.shards = shards
        self.timeout = timeout
        self.method = method
        self.config = options.get('config')
        self._connections, self.processes = [], []
        self._closed = False
        self.stats = []
        self.progress = None
        self._resources = tempfile.TemporaryDirectory(prefix="syncer-resources-")
        self.resource_paths = tuple(str(Path(self._resources.name)/f"replica-{r}.json") for r in range(shards))
        ctx = mp.get_context('spawn')
        initialization = _pack((parameters, manager.num_fragments, method, options))
        pairs = [ctx.Pipe() for _ in range(shards-1)]
        try:
            for rank in range(shards):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_worker, args=(rank, [pair[0] for pair in pairs] if rank == 0 else [pairs[rank-1][1]], child, initialization, timeout, self.resource_paths[rank]), name=f'syncer-replica-{rank}')
                process.start()
                child.close()
                self._connections.append(parent)
                self.processes.append(process)
            for pair in pairs:
                for peer in pair:
                    peer.close()
            deadline = time.monotonic() + timeout
            for connection in self._connections:
                self._receive(connection, deadline)
        except Exception:
            for pair in pairs:
                for peer in pair:
                    peer.close()
            self.close(force=True)
            raise

    @property
    def pids(self):
        return tuple(p.pid for p in self.processes)

    def _receive(self, connection, deadline):
        while not connection.poll(min(.01, max(0, deadline-time.monotonic()))):
            if time.monotonic() >= deadline:
                raise TimeoutError('syncer replica response timed out; aborting run')
            if self.progress is not None:
                self.progress()
        try:
            status, value = _unpack(connection.recv_bytes())
        except (EOFError, OSError) as exc:
            raise RuntimeError('syncer replica disconnected; aborting run') from exc
        if status != 'ok':
            raise RuntimeError(value)
        return value

    def _send(self, connection, value, deadline):
        # A stopped replica must not trap the coordinator in a full IPC pipe.
        packed = _pack(value)
        completed, errors = threading.Event(), []
        def send():
            try:
                connection.send_bytes(packed)
            except Exception as exc:
                errors.append(exc)
            finally:
                completed.set()
        threading.Thread(target=send, daemon=True).start()
        while not completed.wait(min(.01, max(0, deadline-time.monotonic()))):
            if time.monotonic() >= deadline:
                raise TimeoutError('syncer replica send timed out; aborting run')
            if self.progress is not None:
                self.progress()
        if errors:
            raise RuntimeError('syncer replica send failed') from errors[0]

    def merge_step(self, fragment_id, snapshots, weights, merge):
        if self._closed:
            raise RuntimeError('syncer replicas are closed')
        ids = sorted(snapshots)
        assignments = [[] for _ in range(self.shards)]
        for learner_id, weight in zip(ids, weights, strict=True):
            s = snapshots[learner_id]
            assignments[learner_id % self.shards].append((dict(s.baseline), dict(s.current), weight))
        deadline = time.monotonic() + self.timeout
        try:
            for connection, contributions in zip(self._connections, assignments, strict=True):
                self._send(connection, ('step', (fragment_id, contributions, merge)), deadline)
            results = [self._receive(c, deadline) for c in self._connections]
            self.stats.extend(r[2] for r in results)
            return results[0][0], results[0][1]
        except Exception:
            self.close(force=True)
            raise

    def _read(self, command, *args, rank=0):
        if self._closed:
            raise RuntimeError('syncer replicas are closed')
        try:
            deadline = time.monotonic()+self.timeout
            self._send(self._connections[rank], (command, args), deadline)
            return self._receive(self._connections[rank], deadline)
        except Exception:
            self.close(force=True)
            raise

    def snapshot(self, fragment_id):
        return self._read('snapshot', fragment_id)

    def dispatch_snapshot(self, fragment_id):
        return self._read('dispatch_snapshot', fragment_id)

    def momentum_snapshot(self, fragment_id):
        return self._read('momentum_snapshot', fragment_id)

    def model_snapshot(self):
        return self._read('model_snapshot')

    def close(self, *, force=False):
        if self._closed:
            return
        self._closed = True
        if force:
            for process in self.processes:
                if process.is_alive():
                    process.kill()
        for connection in self._connections:
            try:
                if not force:
                    connection.send_bytes(_pack(('close', None)))
            except (OSError, EOFError):
                pass
            connection.close()
        deadline = time.monotonic()+2
        for process in self.processes:
            process.join(max(0, deadline-time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join(2)
        self._resources.cleanup()
