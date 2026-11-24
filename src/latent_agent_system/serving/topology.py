"""Describe worker placement and derive a request route without starting workers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Worker:
    gpu: str
    role: str
    port: int
    side_channel_port: int

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self.port}'


@dataclass(frozen=True)
class Route:
    decode: str
    prefill: str | None = None


@dataclass(frozen=True)
class Placement:
    workers: tuple[Worker, ...]
    prefiller_count: int

    @property
    def gpus(self) -> tuple[str, ...]:
        return tuple(worker.gpu for worker in self.workers)

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(worker.url for worker in self.workers)

    def route(self, index: int) -> Route:
        n = self.prefiller_count
        decode_index = n + index % (len(self.workers) - n)
        prefill = self.workers[index % n].url if n else None
        return Route(decode=self.workers[decode_index].url, prefill=prefill)


def plan_placement(gpus, prefiller_count: int, base_port: int) -> Placement:
    gpus = tuple(gpus)
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError('Choose distinct GPU identifiers.')
    if not 0 <= prefiller_count < len(gpus):
        raise ValueError('The placement requires at least one decode worker.')
    stride = max(16, len(gpus))
    if base_port < 1 or base_port + stride + len(gpus) - 1 > 65535:
        raise ValueError('The placement exceeds the available TCP port range.')
    workers = []
    for index, gpu in enumerate(gpus):
        role = 'colocated'
        if prefiller_count:
            role = 'kv_producer' if index < prefiller_count else 'kv_consumer'
        workers.append(Worker(gpu, role, base_port + index, base_port + stride + index))
    return Placement(tuple(workers), prefiller_count)
