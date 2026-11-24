"""GPU process queries and HF KV movement between devices."""

import subprocess


def state_size(cache):
    return 0 if cache is None else sum(
        tensor.numel() * tensor.element_size()
        for layer in cache.to_legacy_cache() for tensor in layer
    )


def compute_pids(uuid):
    query = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']
    rows = subprocess.check_output(query, text=True).splitlines()
    return {int(row.split(',')[1]) for row in rows if row.split(',')[0].strip() == uuid}


def relocate_tensor(tensor, target, transport):
    if tensor.device == target:
        return tensor
    if transport == 'host_staged':
        return tensor.cpu().to(target)
    return tensor.to(target)


def relocate_cache(cache, target, transport):
    if cache is None or transport == 'same_gpu_ipc':
        return cache
    legacy = cache.to_legacy_cache()
    return type(cache).from_legacy_cache(tuple(
        tuple(relocate_tensor(t, target, transport) for t in layer) for layer in legacy
    ))
