"""Collect and validate NIXL transfer counters for a placement."""

import time

import httpx


def transfer_counters(urls, out, label):
    keys = ('nixl_xfer_time_seconds_count', 'nixl_xfer_time_seconds_sum',
            'nixl_bytes_transferred_sum', 'nixl_num_failed_transfers_total',
            'nixl_num_kv_expired_reqs_total', 'nixl_num_failed_notifications_total')
    totals = {}
    with httpx.Client(timeout=10.0, trust_env=False) as client:
        for index, url in enumerate(urls):
            response = client.get(url + '/metrics')
            response.raise_for_status()
            (out / f'{label}_worker{index}.prom').write_text(response.text)
            for row in response.text.splitlines():
                if not row.startswith('vllm:nixl_'):
                    continue
                metric, value = row.split()[:2]
                key = metric.split('{')[0].removeprefix('vllm:')
                if key in keys:
                    totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def read_delta(before, after):
    return {key: value - before.get(key, 0.0) for key, value in after.items()}


def check_transfer(stats, minimum):
    if (stats.get('nixl_xfer_time_seconds_count', 0) < minimum
            or stats.get('nixl_bytes_transferred_sum', 0) <= 0
            or any(value for key, value in stats.items() if 'failed' in key or 'expired' in key)):
        raise RuntimeError(f'NIXL did not deliver the expected KV without errors: {stats}')


def capture_after(urls, out, label, baseline, count):
    deadline = time.monotonic() + 18.0
    while True:
        delta = read_delta(baseline, transfer_counters(urls, out, label))
        if not count or delta.get('nixl_xfer_time_seconds_count', 0) >= count:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(1)
    if count:
        check_transfer(delta, count)
    return delta
