"""Replay workload arrivals against the native serving client."""

import asyncio
import time

from latent_agent_system.serving.vllm_client import client_for


async def run_serial(cases, send, connections):
    async with client_for(connections) as client:
        return [await send(client, case, index) for index, case in enumerate(cases)]


async def replay(cases, offsets, rate, send):
    async with client_for(len(cases)) as client:
        origin = time.perf_counter() + .05

        async def arrival(index, case, offset):
            scheduled = origin + offset / rate
            await asyncio.sleep(max(0, scheduled - time.perf_counter()))
            row = await send(client, case, index)
            return {**row, 'scheduled': scheduled}

        return await asyncio.gather(*(
            arrival(index, case, offset)
            for index, (case, offset) in enumerate(zip(cases, offsets, strict=True))
        ))
