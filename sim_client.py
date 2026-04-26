"""
Persistent MuJoCo sim client.

Connects once to sim_server.py and runs multi-step sequences with real-time
feedback between each step. No MuJoCo restart needed.

Usage:
    python sim_client.py
"""

import asyncio
import json
import websockets

WS_URL = "ws://localhost:8765"


class SimClient:
    def __init__(self, ws):
        self.ws = ws
        self._pending: asyncio.Future | None = None

    async def _recv_loop(self):
        """Background task: print state updates, resolve pending futures on 'done'."""
        async for raw in self.ws:
            msg = json.loads(raw)
            if msg["type"] == "state":
                # pretty-print at low rate (every ~2 s) so terminal isn't spammed
                t = msg["t"]
                if abs(t - round(t, 0)) < 0.02:
                    print(f"  t={t:.1f}  cube={msg['cube_pos']}  "
                          f"ee={msg['ee_pos']}  touch={msg['touch']}")
            elif msg["type"] == "done":
                print(f"  ✓ done: {msg['cmd']}")
                if self._pending and not self._pending.done():
                    self._pending.set_result(msg)
            elif msg["type"] == "error":
                print(f"  ✗ error: {msg['msg']}")
                if self._pending and not self._pending.done():
                    self._pending.set_exception(RuntimeError(msg["msg"]))

    async def send(self, cmd: dict) -> dict:
        """Send a command and wait for the server's 'done' reply."""
        loop = asyncio.get_running_loop()
        self._pending = loop.create_future()
        await self.ws.send(json.dumps(cmd))
        print(f"→ {cmd}")
        return await self._pending

    # ── convenience helpers ───────────────────────────────────────────────────

    async def pick_and_place(self):
        return await self.send({"cmd": "pick_and_place"})

    async def set_drop(self, x, y):
        return await self.send({"cmd": "set_drop", "xy": [x, y]})

    async def reset(self):
        return await self.send({"cmd": "reset"})


# ── multi-step demo ───────────────────────────────────────────────────────────

async def run_sequence(client: SimClient):
    print("\n=== Step 1: pick cube from default position, drop right ===")
    await client.set_drop(0.60, -0.15)
    await client.pick_and_place()

    print("\n=== Step 2: reset arm to home ===")
    await client.reset()

    print("\n=== Step 3: pick cube from where it landed, drop left ===")
    await client.set_drop(0.60, 0.15)
    await client.pick_and_place()

    print("\n=== Step 4: reset arm ===")
    await client.reset()

    print("\n=== Step 5: pick it back to centre ===")
    await client.set_drop(0.60, 0.0)
    await client.pick_and_place()

    print("\nSequence complete!")


async def main():
    print(f"Connecting to {WS_URL} ...")
    async with websockets.connect(WS_URL) as ws:
        client = SimClient(ws)
        recv   = asyncio.create_task(client._recv_loop())
        try:
            await run_sequence(client)
        finally:
            recv.cancel()


if __name__ == "__main__":
    asyncio.run(main())
