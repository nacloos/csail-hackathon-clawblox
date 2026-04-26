"""Print one state snapshot from the sim. Usage: python sim_state.py"""
import asyncio, json
import websockets

async def run():
    async with websockets.connect("ws://localhost:8765") as ws:
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "state":
                print(json.dumps(msg, indent=2))
                return

asyncio.run(run())
