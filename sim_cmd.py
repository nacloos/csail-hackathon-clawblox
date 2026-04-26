"""Send a single command to sim_server and wait for done. Usage: python sim_cmd.py '<json>'"""
import asyncio, json, sys
import websockets

WS_URL = "ws://localhost:8765"

async def run(cmd):
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps(cmd))
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "state":
                continue
            if msg["type"] == "done":
                print(json.dumps(msg))
                return
            if msg["type"] == "error":
                print(json.dumps(msg), file=sys.stderr)
                sys.exit(1)

asyncio.run(run(json.loads(sys.argv[1])))
