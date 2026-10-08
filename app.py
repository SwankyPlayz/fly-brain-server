import asyncio
import json
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------
# STAND-IN BRAIN: a random spiking network, NOT the real fly connectome yet.
# Later, replace W (and the SUGAR / TOUCH neuron groups) with the real subset.
# ---------------------------------------------------------------
rng = np.random.default_rng(1)
N = 1500
W = ((rng.random((N, N)) < 0.01) * rng.normal(0.4, 0.1, (N, N))).astype(np.float32)
WT = np.ascontiguousarray(W.T)
v = np.zeros(N, dtype=np.float32)       # membrane potentials
drive = np.zeros(N, dtype=np.float32)   # input currents from player actions
SUGAR = slice(0, 100)                   # pretend "taste" neurons
TOUCH = slice(100, 200)                 # pretend "touch" neurons


def step() -> int:
    """Advance the brain one tick and return how many neurons fired."""
    global v
    fired = v > 1.0
    v[fired] = 0.0
    noise = rng.normal(0, 0.02, N).astype(np.float32)
    v = 0.95 * v + 0.5 * (WT @ fired.astype(np.float32)) + drive + noise
    drive *= 0.8
    return int(fired.sum())


# ---------------------------------------------------------------
# Shared community state (kept in memory)
# ---------------------------------------------------------------
state = {
    "hunger": 0.5,
    "happy": 0.5,
    "energy": 0.8,
    "spikes": 0,
    "online": 0,
    "feeds_today": 0,
    "pets_today": 0,
}
clients: set[WebSocket] = set()
current_day = time.strftime("%Y-%m-%d", time.gmtime())


async def safe_send(sock: WebSocket, msg: str):
    try:
        await sock.send_text(msg)
        return None
    except Exception:
        return sock  # report broken connection


async def broadcast():
    if not clients:
        return
    msg = json.dumps(state)
    results = await asyncio.gather(*(safe_send(c, msg) for c in list(clients)))
    for dead in results:
        if dead is not None:
            clients.discard(dead)
    state["online"] = len(clients)


async def tick_loop():
    global current_day
    last = time.time()
    while True:
        now = time.time()
        dt = now - last
        last = now

        # daily counters reset at UTC midnight
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if today != current_day:
            current_day = today
            state["feeds_today"] = 0
            state["pets_today"] = 0

        # slow life-cycle changes (hunger empties in about an hour)
        state["hunger"] = min(1.0, state["hunger"] + 0.00025 * dt)
        state["happy"] = max(0.0, state["happy"] - 0.00015 * dt)
        state["energy"] = max(0.0, min(1.0, 1.0 - 0.7 * state["hunger"]))

        if clients:
            state["spikes"] = step()
            await broadcast()
            await asyncio.sleep(0.2)   # about 5 updates per second
        else:
            state["spikes"] = 0
            await asyncio.sleep(1.0)   # nobody here: idle gently


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(tick_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"name": "Fly World brain server", "ok": True}


@app.get("/health")
def health():
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(sock: WebSocket):
    await sock.accept()
    clients.add(sock)
    state["online"] = len(clients)
    last_action = 0.0
    try:
        while True:
            raw = await sock.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = msg.get("type") if isinstance(msg, dict) else None
            now = time.time()
            if kind not in ("feed", "pet") or now - last_action < 1.0:
                continue  # ignore junk and spam (1 action per second per player)
            last_action = now
            if kind == "feed":
                drive[SUGAR] += 1.5
                state["hunger"] = max(0.0, state["hunger"] - 0.05)
                state["feeds_today"] += 1
            else:
                drive[TOUCH] += 1.5
                state["happy"] = min(1.0, state["happy"] + 0.05)
                state["pets_today"] += 1
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(sock)
        state["online"] = len(clients)
