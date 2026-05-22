import asyncio
import http
import logging
import time
import traceback

import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


def _format_obs_prompt(obs: dict) -> str:
    """Return a readable prompt string from client observation, if present."""
    if "prompt" not in obs:
        return "<missing; server will use default_prompt>"
    prompt = obs["prompt"]
    if hasattr(prompt, "item"):
        prompt = prompt.item()
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8")
    return str(prompt)


def _extract_obs_state(obs: dict) -> np.ndarray | None:
    """Return proprio state vector from common client observation keys."""
    for key in ("state", "observation.state", "observation/state"):
        if key not in obs:
            continue
        state = obs[key]
        if hasattr(state, "cpu"):
            state = state.cpu().numpy()
        state = np.asarray(state).squeeze()
        if state.ndim == 1 and state.size > 0:
            return state
    return None


def _format_obs_gripper_state(obs: dict) -> str:
    """Format gripper values from raw or 24-dim reordered client state."""
    state = _extract_obs_state(obs)
    if state is None:
        return "gripper state=<missing>"

    parts = [f"state_dim={state.size}"]
    # Raw G2 observation.state (159-dim): effectors at indices 0/1.
    if state.size >= 2:
        parts.append(f"raw[0:2]=({state[0]:.4f}, {state[1]:.4f})")
    # 24-dim layout after Go2ACOTInputs reorder: grippers at indices 14/15.
    if state.size >= 16:
        parts.append(f"reordered[14:16]=({state[14]:.4f}, {state[15]:.4f})")
    return "gripper " + ", ".join(parts)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                logger.info("Client prompt: %s", _format_obs_prompt(obs))
                logger.info("Client %s", _format_obs_gripper_state(obs))

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
