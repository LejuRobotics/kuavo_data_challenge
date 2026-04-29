# Copyright (C) 2025-2026 LejuRobotics.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ---
#
# This project includes code from LeRobot (https://github.com/huggingface/lerobot),
# which is licensed under the Apache License, Version 2.0.

import lerobot_patches.custom_patches
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any, Callable, Dict
from io import BytesIO
import cv2
from lerobot.policies.act.modeling_act import ACTPolicy
import torch
import zmq
import numpy as np
from kuavo_deploy.config import load_kuavo_config
import torch
from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from kuavo_train.wrapper.policy.gr00t_n1d5.Gr00tN1d5PolicyWrapper import CustomGr00tN1d5PolicyWrapper
class TorchSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        obj = torch.load(buffer, weights_only=False)
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = "*", port: int = 5555, api_token: str = None):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _validate_token(self, request: dict) -> bool:
        """
        Validate the API token in the request.
        """
        if self.api_token is None:
            return True  # No token required
        return request.get("api_token") == self.api_token

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = TorchSerializer.from_bytes(message)

                # Validate token before processing request
                if not self._validate_token(request):
                    self.socket.send(
                        TorchSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                    )
                    continue

                endpoint = request.get("endpoint", "select_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(TorchSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(TorchSerializer.to_bytes({"error": str(e)}))

class RobotInferenceServer(BaseInferenceServer):
    """
    Server with three endpoints for real robot policies
    """

    def __init__(self, model, host: str = "*", port: int = 5555, api_token: str = None):
        super().__init__(host, port, api_token)
        self.register_endpoint("select_action", model.select_action)

    @staticmethod
    def start_server(policy, port: int, api_token: str = None):
        server = RobotInferenceServer(policy, port=port, api_token=api_token)
        server.run()

#####################################################################################

# Convert raw observations into the observations required by the model
def hardware_obses_to_policy_obs_dict(obs):
    obs_dict = obs
    return obs_dict

class Policy():
    def __init__(self):
        # load config
        config_path = 'configs/deploy/kuavo_env.yaml'
        cfg = load_kuavo_config(config_path)

        use_delta = cfg.env.use_delta
        eval_episodes = cfg.inference.eval_episodes
        seed = cfg.inference.seed
        start_seed = cfg.inference.start_seed
        policy_type = cfg.inference.policy_type
        task = cfg.inference.task
        method = cfg.inference.method
        timestamp = cfg.inference.timestamp
        epoch = cfg.inference.epoch
        env_name = cfg.env.env_name
        depth_range = cfg.env.depth_range

        pretrained_path = Path(f"outputs/train/{task}/{method}/{timestamp}/epoch{epoch}")

        # Select your device
        device = torch.device(cfg.inference.device)
        # self.policy = ACTPolicy.from_pretrained(Path(pretrained_path),strict=True)
        # self.policy = CustomDiffusionPolicyWrapper.from_pretrained(Path(pretrained_path),strict=True)
        if policy_type == "gr00t_n1d5":
            self.policy = CustomGr00tN1d5PolicyWrapper.from_pretrained(Path(pretrained_path),strict=True)
        elif policy_type == "diffusion":
            self.policy = CustomDiffusionPolicyWrapper.from_pretrained(Path(pretrained_path),strict=True)
        elif policy_type == "act":
            self.policy = ACTPolicy.from_pretrained(Path(pretrained_path),strict=True)
        self.policy.eval()
        self.policy.to(device)
        self.policy.reset()

    def select_action(self,obs):
        obs = hardware_obses_to_policy_obs_dict(obs)
        return self.policy.select_action(obs)


def main():
    policy = Policy()

    # Start the server
    server = RobotInferenceServer(policy, port=5555, api_token=None)
    server.run()


if __name__ == "__main__":
    main()
