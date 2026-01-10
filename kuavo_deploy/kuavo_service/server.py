# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
from configs.deploy.config_inference import load_inference_config
import torch
from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper

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
        config_path = 'configs/deploy/kuavo_real_env.yaml'
        cfg = load_inference_config(config_path)

        use_delta = cfg.use_delta
        eval_episodes = cfg.eval_episodes
        seed = cfg.seed
        start_seed = cfg.start_seed
        policy_type = cfg.policy_type
        task = cfg.task
        method = cfg.method
        timestamp = cfg.timestamp
        epoch = cfg.epoch
        env_name = cfg.env_name
        depth_range = cfg.depth_range

        pretrained_path = Path(f"outputs/train/{task}/{method}/{timestamp}/epoch{epoch}")

        # Select your device
        device = torch.device(cfg.device)
        # self.policy = ACTPolicy.from_pretrained(Path(pretrained_path),strict=True)
        self.policy = CustomDiffusionPolicyWrapper.from_pretrained(Path(pretrained_path),strict=True)
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
