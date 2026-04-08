# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Route Env Environment.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

# Flat structure — models and server files are all at repo root
from models import RouteAction, RouteObservation
from server.route_env_environment import RouteEnvironment

app = create_app(
    RouteEnvironment,
    RouteAction,
    RouteObservation,
    env_name="route_env",
    max_concurrent_envs=1,
)


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()