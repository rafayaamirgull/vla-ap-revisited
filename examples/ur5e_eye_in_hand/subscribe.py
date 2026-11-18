"""Read the optional RGB-D ZMQ stream from run.py."""

import argparse
import json

import numpy as np
import zmq


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--connect", default="tcp://127.0.0.1:5555")
args = parser.parse_args()

with zmq.Context() as context:
    with context.socket(zmq.SUB) as subscriber:
        subscriber.connect(args.connect)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"rgbd")
        print(f"Listening on {args.connect}", flush=True)
        while True:
            topic, header, rgb_bytes, depth_bytes = subscriber.recv_multipart()
            meta = json.loads(header)
            shape = (meta["height"], meta["width"])
            rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(*shape, 3)
            depth = np.frombuffer(depth_bytes, dtype="<f4").reshape(shape)
            print(f"frame={meta['frame']} t={meta['simulation_time_s']:.3f}s "
                  f"rgb={rgb.shape} depth={depth.shape} center={depth[shape[0] // 2, shape[1] // 2]:.3f}m",
                  flush=True)
