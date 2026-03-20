#!/usr/bin/env python3
"""
inference_worker.py
Runs TFLite inference in an isolated subprocess.
Reads input from stdin (numpy array as bytes), writes output to stdout.
Called by asl_inference.py via subprocess.
"""
import os
import sys
import numpy as np
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "transformer_int8_static.tflite"

interp = Interpreter(model_path=MODEL_PATH, num_threads=2)
interp.resize_tensor_input(0, [1, 32, 75, 3])
interp.allocate_tensors()
inp_idx = interp.get_input_details()[0]["index"]
out_idx = interp.get_output_details()[0]["index"]

# Signal ready
sys.stdout.write("READY\n")
sys.stdout.flush()

while True:
    try:
        # Read 4 bytes for array size
        size_bytes = sys.stdin.buffer.read(4)
        if not size_bytes or len(size_bytes) < 4:
            break
        size = int.from_bytes(size_bytes, "little")

        # Read array data
        data = sys.stdin.buffer.read(size)
        if len(data) < size:
            break

        arr = np.frombuffer(data, dtype=np.float32).reshape(1, 32, 75, 3)
        interp.set_tensor(inp_idx, arr)
        interp.invoke()
        out = interp.get_tensor(out_idx)[0]  # (250,)

        # Write output back
        out_bytes = out.astype(np.float32).tobytes()
        sys.stdout.buffer.write(len(out_bytes).to_bytes(4, "little"))
        sys.stdout.buffer.write(out_bytes)
        sys.stdout.buffer.flush()

    except Exception as e:
        sys.stderr.write(f"[worker] Error: {e}\n")
        sys.stderr.flush()
        break
