from dataclasses import dataclass
from typing import List
import os
import argparse
import json
import numpy as np


@dataclass
class RequestMocker:

    def __post_init__(self):
        self.rng = np.random.default_rng(seed=42)

    def tfs_input(self, inp: List[List[int]]) -> dict:
        payload = {
            "inputs": {
                "input_ids": inp,
                "training": False,
            },
            "signature_name": "serving_default",
        }
        return payload

    def triton_input(self, inp: List[List[int]]) -> dict:
        batch_size = len(inp)
        seq_len = len(inp[0])
        payload = {
            "inputs": [{
                "name": "input",
                "shape": [batch_size, seq_len],
                "datatype": "INT32",
                "data": inp
            }]
        }
        return payload

    def get_host(self, serving: str, model: str):
        host = "localhost"
        if serving == "tfserving":
            api = f"http://{host}:8501/v1/models/{model}:predict"
        elif serving == "triton":
            api = f"http://{host}:8000/v2/models/{model}/infer"
        else:
            raise NotImplementedError
        return api

    def __call__(self,
                 serving: str,
                 model: str,
                 batch_size: int,
                 seq_len: int = 60):
        inp = self.rng.integers(101,
                                8021, (batch_size, seq_len),
                                dtype=np.int32).tolist()
        if serving == "tfserving":
            body = self.tfs_input(inp)
        elif serving == "triton":
            body = self.triton_input(inp)
        else:
            raise NotImplementedError
        return body


def main():

    """
    python3.8 run.py -s tfserving -m mcls -c 1 -b 1 -t 5S -l log.txt
    """

    rm = RequestMocker()

    parser = argparse.ArgumentParser(description="Stress Test for HTTP API")
    parser.add_argument("-s", dest="serving", type=str, help="model type")
    parser.add_argument("-m", dest="model", type=str, help="model name")
    parser.add_argument("-c", dest="concurrency", type=int, help="concurrency")
    parser.add_argument("-b", dest="batch_size", type=int, help="batch size")
    parser.add_argument("-t", dest="time", type=str, help="execute time")
    parser.add_argument("-l", dest="log", type=str, help="log file")
    args = parser.parse_args()

    print(args)

    serving = args.serving
    model = args.model
    concurrency = args.concurrency
    batch_size = args.batch_size
    log_file = args.log
    time = args.time

    body = rm(serving, model, batch_size)
    body_str = json.dumps(body)
    host = rm.get_host(serving, model)

    cmd = f"siege -j -l{log_file} -c{concurrency} -t{time} --content-type \"application/json\" '{host} POST {body_str}'"

    os.system("echo concurrency:  {}, batch_size: {}  for {}".format(
        concurrency, batch_size, model))
    os.system(cmd)


if __name__ == "__main__":
    main()
