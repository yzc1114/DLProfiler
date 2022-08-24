import argparse
import datetime
import json
import os
from typing import List, Dict
from log import init_logging, logging

import uvicorn
from fastapi import FastAPI, Path
from pydantic import BaseModel

init_logging()

app = FastAPI()

class ProfiledDataModel(BaseModel):
    model_name: str
    batch_size: int
    duration_sec: int
    is_train: bool
    is_DDP: bool
    process_group_backend: str
    world_size: int
    rank: int
    local_rank: int
    master_addr: str
    master_port: int
    iteration_count: int
    iteration_intervals: List[int]
    iteration_intervals_avg: float
    total_time_ns: int
    mem_infos: List[List[int]]
    utilization: List[int]
    computation_proportion: int


@app.post("/receive/{session_id}/{rank}")
def receive(*,
            session_id: str = Path(..., title="session id"),
            rank: int = Path(..., title="rank"),
            profiled_data_model: ProfiledDataModel):
    filename = datetime.datetime.now().strftime(f"{session_id}_rank_{rank}_model_{'train' if profiled_data_model.is_train else 'inference'}_{profiled_data_model.model_name}_batch_{profiled_data_model.batch_size}_comp_{profiled_data_model.computation_proportion}_%Y-%m-%d-%H-%M-%S.json")
    session_dir = os.path.join(args.data_dir_path, session_id)
    if not os.path.exists(session_dir):
        os.mkdir(session_dir)
    filepath = os.path.join(session_dir, filename)
    logging.info(f"received profiled data, session_id = {session_id}, saving file to {filepath}")
    with open(filepath, 'w') as f:
        json.dump(profiled_data_model.dict(), f, indent='\t')
    return {}


@app.get("/health")
def health():
    return "I'm healthy"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Deep Learning Profiler -- Data Collector')
    parser.add_argument('--data-dir-path',
                        type=str,
                        required=True,
                        help="data target dir path")
    parser.add_argument('--host',
                        type=str,
                        default="0.0.0.0",
                        help="host")
    parser.add_argument('--port',
                        type=int,
                        default=80,
                        help="port")
    args = parser.parse_args()
    assert os.path.exists(args.data_dir_path) and os.path.isdir(
        args.data_dir_path), "data dir path not exists or is not a dir!"
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
