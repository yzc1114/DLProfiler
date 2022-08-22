import argparse
import datetime
import json
import os
from typing import List, Dict

import uvicorn
from fastapi import FastAPI, Path
from pydantic import BaseModel

app = FastAPI()

class ProfiledDataModel(BaseModel):
    model_name: str
    batch_size: int
    duration_sec: int
    is_train: bool
    is_DDP: bool
    process_group_backend: str
    world_size: int
    local_rank: int
    master_addr: str
    master_port: int
    iteration_count: int
    iteration_intervals: List[int]
    iteration_intervals_avg: float
    total_time_ns: int
    mem_infos: List[List[int]]
    utilization: List[int]


@app.post("/receive/{session_id}/{rank}")
def receive(*,
            session_id: str = Path(..., title="session id"),
            rank: int = Path(..., title="rank"),
            profiled_data_model: ProfiledDataModel):
    filename = datetime.datetime.now().strftime(f"{session_id}_rank_{rank}_%Y-%m-%d-%H-%M-%S.json")
    session_dir = os.path.join(args.data_dir_path, session_id)
    if not os.path.exists(session_dir):
        os.mkdir(session_dir)
    filepath = os.path.join(session_dir, filename)
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
                        default=39911,
                        help="port")
    args = parser.parse_args()
    assert os.path.exists(args.data_dir_path) and os.path.isdir(
        args.data_dir_path), "data dir path not exists or is not a dir!"
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
