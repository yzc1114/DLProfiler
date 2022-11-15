import argparse
import http
import sys
from log import init_logging, logging
from posixpath import join as urljoin

import requests
import torch.cuda

from common import init_config, process_group
from data_collector import ProfiledDataModel
from model_factory import generate_model_profiler
from objects import ModelDescriptions

init_logging()

def main():
    parser = argparse.ArgumentParser(description='PyTorch Deep Learning Profiler')
    model_names = [md.value.name for md in ModelDescriptions]
    parser.add_argument('--session-id',
                        type=str,
                        required=True,
                        help="session id to identify a unique profile event")
    parser.add_argument('--data-collector-url',
                        type=str,
                        required=True,
                        help="data collector url. profiled data will be sent to the data collector on this url")
    parser.add_argument('--model', type=str,
                        required=True,
                        choices=model_names,
                        help=f'type of model to profile [{model_names}]')
    parser.add_argument('--computation-proportion', type=int,
                        required=True,
                        default=100,
                        help=f'just for recording the specified computation power')
    parser.add_argument('--duration-sec', type=int,
                        required=True,
                        default=10,
                        help=f'how long duration it profiles')
    parser.add_argument('--batch-size',
                        type=int,
                        required=True,
                        help='the batch size of the model')
    parser.add_argument('--mode', choices=["train", "inference", "checkpoint"], type=str, required=True)
    parser.add_argument('--dist-data-parallel',
                        action="store_true",
                        help='use distributed data parallel to train, default is true')
    parser.add_argument('--dist-model-parallel',
                        dest="dist_data_parallel",
                        action="store_true",
                        help='use distributed model parallel to train, default is false')
    parser.add_argument('--device_type',
                        type=str,
                        choices=["cpu", "gpu"],
                        default="cpu",
                        help='using which type of device to train')
    parser.set_defaults(dist_data_parallel=True)
    parser.add_argument('--process-group-backend',
                        type=str,
                        choices=["gloo", "nccl"],
                        default="gloo",
                        help='distributed data parallel process group backend')
    parser.add_argument('--world-size',
                        type=int,
                        default=1,
                        help='world size of distributed data parallel')
    parser.add_argument('--rank',
                        type=int,
                        default=0,
                        help='rank of distributed data parallel')
    parser.add_argument('--local-rank',
                        type=int,
                        default=0,
                        help='local rank of distributed data parallel')
    parser.add_argument('--master-addr',
                        type=str,
                        default="localhost",
                        help='master addr used in distributed data parallel')
    parser.add_argument('--master-port',
                        type=int,
                        default=12345,
                        help='master port used in distributed data parallel')
    parser.add_argument('--cuda-monitor-interval',
                        type=float,
                        default=0.5,
                        help='check cuda memory utilization every {interval} second')
    args = parser.parse_args()
    torch.cuda.init()
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}, torch.cuda.is_initialized(): {torch.cuda.is_initialized()}")
    assert args.local_rank < args.world_size, "local rank cannot be greater than or equal to world size"
    # init singleton configuration object
    init_config(master_addr=args.master_addr,
                master_port=args.master_port,
                world_size=args.world_size,
                rank=args.rank,
                local_rank=args.local_rank,
                device_type=args.device_type,
                process_group_backend=args.process_group_backend,
                mem_utilization_monitor_interval=args.cuda_monitor_interval)
    process_group.setup()
    mode = args.mode
    model_profiler = generate_model_profiler(model_name=args.model,
                                             mode=mode,
                                             is_DDP=args.dist_data_parallel)
    print(f"model profiler generated: model = {args.model}, starting profile for {args.duration_sec} seconds")
    profiled_iterator = model_profiler.profile(batch_size=args.batch_size, duration_sec=args.duration_sec)
    print(f"model {args.model} profiling done, profiled_iterator: {profiled_iterator}")
    profiled_data = profiled_iterator.to_dict()
    model = ProfiledDataModel(
        model_name=args.model,
        batch_size=args.batch_size,
        duration_sec=args.duration_sec,
        mode=mode,
        is_DDP=args.dist_data_parallel,
        process_group_backend=args.process_group_backend,
        world_size=args.world_size,
        rank=args.rank,
        local_rank=args.local_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        iteration_intervals=profiled_data.get("iteration_intervals"),
        iteration_count=profiled_data.get("iteration_count"),
        iteration_intervals_avg=profiled_data.get("iteration_intervals_avg"),
        total_time_ns=profiled_data.get("total_time_ns"),
        mem_infos=profiled_data.get("mem_infos"),
        utilization=profiled_data.get("utilization"),
        extra_dict=profiled_data.get("extra_dict"),
        computation_proportion=args.computation_proportion,
    )
    session_id = args.session_id
    data_collector_url = args.data_collector_url
    response = requests.post(urljoin(data_collector_url, f"{session_id}/{args.rank}"), data=model.json())
    if response.status_code != http.HTTPStatus.OK:
        print(f"problem encountered sending profiled data to {data_collector_url}, status code = {response.status_code}")
        sys.exit(-1)
    print(f"profiled data is sent to {data_collector_url}")
    return


if __name__ == '__main__':
    main()
