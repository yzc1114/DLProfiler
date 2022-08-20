import argparse
import http
import sys
from posixpath import join as urljoin

import requests

from common import init_config, process_group
from data_collector import ProfiledDataModel
from model_factory import generate_model_profiler
from profiler_objects import ModelDescriptions


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
    parser.add_argument('--duration-sec', type=int,
                        required=True,
                        default=10,
                        help=f'how long duration it profiles')
    parser.add_argument('--batch-size',
                        type=int,
                        required=True,
                        help='the batch size of the model')
    parser.add_argument('--is-train',
                        type=bool,
                        required=True,
                        choices=[True, False],
                        help='profile training or inference')
    parser.add_argument('--dist-data-parallel',
                        type=bool,
                        default=True,
                        choices=[True, False],
                        help='use distributed data parallel to train, default is true')
    parser.add_argument('--process-group-backend',
                        type=str,
                        choices=["gloo", "nccl"],
                        default="gloo",
                        help='distributed data parallel process group backend')
    parser.add_argument('--world-size',
                        type=int,
                        default=1,
                        help='world size of distributed data parallel')
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
                        default=39910,
                        help='master port used in distributed data parallel')
    parser.add_argument('--cuda-monitor-interval',
                        type=float,
                        default=0.5,
                        help='check cuda memory utilization every {interval} second')
    args = parser.parse_args()
    assert args.local_rank < args.world_size, "local rank cannot be greater than or equal to world size"
    process_group.setup(args.local_rank, args.world_size)

    # init singleton configuration object
    init_config(master_addr=args.master_addr, master_port=args.master_port, world_size=args.world_size,
                local_rank=args.local_rank, process_group_backend=args.process_group_backend,
                mem_utilization_monitor_interval=args.cuda_monitor_interval)

    model_profiler = generate_model_profiler(model_name=args.model, is_train=args.is_train,
                                             is_DDP=args.dist_data_parallel)
    profiled_iterator = model_profiler.profile(batch_size=args.batch_size, duration_sec=args.duration_sec)
    profiled_data = profiled_iterator.to_dict()
    model = ProfiledDataModel(
        model_name=args.model,
        batch_size=args.batch_size,
        duration_sec=args.duration_sec,
        is_train=args.is_train,
        is_DDP=args.dist_data_parallel,
        process_group_backend=args.process_group_backend,
        world_size=args.world_size,
        local_rank=args.local_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        iteration_intervals=profiled_data.get("iteration_intervals"),
        iteration_count=profiled_data.get("iteration_count"),
        iteration_intervals_avg=profiled_data.get("iteration_intervals_avg"),
        total_time_ns=profiled_data.get("total_time_ns"),
        mem_infos=profiled_data.get("mem_infos"),
        utilization=profiled_data.get("utilization"),
    )
    session_id = args.session_id
    data_collector_url = args.data_collector_url
    response = requests.post(urljoin(data_collector_url, f"{session_id}/{args.local_rank}"), data=model.json())
    if response.status_code != http.HTTPStatus.OK:
        sys.exit(-1)
    return


if __name__ == '__main__':
    main()
