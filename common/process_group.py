import os

import torch.distributed as dist

from .config import Config


def setup():
    c = Config()
    print(f"setup process group, master_addr: {c.master_addr}, master_port: {c.master_port}, world_size: {c.world_size}, rank: {c.rank}, backend: {c.process_group_backend}")
    os.environ['MASTER_ADDR'] = c.master_addr
    os.environ['MASTER_PORT'] = str(c.master_port)

    # initialize the process group
    dist.init_process_group(c.process_group_backend, rank=c.rank, world_size=c.world_size)


def cleanup():
    dist.destroy_process_group()
