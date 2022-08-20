import os

import torch.distributed as dist

from .config import Config


def setup(rank, world_size):
    c = Config()
    os.environ['MASTER_ADDR'] = c.master_addr
    os.environ['MASTER_PORT'] = str(c.master_port)

    # initialize the process group
    dist.init_process_group(c.process_group_backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
