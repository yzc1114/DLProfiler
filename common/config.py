from common.singleton import singleton


def init_config(master_addr="localhost",
                master_port=39901,
                world_size=1,
                rank=0,
                local_rank=0,
                device_type="cpu",
                process_group_backend="gloo",
                mem_utilization_monitor_interval:float=0.5):
    c = Config()
    c.master_addr = master_addr
    c.master_port = master_port
    c.world_size = world_size
    c.rank = rank
    c.local_rank = local_rank
    c.process_group_backend = process_group_backend
    c.device_type = device_type
    c.mem_utilization_monitor_interval = mem_utilization_monitor_interval


@singleton
class Config:
    def __init__(self, master_addr="localhost",
                 master_port=39901,
                 world_size=1,
                 rank=0,
                 local_rank=0,
                 device_type="cpu",
                 process_group_backend="gloo",
                 mem_utilization_monitor_interval: float = 1):
        self.master_addr: str = master_addr
        self.master_port: int = master_port
        self.world_size: int = world_size
        self.rank: int = rank
        self.local_rank: int = local_rank
        self.device_type: str = device_type
        self.process_group_backend: str = process_group_backend
        self.mem_utilization_monitor_interval: float = mem_utilization_monitor_interval
        self.device: str = "cpu" if device_type == "cpu" else self.local_rank
