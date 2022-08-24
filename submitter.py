import argparse
import sys
import traceback
import json
import time
import shortuuid
from collections import defaultdict
from itertools import product
from typing import List, Dict, Optional

from kubernetes import client, config

from common import singleton
from log import init_logging, logging
from objects import ModelDescriptions

init_logging()

NAMESPACE = "dl-profiler"

model_names = [model_desc.value.name for model_desc in ModelDescriptions]

config.load_kube_config()


class NodeSpec:
    def __init__(self, node_name: str, acc_specs: Dict[int, 'AccSpec'], ip_addr: str):
        self.node_name: str = node_name
        self.acc_specs: 'Dict[int, AccSpec]' = acc_specs
        self.ip_addr: str = ip_addr

    def get_acc_spec(self, acc_device_id: int) -> 'AccSpec':
        return self.acc_specs[acc_device_id]


class AccSpec:
    def __init__(self, acc_name: str, acc_device_id: int, acc_mem: int):
        assert " " not in acc_name, "acc name must does not contain space"
        self.acc_name: str = acc_name
        self.acc_device_id: int = acc_device_id
        self.acc_mem: int = acc_mem


def train_or_inference(is_train: bool):
    return "train" if is_train else "inference"


@singleton
class Config:
    """
    config sample:
        {
            "options": {
                "image": "yzc1114/dl-profiler:v0.1",
                "pull_policy": "Never",
                "default_computation_proportions": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "data_collector_receive_url": "http://133.133.135.75:32101/receive",
                "dist_master_port": 39920,
                "process_group_backend": "gloo",
                "profile_duration_sec": 30
            },
            "node_specs": {
                "dell01": {
                    "ip": "133.133.135.71",
                    "acc_specs": {
                        "0": {
                            "name": "Tesla T4",
                            "memory": 60  # 16 GBi. unit = 256 MBi
                        }
                    }
                },
                "dell04": {
                    "ip": "133.133.135.74",
                    "acc_specs": {
                        "0": {
                            "name": "RTX 2080Ti",
                            "memory": 44  # 11 GBi
                        },
                        "1": {
                            "name": "RTX 2080Ti",
                            "memory": 44
                        }
                    }
                }
            },
            "job_specs": {
                "mono": {
                    "MobileNet": {
                        "train": {
                            "node_acc_device_ids": [[["dell04"], [0]], [["dell04", "dell04"], [0, 1]], [["dell01"], [0]]],
                            "batch_sizes": [16, 32, 64],
                            "computation_proportions": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                            "memory_proportions": [44]
                        },
                        "inference": {
                            "node_acc_device_ids": [[["dell04"], [0]], [["dell04", "dell04"], [0, 1]], ["dell01", [0]]],
                            "batch_sizes": [16, 32, 64],
                            "computation_proportions": null,  # use default
                            "memory_proportions": null  # use maximum memory
                        }
                    }
                }
            }
        }
    """

    class Option:
        def __init__(self):
            self.default_computation_proportions: Optional[List[int]] = None
            self.data_collector_receive_url: Optional[str] = None
            self.dist_master_port: Optional[int] = None
            self.pull_policy: Optional[str] = None
            self.image: Optional[str] = None
            self.process_group_backend: Optional[str] = None
            self.profile_duration_sec: Optional[int] = None

    def __init__(self):
        self.options: Optional['Config.Option'] = None
        self.node_specs: Optional[Dict[str, NodeSpec]] = None
        self.mono_jobs: Optional[List['MonoJobConfig']] = None

    def parse(self, config_json: Dict):
        options_dict = config_json["options"]
        self.options = self.Option()
        self.options.default_computation_proportions = options_dict["default_computation_proportions"]
        self.options.data_collector_receive_url = options_dict["data_collector_receive_url"]
        self.options.dist_master_port = options_dict["dist_master_port"]
        self.options.pull_policy = options_dict["pull_policy"]
        self.options.image = options_dict["image"]
        self.options.process_group_backend = options_dict["process_group_backend"]
        self.options.profile_duration_sec = options_dict["profile_duration_sec"]

        node_specs_raw = config_json["node_specs"]
        self.node_specs = dict()
        for node_name, node_spec_raw in node_specs_raw.items():
            acc_specs = dict()
            ip_addr = node_spec_raw["ip"]
            for acc_device_id, acc_spec_raw in node_spec_raw["acc_specs"].items():
                acc_device_id = int(acc_device_id)
                acc_specs[acc_device_id] = AccSpec(acc_device_id=acc_device_id,
                                                   acc_name=acc_spec_raw["name"],
                                                   acc_mem=acc_spec_raw["memory"])
            self.node_specs[node_name] = NodeSpec(node_name=node_name,
                                                  acc_specs=acc_specs,
                                                  ip_addr=ip_addr)
        job_specs_raw = config_json["job_specs"]
        mono_job_specs_raw = job_specs_raw["mono"]
        self.mono_jobs = list()
        for model_name, mono_job_spec_raw in mono_job_specs_raw.items():
            for train_or_inference_text, job_spec_combinations in mono_job_spec_raw.items():
                assert train_or_inference_text in ["train", "inference"]
                is_train = train_or_inference_text == "train"
                node_acc_device_ids_list = job_spec_combinations["node_acc_device_ids"]
                for node_acc_device_ids in node_acc_device_ids_list:
                    node_names, device_ids = node_acc_device_ids
                    batch_sizes = job_spec_combinations["batch_sizes"]
                    computation_proportions = job_spec_combinations.get("computation_proportions", None)
                    memory_proportions = job_spec_combinations.get("memory_proportions", None)
                    if computation_proportions is None:
                        computation_proportions = self.options.default_computation_proportions
                    if memory_proportions is None:
                        memory_proportions = [self.node_specs[node_names[0]].get_acc_spec(device_ids[0]).acc_mem]
                    products = list(product(batch_sizes, computation_proportions, memory_proportions))
                    for product_item in products:
                        batch_size, computation_proportion, memory_proportion = \
                            product_item
                        self.mono_jobs.append(MonoJobConfig(
                            model_name=model_name,
                            node_names=node_names,
                            acc_device_ids=device_ids,
                            computation_proportion=computation_proportion,
                            memory_proportion=memory_proportion,
                            batch_size=batch_size,
                            is_train=is_train
                        ))


class MonoJobConfig:
    def __init__(self,
                 model_name: str,
                 node_names: List[str],
                 acc_device_ids: List[int],
                 computation_proportion: int,
                 memory_proportion: int,
                 batch_size: int,
                 is_train: bool
                 ):
        self.model_name: str = model_name
        assert len(node_names) == len(acc_device_ids), "node_names与acc_device_ids应一一对应，每个device_id代表一个worker"
        self.worker_count = len(node_names)
        self.node_names: List[str] = node_names
        self.acc_device_ids: List[int] = acc_device_ids
        self.computation_proportion: int = computation_proportion
        self.memory_proportion: int = memory_proportion
        self.batch_size: int = batch_size
        self.is_train: bool = is_train

    def get_session_id(self) -> str:
        c = Config()
        node_acc_reprs = []
        for i, node_name in enumerate(self.node_names):
            node_spec = c.node_specs[node_name]
            acc_names = []
            for acc_device_id in self.acc_device_ids:
                acc_names.append(node_spec.get_acc_spec(acc_device_id).acc_name)
            acc_repr = "_".join([f"{acc_names[i]}_{did}" for i, did in enumerate(self.acc_device_ids)])
            node_acc_repr = f"{node_name}_{acc_repr}"
            node_acc_reprs.append(node_acc_repr)
        node_acc_reprs_str = "_".join(node_acc_reprs)
        return f"mono_{self.model_name}_{train_or_inference(self.is_train)}_{node_acc_reprs_str}"

    def generate_name(self, worker_id: int, uuid_value: str) -> str:
        return f"profile-{self.model_name}-{self.batch_size}-{train_or_inference(self.is_train)}-{worker_id}-{uuid_value}".lower()

    def __str__(self) -> str:
        return f"mono_job: {self.model_name}-batch-size-{self.batch_size}-{train_or_inference(self.is_train)}"


class Submitter:
    @staticmethod
    def preflight_check():
        core_api = client.CoreV1Api()
        logging.info("Preflight checking...")
        # check & create namespace
        namespaces = core_api.list_namespace()
        namespaces = [ns.metadata.name for ns in namespaces.items]
        if NAMESPACE in namespaces:
            logging.info(f"Namespace {NAMESPACE} exists.")
        else:
            namespace_metadata = client.V1ObjectMeta(name=NAMESPACE)
            core_api.create_namespace(
                client.V1Namespace(metadata=namespace_metadata)
            )
            logging.info(f"Created namespace {NAMESPACE}.")

    @staticmethod
    def create_profiling_containers(mono_job_config: MonoJobConfig, uuids: List[str]) -> Dict[str, client.V1Container]:
        c = Config()
        assert len(mono_job_config.node_names) == len(mono_job_config.acc_device_ids)
        node_name_to_container = dict()
        for i, zipped in enumerate(zip(mono_job_config.node_names, mono_job_config.acc_device_ids)):
            node_name, acc_device_id = zipped
            python_script = [
                "python",
                "profiler.py",
                "--session-id",
                mono_job_config.get_session_id(),
                "--data-collector-url",
                c.options.data_collector_receive_url,
                "--process-group-backend",
                c.options.process_group_backend,
                "--model",
                mono_job_config.model_name,
                "--duration-sec",
                str(c.options.profile_duration_sec),
                "--batch-size",
                str(mono_job_config.batch_size),
                "--train" if mono_job_config.is_train else "--inference",
                "--computation-proportion",
                str(mono_job_config.computation_proportion),
                "--master-addr",
                c.node_specs[mono_job_config.node_names[0]].ip_addr,
                "--master-port",
                str(c.options.dist_master_port),
                "--world-size",
                str(len(mono_job_config.acc_device_ids)),
                "--local-rank",
                "0",  # local rank 总是0：因为kubernetes会为该pod分配单独的GPU，在容器中的视角，总是只有一个GPU
                "--rank",
                str(i),
            ]
            python_script_arg = " ".join(python_script)
            container_name = mono_job_config.generate_name(i, uuids[i])
            container = client.V1Container(
                image=c.options.image,
                name=container_name,
                image_pull_policy=c.options.pull_policy,
                args=["-c", python_script_arg],
                resources=client.V1ResourceRequirements(
                    limits={
                        "tencent.com/vcuda-core": mono_job_config.computation_proportion,
                        "tencent.com/vcuda-memory": mono_job_config.memory_proportion
                    },
                    requests={
                        "tencent.com/vcuda-core": mono_job_config.computation_proportion,
                        "tencent.com/vcuda-memory": mono_job_config.memory_proportion
                    }
                )
            )

            logging.info(
                f"Created container with name: {container.name}, "
                f"image: {container.image} and args: {container.args}"
            )
            node_name_to_container[node_name] = container

        return node_name_to_container

    @staticmethod
    def create_profiling_pod_templates(mono_job_config: MonoJobConfig, uuids: List[str]) -> List[
        client.V1PodTemplateSpec]:
        node_name_to_container = Submitter.create_profiling_containers(mono_job_config, uuids)
        pod_templates = list()
        for i, node_name in enumerate(mono_job_config.node_names):
            pod_name = mono_job_config.generate_name(i, uuids[i])
            pod_template = client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    node_name=node_name,
                    restart_policy="Never",
                    host_network=True,
                    host_ipc=True,
                    containers=[node_name_to_container[node_name]]
                ),
                metadata=client.V1ObjectMeta(
                    name=pod_name,
                    labels={"app": pod_name}
                ),
            )
            pod_templates.append(pod_template)
        return pod_templates

    @staticmethod
    def create_profiling_jobs(mono_job_config: MonoJobConfig) -> List[client.V1Job]:
        uuids = [shortuuid.uuid().__str__() for _ in range(mono_job_config.worker_count)]
        pod_templates = Submitter.create_profiling_pod_templates(mono_job_config, uuids)
        jobs = list()
        for i, pod_template in enumerate(pod_templates):
            job_name = mono_job_config.generate_name(i, uuids[i])
            metadata = client.V1ObjectMeta(name=job_name, namespace=NAMESPACE)
            job = client.V1Job(
                api_version="batch/v1",
                kind="Job",
                metadata=metadata,
                spec=client.V1JobSpec(backoff_limit=0, template=pod_template),
            )
            jobs.append(job)
        return jobs

    @staticmethod
    def run_to_terminated(jobs: List[client.V1Job]):
        try:
            batch_api = client.BatchV1Api()

            def fail_clear():
                for j in jobs:
                    batch_api.delete_namespaced_job(j.metadata.name, NAMESPACE)

            job_names = list()
            for job in jobs:
                job_names.append(job.metadata.name)
                job_response = batch_api.create_namespaced_job(NAMESPACE, job)
                print(f"job {job.metadata.name} created, status={job_response.status}")
            c = Config()
            logging.info(f"submitted jobs: {job_names}")
            maximum_waiting_duration = 2 * c.options.profile_duration_sec
            logging.info(f"waiting jobs to finish in {maximum_waiting_duration} seconds")
            FAIL = -1
            SUCCEED = 1
            UNKNOWN = 0
            job_statuses = defaultdict(lambda: UNKNOWN)
            start_waiting = time.time()
            fail_reason = None
            while True:
                time.sleep(1)
                for job in jobs:
                    job_status = batch_api.read_namespaced_job_status(job.metadata.name, NAMESPACE)
                    if not isinstance(job_status, client.V1Job):
                        job_statuses[job.metadata.name] = UNKNOWN
                        continue
                    if job_status.status.succeeded is not None and job_status.status.succeeded > 0:
                        job_statuses[job.metadata.name] = SUCCEED
                    elif job_status.status.failed is not None and job_status.status.failed > 0:
                        job_statuses[job.metadata.name] = FAIL
                    else:
                        job_statuses[job.metadata.name] = UNKNOWN
                if set(job_statuses.values()) == {SUCCEED}:
                    break
                if set(job_statuses.values()) == {FAIL}:
                    fail_reason = "jobs are all failed"
                    break
                if time.time() - start_waiting > maximum_waiting_duration:
                    fail_reason = f"timeout for {maximum_waiting_duration}"
                    break
            done_time = time.time()

            if fail_reason is not None:
                logging.error(f"failed running jobs: {job_names}")
                logging.error(f"reason: {fail_reason}")
                fail_clear()
            else:
                logging.info(f"jobs are succeeded: {job_names}")
                logging.info(f"final execution time: {done_time - start_waiting}")
        except KeyboardInterrupt:
            sys.exit()
        except:
            traceback.print_exc()
            logging.error("exception encountered, skipping")


def parse_args_init_config():
    parser = argparse.ArgumentParser(description='PyTorch Deep Learning Profiler Job Submitter')
    parser.add_argument('--config-path',
                        type=str,
                        required=False,
                        help="configuration file path")
    parser.add_argument('--config-text',
                        type=str,
                        required=False,
                        help="configuration file path")
    args = parser.parse_args()
    if args.config_path is None and args.config_text is None:
        assert False, "one of config-path and config-text must be specified"
    if args.config_text is not None:
        Config().parse(json.loads(args.config_text))
    else:
        with open(args.config_path, 'r') as f:
            Config().parse(json.load(f))


def do_test():
    parse_args_init_config()
    Submitter.preflight_check()
    c = 0
    for mono_job_config in Config().mono_jobs:
        profiling_jobs = Submitter.create_profiling_jobs(mono_job_config)
        for j in profiling_jobs:
            print("job: ", j.metadata.name, j.to_str())
        c += 1
        if c > 5:
            break


def main():
    parse_args_init_config()
    Submitter.preflight_check()
    for mono_job_config in Config().mono_jobs:
        profiling_jobs = Submitter.create_profiling_jobs(mono_job_config)
        session_id = mono_job_config.get_session_id()
        logging.info(f"start profiling for session_id: {session_id}, job: {str(mono_job_config)}")
        Submitter.run_to_terminated(profiling_jobs)
        logging.info(f"end profiling for session_id: {session_id}, job: {str(mono_job_config)}")


if __name__ == '__main__':
    # do_test()
    main()
