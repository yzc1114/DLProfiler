import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from itertools import product
from typing import List, Dict, Optional, Union, Tuple

import shortuuid
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
        self.colocate_jobs: Optional[List['ColocateJobConfig']] = None

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
        mono_job_specs_raw = job_specs_raw.get("mono", None)

        def parse_mono():
            self.mono_jobs = list()
            for model_name, mono_job_spec_raw in mono_job_specs_raw.items():
                assert model_name in model_names, f"wrong model name {model_name}"
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

        if mono_job_specs_raw is not None:
            parse_mono()
        colocate_job_spec_raw = job_specs_raw.get("colocate", None)
        self.colocate_jobs = list()

        def parse_colocate():
            for inn_node_name, acc_device_id_to_mono_jobs in colocate_job_spec_raw.items():
                for inn_acc_device_id, colocate_mono_jobs_list in acc_device_id_to_mono_jobs:
                    acc_spec = self.node_specs[inn_node_name].get_acc_spec(inn_acc_device_id)
                    acc_mem = acc_spec.acc_mem
                    for colocate_mono_jobs in colocate_mono_jobs_list:
                        total_products = list()
                        for i, mono_job in enumerate(colocate_mono_jobs):
                            batch_sizes = mono_job["batch_sizes"]
                            computation_proportions = mono_job["computation_proportions"]
                            memory_proportions = mono_job["memory_proportions"]
                            products = list(product([i], batch_sizes, computation_proportions, memory_proportions))
                            total_products.append(products)
                        combined_total_products = product(total_products)
                        for combined_total_product in combined_total_products:
                            total_computation_proportion = 0
                            total_memory_proportion = 0
                            for mono_job_product in combined_total_product:
                                _, _, computation_proportion, memory_proportion = mono_job_product
                                total_computation_proportion += computation_proportion
                                total_memory_proportion += memory_proportion
                            if total_computation_proportion > 100 or total_memory_proportion > acc_mem:
                                # skip since over subscript computation or memory
                                continue
                            generated_colocate_mono_jobs = list()
                            for mono_job_product in combined_total_product:
                                job_idx, batch_size, computation_proportion, memory_proportion = mono_job_product
                                mono_job = colocate_mono_jobs[job_idx]
                                model_name = mono_job["model_name"]
                                is_train = mono_job["is_train"]
                                generated_colocate_mono_jobs.append(
                                    MonoJobConfig(
                                        model_name=model_name,
                                        node_names=[inn_node_name],
                                        acc_device_ids=[inn_acc_device_id],
                                        computation_proportion=computation_proportion,
                                        memory_proportion=memory_proportion,
                                        batch_size=batch_size,
                                        is_train=is_train)
                                )
                            self.colocate_jobs.append(ColocateJobConfig(
                                node_name=inn_node_name,
                                acc_device_id=inn_acc_device_id,
                                mono_job_configs=generated_colocate_mono_jobs
                            ))

        if colocate_job_spec_raw is not None:
            parse_colocate()


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
            acc_device_id = self.acc_device_ids[i]
            acc_name = node_spec.get_acc_spec(acc_device_id).acc_name
            acc_repr = f"{acc_name}_{acc_device_id}"
            node_acc_repr = f"{node_name}_{acc_repr}"
            node_acc_reprs.append(node_acc_repr)
        node_acc_reprs_str = "_".join(node_acc_reprs)
        return f"mono_{self.model_name}_{train_or_inference(self.is_train)}_{node_acc_reprs_str}"

    def generate_name(self, worker_id: int, uuid_value: str) -> str:
        return f"profile-{self.model_name}-{self.batch_size}-{train_or_inference(self.is_train)}-{worker_id}-{uuid_value}".lower()

    def __str__(self) -> str:
        return f"mono_job: {self.model_name}-batch-size-{self.batch_size}-{train_or_inference(self.is_train)}"


class ColocateJobConfig:
    def __init__(self,
                 node_name: str,
                 acc_device_id: int,
                 mono_job_configs: List['MonoJobConfig']
                 ):
        self.node_name: str = node_name
        self.acc_device_id: int = acc_device_id
        self.mono_job_configs: List['MonoJobConfig'] = mono_job_configs

    def get_session_id(self) -> str:
        c = Config()
        node_spec = c.node_specs[self.node_name]
        node_acc_desc = f"{self.node_name}-{node_spec.get_acc_spec(self.acc_device_id).acc_name}-{self.acc_device_id}"
        model_desc_list = list()
        for mono_job_config in self.mono_job_configs:
            model_desc_list.append(
                f"{mono_job_config.model_name}_{train_or_inference(mono_job_config.is_train)}_comp_{mono_job_config.computation_proportion}")
        model_desc_str = "_".join(model_desc_list)
        return f"colocate_{node_acc_desc}_{model_desc_str}"

    def __str__(self) -> str:
        return f"colocate_job: {''.join([str(j) for j in self.mono_job_configs])}"


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
    def create_profiling_containers(mono_job_config: MonoJobConfig, uuids: List[str]) -> List[
        Tuple[str, client.V1Container]]:
        c = Config()
        assert len(mono_job_config.node_names) == len(mono_job_config.acc_device_ids)
        containers: List[Tuple[str, client.V1Container]] = list()
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
            computation_proportion = mono_job_config.computation_proportion
            if computation_proportion == 100:
                computation_proportion = 99
            container = client.V1Container(
                image=c.options.image,
                name=container_name,
                image_pull_policy=c.options.pull_policy,
                command=["/bin/bash"],
                env=[client.V1EnvVar(name="_", value=""), client.V1EnvVar(name="SHLVL", value="")],
                args=["-c", python_script_arg],
                resources=client.V1ResourceRequirements(
                    limits={
                        "tencent.com/vcuda-core": computation_proportion,
                        "tencent.com/vcuda-memory": mono_job_config.memory_proportion
                    },
                    requests={
                        "tencent.com/vcuda-core": computation_proportion,
                        "tencent.com/vcuda-memory": mono_job_config.memory_proportion
                    }
                )
            )

            logging.info(
                f"Created container with name: {container.name}, "
                f"image: {container.image} and args: {container.args}"
            )
            containers.append((node_name, container))

        return containers

    @staticmethod
    def create_profiling_pod_templates(mono_job_config: MonoJobConfig, uuids: List[str]) -> List[
        client.V1PodTemplateSpec]:
        containers = Submitter.create_profiling_containers(mono_job_config, uuids)
        pod_templates = list()
        for i, node_name_container in enumerate(containers):
            node_name, container = node_name_container
            pod_name = mono_job_config.generate_name(i, uuids[i])
            pod_template = client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    node_name=node_name,
                    restart_policy="Never",
                    host_network=True,
                    containers=[container]
                ),
                metadata=client.V1ObjectMeta(
                    name=pod_name,
                    labels={"app": pod_name}
                ),
            )
            pod_templates.append(pod_template)
        return pod_templates

    @staticmethod
    def create_profiling_jobs(job_config: Union[MonoJobConfig, ColocateJobConfig]) -> List[client.V1Job]:
        if isinstance(job_config, MonoJobConfig):
            return Submitter.create_profiling_mono_jobs(job_config)
        elif isinstance(job_config, ColocateJobConfig):
            return Submitter.create_profiling_colocate_jobs(job_config)

    @staticmethod
    def create_profiling_mono_jobs(mono_job_config: MonoJobConfig) -> List[client.V1Job]:
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
    def create_profiling_colocate_jobs(colocate_job_config: ColocateJobConfig) -> List[client.V1Job]:
        jobs = list()
        for mono_job_config in colocate_job_config.mono_job_configs:
            jobs.extend(Submitter.create_profiling_mono_jobs(mono_job_config))
        return jobs

    @staticmethod
    def run_to_terminated(jobs: List[client.V1Job]):
        try:
            batch_api = client.BatchV1Api()

            def fail_clear():
                for j in jobs:
                    batch_api.delete_namespaced_job(j.metadata.name, NAMESPACE)

            job_names = list()
            has_more_than_one = len(jobs) > 0
            for job in jobs:
                job_names.append(job.metadata.name)
                job_response = batch_api.create_namespaced_job(NAMESPACE, job)
                logging.info(f"job {job.metadata.name} created, status={job_response.status}")
                if has_more_than_one:
                    # 腾讯的bug：两个任务不能同时启动，否则无法load cuda库
                    time.sleep(15)
                    logging.info("waiting for 15 seconds due to tencent bug...")
            c = Config()
            logging.info(f"submitted jobs: {job_names}")
            maximum_waiting_duration = 3 * c.options.profile_duration_sec
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
                # fail_clear()
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
    c = 0
    for colocate_job_config in Config().colocate_jobs:
        profiling_jobs = Submitter.create_profiling_jobs(colocate_job_config)
        for j in profiling_jobs:
            print("job: ", j.metadata.name, j.to_str())
        c += 1
        if c > 5:
            break


def main():
    parse_args_init_config()
    Submitter.preflight_check()

    def profile_for(job_type, jobs):
        for job_config in jobs:
            profiling_jobs = Submitter.create_profiling_jobs(job_config)
            session_id = job_config.get_session_id()
            logging.info(f"start profiling for {job_type} job session_id: {session_id}, job: {str(job_config)}")
            Submitter.run_to_terminated(profiling_jobs)
            logging.info(f"end profiling for {job_type} job session_id: {session_id}, job: {str(job_config)}")
            time.sleep(5)

    profile_for("mono", Config().mono_jobs)
    profile_for("colocate", Config().colocate_jobs)


if __name__ == '__main__':
    # do_test()
    main()
