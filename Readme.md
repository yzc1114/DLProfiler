# DLProfiler

### data_collector.py
数据收集服务器脚本。

``` shell
sudo docker run 
--net host -v /path/to/data/dir:/data-dir -d \
somebody/dl-profiler:latest data_collector.py --data-dir-path /data-dir
```

默认运行在39911端口，可通过参数指定。

### profiler.py

测算器脚本入口。

``` shell
sudo docker run --gpus all \
    --rm --ipc=host --ulimit memlock=-1 \
    --net host --ulimit stack=67108864 -it \    # 以上为容器参数
    somebody/dl-profiler:latest profiler.py \   # 以下为脚本参数
    --session-id testsession \  #指定测算器的session，用于区分不同类别的测算
    --data-collector-url http://localhost:39911/receive \  # 发送给数据收集器的url
    --process-group-backend nccl \  # GPU worker的通信后端
    --model LSTM \  # 待测算的模型
    --duration-sec 10 \  # 测量的时间
    --batch-size 32 \  # batch size
    --inference  # --inference 或 --train 指定测算推理还是训练
```