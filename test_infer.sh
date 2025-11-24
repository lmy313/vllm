
export HCCL_SOCKET_IFNAME="ens45"
export TP_SOCKET_IFNAME="ens45"
export GLOO_SOCKET_IFNAME="ens45"
export HCCL_EXEC_TIMEOUT="7200"
export HCCL_CONNECT_TIMEOUT="7200"
export HCCL_IF_BASE_PORT="23999"
export HCCL_ASYNC_ERROR_HANDLING="0"
export P2P_HCCL_BUFFSIZE="20"
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"

torchrun --nproc_per_node=8 --master_port=29500 generate_tp8.py