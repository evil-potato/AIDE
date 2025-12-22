
GPU_NUM=1
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29512


DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
#modified for torch 2.0 distributed launch
# DISTRIBUTED_ARGS="
#     --nproc_per_node $GPU_NUM \
#     --nnodes $WORLD_SIZE \
#     --node-rank $RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
# "

PY_ARGS=${@:1}  # Any other arguments 

python -m torch.distributed.launch $DISTRIBUTED_ARGS main_finetune.py \
    --model AIDE \
    --batch_size 32 \
    --blr 1e-4 \
    --epochs 20 \
    ${PY_ARGS}
#modified for torch 2.0 distributed launch
# python -m torch.distributed.run $DISTRIBUTED_ARGS main_finetune.py \
#     --model AIDE \
#     --batch_size 32 \
#     --blr 1e-4 \
#     --epochs 20 \
#     ${PY_ARGS}
