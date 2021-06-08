#!/bin/bash

#Step 1 - get master ip and port 
source ~/.bashrc
INTERFACE=infinibond0
MASTER_IP=$(ip -4 addr show $INTERFACE | grep -oP "(?<=inet ).*(?=/)" 2>&1 | head -n 1)
echo "IP ADDRESS OF ROOT PROCESS  IS $MASTER_IP"
MASTER_PORT=8889

#Step 2 - Decide number of GPUs and Nodes
if [ -z ${NUM_NODES+x} ]; then
    echo "Value for NUM_NODES is not provided, defaulting to 1";
    NUM_NODES=1
else
    echo "NUM_NODES is set to '$NUM_NODES'";
fi

if [ -z ${NUM_GPUS_PER_NODE+x} ]; then
    echo "Value for NUM_NODES is not provided, defaulting to 1";
    NUM_GPUS_PER_NODE=1
else
    echo "NUM_GPUS_PER_NODE is set to '$NUM_GPUS_PER_NODE'";
fi

#Step 3 - Copy to SSD
# mpirun -n $NUM_NODES -npernode 1 -hostfile $COBALT_NODEFILE cp /lus/theta-fs0/projects/CharmRTS/charmnn/wikitext-103_bert* /raid/scratch/


NUM_GPUS=$(($NUM_GPUS_PER_NODE*$NUM_NODES))
WORLD_SIZE=$NUM_GPUS
MP_SIZE=$NUM_GPUS


#Step 4 - Set datapath, checkpount path and vocab paths....
VOCAB_DIR=~/parallel-dl-scripts/megatron/
CKP_DIR=/raid/scratch
MEGATRON_HOME=~/Megatron-LM
DATA_DIR=/lus/theta-fs0/projects/CharmRTS/charmnn/

CHECKPOINT_PATH=$CKP_DIR/checkpoints/bert
mkdir -p $CHECKPOINT_PATH

VOCAB_FILE=$VOCAB_DIR/bert-large-uncased-vocab.txt
DATA_PATH=$DATA_DIR/wikitext-103_bert_text_sentence

#megatron 2.0 paper 
#-N 256 -D 2048 -I 8192 -H 16
CONF="--num-layers 40 --hidden-size 6144 --num-attention-heads 48" #18B
#CONF="--num-layers 32 --hidden-size 4096 --num-attention-heads 32" #6B
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=2  
SEQ_LEN=128

BERT_ARGS="
        $CONF \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LEN \
       --max-position-embeddings 512 \
       --train-iters 100 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 --fp16 --initial-loss-scale 1024 --checkpoint-activations --checkpoint-num-layers 4 
       "

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 1000 \
             --eval-interval 1000 \
             --eval-iters 10
           "

export MASTER_ADDR=$MASTER_IP
export MASTER_PORT=$MASTER_PORT


cd $MEGATRON_HOME

mpirun -n $NUM_GPUS -npernode $NUM_GPUS_PER_NODE -x PATH -x MASTER_ADDR -x MASTER_PORT -hostfile $COBALT_NODEFILE -cpus-per-proc 16 -bind-to none python ./pretrain_bert.py $BERT_ARGS $OUTPUT_ARGS --save $CHECKPOINT_PATH --load $CHECKPOINT_PATH --data-path $DATA_PATH --tensor-model-parallel-size $MP_SIZE --DDP-impl local --split 100,0,0

