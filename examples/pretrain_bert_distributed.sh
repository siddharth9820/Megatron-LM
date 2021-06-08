#!/bin/bash

SUMMIT_FS_HOME=/gpfs/alpine/csc452/scratch/ssingh37/
export LC_CTYPE=en_US.UTF-8

nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch))
head=${nodes[0]}

export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=$head
export MASTER_PORT=29500 # default from torch launcher

echo "Setting env_var RANK=${RANK}"
echo "Setting env_var LOCAL_RANK=${LOCAL_RANK}"
echo "Setting env_var WORLD_SIZE=${WORLD_SIZE}"
echo "Setting env_var MASTER_ADDR=${MASTER_ADDR}"
echo "Setting env_var MASTER_PORT=${MASTER_PORT}"


MP_SIZE=$WORLD_SIZE


#Step 4 - Set datapath, checkpount path and vocab paths....
VOCAB_DIR=$SUMMIT_FS_HOME
CKP_DIR=$SUMMIT_FS_HOME
MEGATRON_HOME=~/Megatron-LM
DATA_DIR=$SUMMIT_FS_HOME

CHECKPOINT_PATH=$CKP_DIR/checkpoints/bert
rm -rf $CHECKPOINT_PATH
mkdir -p $CHECKPOINT_PATH

VOCAB_FILE=$VOCAB_DIR/bert-large-uncased-vocab.txt
DATA_PATH=$DATA_DIR/wikitext-103_bert_text_sentence

#megatron 2.0 paper 
#-N 256 -D 2048 -I 8192 -H 16


#these are all in the megatron 1.0 paper
CONF1="-N 40 -D 1536 -H 16 -I 6144"
CONF2="-N 56 -D 1920 -H 20 -I 7680"
CONF3="-N 64 -D 2304 -H 24 -I 9216"
CONF4="-N 72 -D 3072 -H 32 -I 12288"

CONF="--num-layers 72 --hidden-size 3072 --num-attention-heads 32" #8B
#CONF="--num-layers 32 --hidden-size 4096 --num-attention-heads 32" #6B

GLOBAL_BATCH_SIZE=16
MICRO_BATCH_SIZE=16  
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
       --lr-warmup-fraction .01 --fp16 --initial-loss-scale 1024 --checkpoint-activations --checkpoint-num-layers 9 
       "

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 1000 \
             --eval-interval 1000 \
             --eval-iters 10
           "


cd $MEGATRON_HOME

python ./pretrain_bert.py $BERT_ARGS $OUTPUT_ARGS --save $CHECKPOINT_PATH --load $CHECKPOINT_PATH --data-path $DATA_PATH --tensor-model-parallel-size $MP_SIZE --DDP-impl local --split 100,0,0

