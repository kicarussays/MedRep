#!/bin/bash
NUM_CPUS=64
SEQ_LENGTH=2048
hospitals=("mimic" "ehrshot")

for HOSPITAL in "${hospitals[@]}"; do
    python preprocessing/create_trajectory.py --num_cpus $NUM_CPUS --hospital $HOSPITAL
    python preprocessing/tokenize_for_pretrain_1.py --num_cpus $NUM_CPUS --hospital $HOSPITAL --seq-length $SEQ_LENGTH

    if [[ $HOSPITAL != "ehrshot" ]]; then
        python preprocessing/tokenize_for_pretrain_2.py --num_cpus $NUM_CPUS --hospital $HOSPITAL --seq-length $SEQ_LENGTH
        python preprocessing/tokenize_for_pretrain_3.py --num_cpus $NUM_CPUS --hospital $HOSPITAL --seq-length $SEQ_LENGTH
    fi

    python preprocessing/labeling.py --num_cpus $NUM_CPUS --hospital $HOSPITAL
    python preprocessing/tokenize_for_finetune.py --num_cpus $NUM_CPUS --hospital $HOSPITAL
done
