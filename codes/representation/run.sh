#!/bin/bash

# Config
CHUNK_NUM=20
NUM_CPUS=64
DEVICE=3
BATCH_SIZE=2048
GPU_DEVICES="0 1 2 3 4 5 6 7"

# Preprocessing for representation
python representation/preprocess_for_representation.py --num_cpus $NUM_CPUS --chunk_num $CHUNK_NUM

# Generate descirption for each concept ID
for ((i=0; i<20; i++)); do
    python representation/gpt_description.py -c "$i"
done
python representation/gpt_description_merge.py
python representation/description_tokenize.py  --num_cpus $NUM_CPUS

# Yield BioBERT representation
python representation/biobert_representation.py -d $DEVICE --num_cpus $NUM_CPUS --bs $BATCH_SIZE

# Yield Description representation
python representation/description_representation_1.py --gpu-devices $GPU_DEVICES
python representation/description_representation_2.py -d $DEVICE

# Yield BioBERT+GNN representation
python representation/gnn_representation.py -d $DEVICE --num_cpus $NUM_CPUS --bs $BATCH_SIZE --rep_type biobert

# Yield Description+GNN representation
python representation/gnn_representation.py -d $DEVICE --num_cpus $NUM_CPUS --bs $BATCH_SIZE --rep_type description

# Extract neighbors of representations
python representation/extract_neighbors.py --num_cpus $NUM_CPUS

# MedTok training
python representation/medtok_train.py

