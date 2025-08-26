# MedRep

The official source code for MedRep. <br>

  <br/>

## Release Notes

- **April 24, 2025:** Full version of OMOP concept representations released.
- **August 15, 2025:** Graph-free version of MedRep added.
- **August 26, 2025:** DeBERTa weights pretrained with OMOP concept descriptions released.

  <br/>
## Downloads
- **MedRep (full version, 22.1GB):** [Download](https://dl.dropboxusercontent.com/scl/fi/h4j36yphpctluxdm4wo2m/concept_representation_medrep.npy?rlkey=a2awq87n4x7m7fy45hs7e2a3r&st=x1thytgc)
- **MedRep (graph-free version, 22.1GB):** [Download](https://dl.dropboxusercontent.com/scl/fi/i7h11v1707t5zbjj3ft4h/concept_representation_description.npy?rlkey=jrnpzp2cyqylprdkzggpchlz7&st=4n4rxsxn)  
- **Concept index (65.0MB):** [Download](https://dl.dropboxusercontent.com/scl/fi/n0iao7c9ftwxhr6pwl7ab/concept_idx.csv?rlkey=qubzngglvmensaq2dp5gyja3j&st=9io2obza)
- **DeBERTa weights (1.6GB):** [Download](https://dl.dropboxusercontent.com/scl/fi/wqvdv283ms5yh038cpy0p/deberta_full.tar?rlkey=7xreaoyhmm5fm7gxsreps0pho&st=avqoz23j)

  <br/>
## Abstract
Electronic health record (EHR) foundation models have been an area ripe for exploration with their improved performance in various medical tasks. Despite the rapid advances, there exists a fundamental limitation: Processing unseen medical codes out of vocabulary. This problem limits the generalizability of EHR foundation models and the integration of models trained with different vocabularies. To alleviate this problem, we propose a set of novel medical concept representations (MedRep) for EHR foundation models based on the observational medical outcome partnership (OMOP) common data model (CDM). For concept representation learning, we enrich the information of each concept with a minimal definition through large language model (LLM) prompts and complement the text-based representations through the graph ontology of OMOP vocabulary. Our approach outperforms the vanilla EHR foundation model and the model with a previously introduced medical code tokenizer in diverse prediction tasks. We also demonstrate the generalizability of MedRep through external validation.


**Illustration of Concept Representation Learning**
<br/>
<p align="center"><img src="images/crl.png" width="800px"/></p>
<br/>


## Data Preparation
|Dataset|Details|URL|
|------|---|---|
|OMOP Vocabulary|2 Files are required: <br> - CONCEPT.csv <br> - CONCEPT_RELATIONSHIP.csv|https://athena.ohdsi.org/|
|MIMIC-IV 2.2|The original data should be converted to OMOP CDM format. <br> 8 Files are required: <br> - patients.csv (original) <br> - condition_occurrence.csv <br> - drug_exposure.csv <br> - measurement.csv <br> - procedure_occurrence.csv <br> - visit_occurrence.csv <br> - person.csv <br> - death.csv |https://physionet.org/content/mimiciv/2.2/|
|EHRSHOT|7 Files are required: <br> - condition_occurrence.csv <br> - drug_exposure.csv <br> - measurement.csv <br> - procedure_occurrence.csv <br> - visit_occurrence.csv <br> - person.csv <br> - death.csv <br>  |https://redivis.com/datasets/53gc-8rhx41kgt|

<br/>

## File Tree
Place the following files in the `usedata/representation` folder:

- `concept_idx.csv`
- `concept_representation_description.npy`
- `concept_representation_medrep.npy`

```bash
.
├── codes
├── data
│   ├── concepts
│   │   ├── CONCEPT.csv
│   │   └── CONCEPT_RELATIONSHIP.csv
│   ├── ehrshot
│   │   ├── condition_occurrence.csv
│   │   ├── death.csv
│   │   ├── drug_exposure.csv
│   │   ├── measurement.csv
│   │   ├── person.csv
│   │   ├── procedure_occurrence.csv
│   │   └── visit_occurrence.csv
│   └── mimic
│       ├── condition_occurrence.csv
│       ├── death.csv
│       ├── drug_exposure.csv
│       ├── measurement.csv
│       ├── patients.csv
│       ├── person.csv
│       ├── procedure_occurrence.csv
│       └── visit_occurrence.csv
├── usedata
│       ├── mimic
│       ├── ehrshot
│       └── representation
│           ├── concept_idx.csv
│           ├── concept_representation_description.npy
│           └── concept_representation_medrep.npy
├── results
..
```

<br/>

## Requirements
```bash
- Python 3.9.19
- torch 2.6.0
- transformers 4.49.0
- torch-geometric 2.6.1
- torch_scatter 2.1.2
- torch_sparse 0.6.18
```

<br/>

## How to Run
**Data Preprocessing**
<br/> The datasets for experiments will be generated in the `usedata` folder.
```bash
cd ./codes
bash preprocessing/run.sh
```
<br/> 

**Learning Representations**
<br/> The representations for experiments will be generated in `usedata/representation` folder.
```bash
cd ./codes
bash representation/run.sh
```
<br/> 

**Model Pretraining**
<br/> 50 epochs on MIMIC-IV using 8 NVIDIA RTX A6000 GPUs (~15 hours)
```bash
cd ./codes
python experiments/pretrain.py --model behrt --rep-type medrep --gpu-devices 0 1 2 3 4 5 6 7
```
<br/> 

**Model Finetuning** 
<br> up to 50 epochs on MIMIC-IV using 1 NVIDIA RTX A6000 GPU (~3–6 hours)
```bash
cd ./codes
python experiments/finetune.py -d 0 --model behrt --rep-type medrep --outcome MT
```
<br/> 

**Extract important features** 
<br> Important features can be extracted after finetuning as follows:
```bash
cd ./codes
python experiments/finetune.py -d 0 --model behrt --rep-type medrep --outcome MT --extract-attention-score
```

<br> Run the MedGemma-27B implementation as follows:
```bash
cd ./codes
CUDA_VISIBLE_DEVICES=0 python experiments/medgemma.py --rep medrep --outcome MT --seed 100 --ex mimic
```
