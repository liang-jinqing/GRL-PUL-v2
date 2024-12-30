# GRL-PUL

GRL-PUL: Predicting microbe-drug association based on graph representation learning and positive unlabeled learning



## Dataset

- MDAD: MADA consists of  173 microbes and 1373 drugs with 2470 known microbe-drug associations.
- aBiofilm: aBiofilm consists of 140 drugs and 1720 microbes with 2884 known microbe-drug associations.
- DrugVirus: DrugVirus consists of 95 drugs and 175 microbes with 933 known microbe-drug associations.



## Data description

- adj: interaction pairs between microbes and drugs.
- drugs: IDs and names for drugs.
- microbes/viruses: IDs and names for microbes/viruses.
- microbe_features: Microbe genome sequence attribute.
- microbe_similarity: Microbe functional similarity attribute.
- drug_features: Drug network topological attribute.
- drug_similarity: Drug integrated similarity attribute.



## Requirements

- python==3.8.0
- numpy==1.22.3
- scipy==1.8.0
- scikit-learn==1.0.2
- torch==1.8.1
- torch_scatter==2.0.8
- torch_sparse==0.6.12
- torch_cluster==1.5.9
- torch_spline_conv==1.2.1
- Pillow==9.1.0
- networkx==3.1
- tqdm==4.66.4



## Run

Run main.py to train the model and get the performance evaluations on MDAD dataset. Other parameters are set by default.