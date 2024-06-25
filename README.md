# Transfer Learning in Plasma Catalysis

This is the codebase developed from the Open Catalysis Project (OCP), for transfer learning from thermal catalysis to plasma catalysis. The installation, usage, etc of the code remain the same as in the [`OCP`](https://github.com/wwwccttoo/ocp/blob/main/README_OCP.md)

## Features

The main added features are the model and dataloader for plasma catalysis. They can be found in:
* [`plasma_v2`](https://github.com/wwwccttoo/ocp/ocpmodels/datasets/plasma_dataset_v2.py) The dataloader class for plasma catalysis

* [`equiformer_v2_plasma`](https://github.com/wwwccttoo/ocp/ocpmodels/models/equiformer_v2_plasma) The model used for Task1 (Transfer Learning from Thermal Catalysis to Plasma Catalysis for Single Metal Atoms) and Task3 (Transfer Learning from Single Atoms to Metal Clusters)
* [`gemnet_equiformer_v2_newdist`](https://github.com/wwwccttoo/ocp/ocpmodels/models/gemnet_equiformer_v2_newdist) The model used for Task2 (Interpretable Transfer Learning to Elucidate the Role of Surface Charge)


## Acknowledgements

This project uses code adapted from https://github.com/FAIR-Chem/fairchem (Yes, they renamed it), which is available under MIT license. We thank the original authors for their work.


## Data
We provide the link for all the datasets and training configs we used. Additionally, all the checkpoints for the trained model can be found in the same link.
https://drive.google.com/drive/folders/1mCco444-XpJ7yrezEqb7MqweQ2QXIFbK?usp=drive_link


## Rights
(c) 2024 Mesbah Lab

Questions regarding this code may be directed to ketong_shao (at) berkeley.edu
