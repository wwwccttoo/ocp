# Transfer Learning in Plasma Catalysis

This is the codebase developed from the Open Catalysis Project (OCP), for transfer learning from thermal catalysis to plasma catalysis. The installation, usage, etc of the code remain the same as in the [`OCP`](https://github.com/wwwccttoo/ocp/blob/main/README_OCP.md)

## Tips for Installation
From our experience, following our tested installation steps should be faster and more convenient than the official one. These installation steps work for Red Hat Enterprise Linux 9.4 (Plow). The overall installation time should be maximumly around 10 mins.

1. Create a new conda environment with python version of 3.9.18
2. Activate the conda environment and ```pip install -r env.txt``` for all the other dependencies
3. Install pytorch ```pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116```
4. Install torch extensions ```pip install torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.1 torch-spline-conv==1.2.2 torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cu116.html```
5. Install OCP ```git clone https://github.com/wwwccttoo/ocp.git```
6. Enter the cloned folder
7. ```pip install -e .```

## Training
The following is an example of training
```python3 -u -m torch.distributed.launch --nproc_per_node=4 ocp/main.py --distributed --num-gpus 4 --mode train --config-yml configs/s2ef/all/equiformer_v2_plasma/Task1_equiformer_v2_plasma_all_traj_scratch_31M.yml --amp --checkpoint checkpoints/2024-04-11-18-28-16/checkpoint.pt```

Here, 4 gpus are used. The setup according to `Task1_equiformer_v2_plasma_all_traj_scratch_31M` is used to define the model, optimization, etc. A checkpoint is used to restart the training or start the training from a pre-trained model.

Checkpoints will be automatically saved for restarting or fine-tuning.

Note: due to the number of atoms in the plasma catalysis data, a GPU of mininmal 12 GB should be used. In this training, we used 4 12 GB GPUS for a bathsize 4 training. Training for 100 epoch will take ~7 days.

## Test
The following is an example of test
```python3 -u -m torch.distributed.launch --nproc_per_node=4 ocp/main.py --distributed --num-gpus 4 --mode predict --config-yml configs/s2ef/all/equiformer_v2_plasma/Task1_equiformer_v2_plasma_all_traj_scratch_31M.yml --amp --checkpoint checkpoints/2024-04-11-18-28-16/checkpoint.pt```

The only difference here is the `train` is set to `predict`. An output of the corresponding energy and atomic forces for each of the catalysis system in the test dataset will be generated. The test dataset should be specified in the config file.

## Reproduction

We provide all the generated predictions for traning, validation, test and extrapolation. As well as the attention score collected for task2. They can be found in the `Data` section.

We also provide three .ipynb scripts that can be used to regenerate the results we put in the manuscripts. However, the data should be downloaded and the address should be reset in these scripts.

## Features

The main added features are the model and dataloader for plasma catalysis. They can be found in:
* [`plasma_v2`](https://github.com/wwwccttoo/ocp/blob/main/ocpmodels/datasets/plasma_dataset_v2.py) The dataloader class for plasma catalysis

* [`equiformer_v2_plasma`](https://github.com/wwwccttoo/ocp/blob/main/ocpmodels/models/equiformer_v2_plasma) The model used for Task1 (Transfer Learning from Thermal Catalysis to Plasma Catalysis for Single Metal Atoms) and Task3 (Transfer Learning from Single Atoms to Metal Clusters)
* [`gemnet_equiformer_v2_newdist`](https://github.com/wwwccttoo/ocp/blob/main/ocpmodels/models/gemnet_equiformer_v2_newdist) The model used for Task2 (Interpretable Transfer Learning to Elucidate the Role of Surface Charge)


## Acknowledgements

This project uses code adapted from https://github.com/FAIR-Chem/fairchem (Yes, they renamed it), which is available under MIT license. We thank the original authors for their work.


## Data
We provide the link for all the datasets and training configs we used. Additionally, all the checkpoints for the trained model can be found in the same link.
https://drive.google.com/drive/folders/1mCco444-XpJ7yrezEqb7MqweQ2QXIFbK?usp=drive_link


## Rights
(c) 2024 Mesbah Lab

Questions regarding this code may be directed to ketong_shao (at) berkeley.edu
