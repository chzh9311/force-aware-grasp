# Force-Aware 3D Contact Modeling for Stable Grasp Generation (AAAI'26)

:warning: **Note that this is currently a dirty repo with redundant code and incomplete descriptions. The sanitizing will be done soon.**

# Framework

![Framework](docs/images/framework.png)

## Requirements

Install the pytorch version with CUDA support according to your device, then install the rest by: 

```shell
pip install -r requirements.txt
```

## Data Preparation

We use GRAB dataset for training & evaluation, and HO3D objects for out-of-domain test. For GRAB dataset, please follow their instructions to do data-preprocessing; for HO3D_v2, since we only need the object templates for testing, please download the YCB video object dataset.

Then, download MANO model and extract it to data folder.

The preprocessed data structure should look like this:

```text
data
 |-- grab
 |    |-- train
 |    |-- test
 |    |-- val
 |    |-- obj_hulls
 |    |-- obj_info.npy
 |    `-- sbj_info.npy
 |
 |-- YCB_Video
 |    |-- models
 |    `-- obj_hulls
 |
 `-- mano_v1_2
      |-- models
      |-- webuser
      |-- __init__.py
      `-- LICENSE.txt
```

## Training

### Automatic labelling

Run the automatic labelling procedure for GRAB dataset by 

```shell
python scripts/run_automatic_labelling.py
```

Labels will be saved in `force_labels/grab` folder.

We use [Weights & Bias](https://wandb.ai) to track the training process. You can sign up for a free account and obtain a login key.

Suppose your login key is `<your_wandb_key>`, then you can run the training process by 

```shell
python scripts/train_vae.py --dataset grab --run_phase train --key <your_wandb_key>
```
and track the training process in `force_aware_grasp` project in your W&B dashboard.

## Evaluation

After training is done, run the test process by

```shell
python scripts/train_vae.py --dataset grab --run_phase test --key <your_wandb_key> --ckpt path/to/checkpoint
```

Optionally, you can download our pretrained checkpoint from Google Drive via [this link](https://drive.google.com/file/d/14oID0Hy4EEd6oI0wSVGSbCS2Vhjq46wZ/view?usp=sharing). In this way, you can skip the force labelling step.

Place the checkpoint in `checkpoint` folder, then run 

```shell
python scripts/train_vae.py --dataset grab --run_phase test --key <your_wandb_key> --ckpt checkpoints/epoch=99-step=47200.ckpt
```

## Citation

If you use our code, please cite us with

```text
@inproceedings{chen2026forceaware3d,
    title={Force-Aware 3D Contact Modeling for Stable Grasp Generation},
    author={Chen, Zhuo and Zhang, Zhongqun and Cheng, Yihua and Leonardis, Ales and Chang, Hyung Jin},
    booktitle={Annual AAAI Conference on Artificial Intelligence (AAAI)},
    year={2026},
}
```