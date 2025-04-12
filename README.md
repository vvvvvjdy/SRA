<h1 align="center">🎭SRA  <br>No Other Representation Component Is Needed:  <br>Diffusion Transformers Can Provide Representation
            Guidance by Themselves
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2410.06940">
    <img src="https://img.shields.io/badge/arXiv%20paper-2410.06940-b31b1b.svg" alt="arXiv Paper">
  </a>
  &nbsp;
  <a href="https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=representation-alignment-for-generation">
    <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-alignment-for-generation/image-generation-on-imagenet-256x256" alt="PWC">
  </a>
  &nbsp;
  <a href="https://www.xiaohongshu.com/user/profile/60195f8f0000000001009cc6">
    <img src="https://img.shields.io/badge/Xiaohongshu(RedNote) Profile-Dy Jiang-red" alt="RedNote Profile">
  </a>
</p>


<h3 align="center">[<a href="https://sihyun.me/REPA">project page</a>]&emsp;[<a href="http://arxiv.org/abs/2410.06940">paper</a>]</h3>
<br>

### 💥1.News
- **[2025.01.01]** We have released the paper and code for SRA! 


### 🌟2.Highlight

-  **Diffusion transformer itself to provide representation guidance:** We assume the unique  discriminative process of diffusion transformer makes it possible to provide the guidance without introducing extraneous representation component.

- **Self-Representation Alignment (SRA):** SRA aligns the output 
         latent representation of the diffusion transformers in earlier layer with higher noise to that in later layer  with lower noise to achieve self-representation enhancement.

- **Improved Performance**. SRA accelerates training and improves generation performance for both DiTs and SiTs.

### 🏡3.Environment Setup

```bash
conda create -n sra python=3.12 -y
conda activate sra
pip install -r requirements.txt
```

### 📜4.Dataset Preparation


Currently, we provide experiments for [ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data). You can place the data that you want and can specify it via `--data-dir` arguments in training scripts. \
Note that we preprocess the data for faster training. Please refer to [preprocessing guide](https://github.com/vvvvvjdy/SRA/tree/main/preprocessing) for detailed guidance.

### 🔥5.Training
Here we provide the training code for SiTs and DiTs.

##### 5.1.Training with SiT + SRA
```bash
cd SiT-SRA
accelerate launch --config_file configs/default.yaml train.py \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --resolution=256 \
  --batch-size=32 \
  --weighting="uniform" \
  --model="SiT-XL/2" \
  --block-out-s=8 \
  --block-out-t=20 \
  --t-max=0.2 \
  --output-dir="exps" \
  --exp-name="sitxl-ab820-t0.2-res256" \
  --data-dir=[YOUR_DATA_PATH]
```

Then this script will automatically create the folder in `exps` to save logs,samples, and checkpoints. You can adjust the following options:

- `--models`: Choosing from [SiT-B/2, SiT-L/2, SiT-XL/2]
- `--block-out-s`: Student's output block layer for alignment
- `--block-out-t`: Teacher's output block layer for alignment
- `--t-max`: Maximum time interval for alignment (we only use dynamic interval here)
- `--output-dir`: Any directory that you want to save checkpoints, samples, and logs
- `--exp-name`: Any string name (the folder will be created under `output-dir`)
- `--batch-size`: The local batch size (by default we use 1 node of 8 GPUs), you need to adjust this value according to your GPU number to make total batch size of 256


##### 5.2.Training with DiT + SRA
```bash
cd DiT-SRA
accelerate launch --config_file configs/default.yaml train.py \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --resolution=256 \
  --batch-size=32 \
  --model="DiT-XL/2" \
  --block-out-s=8 \
  --block-out-t=20 \
  --t-max=0.2 \
  --output-dir="exps" \
  --exp-name="ditxl-ab820-t0.2-res256" \
  --data-dir=[YOUR_DATA_PATH]
```

Then this script will automatically create the folder in `exps` to save logs and checkpoints. You can adjust the following options (others are same as above SiTs):

- `--models`: Choosing from [DiT-B/2, DiT-L/2, DiT-XL/2]



### 🌠6.Evaluation
Here we provide the generating code for SiTs and DiTs to get the samples for evaluation. (and the .npz file can be used for [ADM evaluation](https://github.com/openai/guided-diffusion/tree/main/evaluations) suite) through the following script:

##### 6.1.Sampling with SiT + SRA

You can download our pretrained model here:

| Model                   | Image Resolution | FID-50K | Inception Score | Epochs | 
|-------------------------|------------------|---------|-----------------|--------|
| [SiT-XL/2 + SRA](to do) | 256x256          | 2.06    | 270.27         | 800    |
```bash
cd SiT-SRA
bash gen.sh
```
Note that there are several options in `gen.sh` file that you need to complete:
- `SAMPLE_DIR`: Base directory to save the generated images and .npz file
- `CKPT`: Checkpoint path (This can also be your downloaded local file of the ckpt file we provide above)

##### 6.2.Sampling with DiT + SRA
```bash
cd DiT-SRA
bash gen.sh
```
### 📣7.Note

It's possible that this code may not accurately replicate the results outlined in the paper due to potential human errors during the preparation and cleaning of the code for release as well as the difference of the hardware facility. If you encounter any difficulties in reproducing our findings, please don't hesitate to inform us. 

### 🤝🏻8.Acknowledgement

This code is mainly built upon [REPA](https://github.com/sihyun-yu/REPA), [DiT](https://github.com/facebookresearch/DiT), [SiT](https://github.com/willisma/SiT) repositories. 
Thanks for their solid work!


### 💜9.Citation
If you find SRA useful, please kindly cite our paper:
```bibtex
@article{yu2024repa,
  title={Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think},
  author={Sihyun Yu and Sangkyung Kwak and Huiwon Jang and Jongheon Jeong and Jonathan Huang and Jinwoo Shin and Saining Xie},
  year={2024},
  journal={arXiv preprint arXiv:2410.06940},
}
```

