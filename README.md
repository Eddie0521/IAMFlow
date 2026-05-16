<p align="center" style="border-radius: 10px">
  <img src="assets/icon+name.png" width="50%" alt="logo"/>
</p>

# <div align="center" >Advancing Narrative Long Video Generation via Training-Free Identity-Aware Memory<div align="center">

<div align="center">
  <p>
    <a href="https://eddie0521.github.io/">Jinzhuo Liu</a><sup>1</sup>,
    <a href="https://zhangzjn.github.io">Jiangning Zhang</a><sup>1<a href="mailto:186368@zju.edu.cn">✉</a></sup>,
    <a href="https://github.com/Rinke02">Wencan Jiang</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=xiK4nFUAAAAJ&hl=zh-CN">Yabiao Wang</a><sup>2</sup>,
    <a href="https://dk-liang.github.io/">Dingkang Liang</a><sup>3</sup>,
    <a href="https://scholar.google.com/citations?user=m3KDreEAAAAJ&hl=en">Zhucun Xue</a><sup>1</sup>,
    <a href="https://yiranran.github.io/">Ran Yi</a><sup>4</sup>,
    <a href="https://person.zju.edu.cn/yongliu">Yong Liu</a><sup>1</sup>
  </p>
  <p>
    <sup>1</sup>Zhejiang University, &nbsp;&nbsp;
    <sup>2</sup>Tencent Youtu Lab, &nbsp;&nbsp;
    <sup>3</sup>Huazhong University of Science and Technology,<br>
    <sup>4</sup>Shanghai Jiao Tong University &nbsp;&nbsp;
    <sup><a href="mailto:186368@zju.edu.cn">✉</a></sup>Corresponding author
  </p>
</div>
<p align="center">
  <a href="https://eddie0521.github.io/projects/iamflow/"><img src="https://img.shields.io/badge/Project-Page-Green"></a>
  &nbsp;
  <img src="https://img.shields.io/static/v1?label=arXiv&message=Coming%20Soon&color=red&logo=arxiv">
  &nbsp;
  <a href="https://huggingface.co/Eddie0521/IAMFlow-FP8"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange"></a>
</p>

## 🔥 Updates

- __[2026.05.15]__: We release the [github repo](https://github.com/Eddie0521/IAMFlow), the [project page](https://eddie0521.github.io/projects/iamflow/), the quantized [model checkpoints](https://huggingface.co/Eddie0521/IAMFlow-FP8), and the [NarraStream-Bench](https://github.com/Eddie0521/NarraStream-Bench). The arxiv Version is coming soon.

## 📷 Introduction
💡**TL;DR:** 
 IAMFlow uses explicit identity-aware memory to keep identities consistent across evolving narrative prompts, achieving faster and stronger long video generation on NarraStream-Bench.


## ✨ Highlights
1. We introduce **IAMFlow**, a training-free identity-aware memory framework that explicitly organizes historical information around persistent entities and attributes, enabling reliable identity preservation across evolving prompt transitions.
2. We design a systematic inference acceleration pipeline to make the framework computationally practical, combining asynchronous visual verification, adaptive prompt transition, and model quantization to preserve long-term consistency without sacrificing generation speed.
3. We introduce **NarraStream-Bench**, a modern benchmark suite for assessing long-term consistency in narrative streaming video generation. Extensive experiments and ablation studies demonstrate that IAMFlow achieves superior performance across various metrics while enabling more efficient inference.

## 🛠️ Installation
### 1. Install Requirements

```
git clone git@github.com:Eddie0521/IAMFlow.git
cd IAMFlow
conda create -n iamflow python=3.12 -y
conda activate iamflow

# Install PyTorch first according to your CUDA environment.
python -m pip install torch==2.9.1 torchvision==0.24.1
python -m pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### 2. Download Checkpoints
Download models using hf:
``` sh
pip install "huggingface_hub[cli]"
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir pretrained/Wan2.1-T2V-1.3B
hf download Eddie0521/IAMFlow --local-dir pretrained/iamflow_models
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir pretrained/Qwen3-VL-2B-Instruct
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir pretrained/Qwen3-4B-Instruct-2507
```

## 🔑 Inference
We deploy DiT, TextEncoder, and LLM on one GPU, while VAE and VLM are deployed on another GPU.

```sh
bash ./scripts/run_iamflow.sh
```


## 📏 Evaluation & Benchmark
See the [NarraStream-Bench](https://github.com/Eddie0521/NarraStream-Bench).

## 🤗 Acknowledgement
- [MemFlow](https://github.com/KlingAIResearch/MemFlow): the codebase we built upon. Thanks for their wonderful work.
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing): the algorithm we built upon. Thanks for their wonderful work.
- [Wan](https://github.com/Wan-Video/Wan2.1): the base model we built upon. Thanks for their wonderful work.

## 🌟 Citation
Please leave us a star 🌟 and cite our paper if you find our work helpful.

```
Coming Soon
```
