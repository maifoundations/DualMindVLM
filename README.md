<div align="center">

<h1> Learning to Think Fast and Slow for Visual Language Models </h1>

<h5 align="center"> 

<a href='https://arxiv.org/pdf/2511.16670'>
  <img src='https://img.shields.io/badge/Paper-Arxiv-red'>
</a>

<a href='https://huggingface.co/maifoundations/DualMindVLM'>
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-blue'>
</a>

</h5>
</div>

## 💡 Overview

We introduce **DualMindVLM**, a dual-mode thinking VLM that can **automatically switch between fast and slow thinking modes** based on the difficulty level of the problem. DualMindVLM is optimized using a simple RL approach built only on question–answer pairs. The approach consists of two stages: The first stage utilizes the output length variation of the pretrained VLM to assign each sample a thinking mode label. The second stage develops dual-mode thinking in the model through GRPO-based reinforcement learning, where half the sampled candidates are guided by the assigned label. Despite its simplicity, DualMindVLM significantly outperforms the base model and achieves performance on par with state-of-the-art visual reasoning models, while maintaining exceptionally high token efficiency.

---

## 🚀 Release Progress

| Component | Status | Notes |
|-----------|--------|-------|
| 🧩 **Model** | ✔️ Released | Available on 🤗 HuggingFace |
| ⚙️ **Inference + Evaluation Code** | 🕒 Coming Soon | vLLM-based inference, string-matching evaluation |
| 📚 **Dataset** | 🕒 Coming Soon | Automatically labeled Fast/Slow training set |
| 🔥 **Training Code** | 🕒 Coming Soon | GRPO-based training framework |

---

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{lin2025dualmindvlm,
  title     = {Learning to Think Fast and Slow for Visual Language Models},
  author    = {Chenyu Lin and Cheng Chi and Jinlin Wu and Sharon Li and Kaiyang Zhou},
  journal   = {arXiv preprint arXiv:2511.16670},
  year      = {2025}
}
