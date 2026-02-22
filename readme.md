# CAGCL: A Community-Aware Graph Contrastive Learning Model for Social Bot Detection

Kaihang Wei, Min Teng, Haotong Du, Songxin Wang, Jinhe Zhao, Chao Gao, A community-aware graph contrastive learning model for social bot detection, Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM25), 2025, 3282-3291

## Abstract
Malicious social bot detection is vital for social network security. While graph neural networks (GNNs) based methods have improved performance by modeling structural information, they often overlook latent community structures, resulting in homogeneous node representations. 
Leveraging community structures, which capture discriminative group-level patterns, is therefore essential for more robust detection.
In this paper, we propose a new Community-Aware Graph Contrastive Learning  (CAGCL) framework for enhanced social bot detection. 
Specifically, CAGCL first exploits the latent community structures to uncover the potential group-level patterns. 
Then, a dual-perspective community enhancement module is proposed, 
which strengthens the structural awareness and reinforces topological consistency within communities, 
thereby enabling more distinctive node representations and deeper intra-community message passing. 
Finally, a community-aware contrastive learning module is proposed, which considers nodes within the same community as positive pairs and those from different communities as negative pairs, enhancing the discriminability of node representations. 
Extensive experiments conducted on multiple benchmark datasets demonstrate that CAGCL consistently outperforms state-of-the-art baselines. 

## 🚀 Quick Start

This project supports three main workflows: generating community labels, training the model, and reproducing the reported results. The detailed instructions for each step are provided below.

---

### 1️⃣ Generate Community Labels

To generate the required `community_labels.npy` file, run the following commands:

```bash
# Generate community labels for the Cresci15 dataset
python danmf_cresci15.py

# Generate community labels for the TwiBot-20 dataset
python danmf_twibot20.py

# Generate community labels for the MGTAB dataset
python danmf_mgtab.py
```

The generated files will be saved in:
```
outputs/cresci15/community_labels.npy
outputs/twibot20/community_labels.npy
outputs/mgtab/community_labels.npy
```

> ⚠️ If you only want to reproduce the reported results, you can skip this step and use the pre-generated files.

---

### 2️⃣ Train the Model

If you want to train the model from scratch, run the corresponding script according to the dataset:

```bash
# Train the CAGCL model on Cresci15
python main_cresci15.py

# Train the CAGCL model on TwiBot-20
python main_twibot20.py

# Train the CAGCL model on MGTAB
python main_mgtab.py
```

During training, the program will automatically:

- Load the `community_labels.npy` file
- Save model checkpoints, logs, and evaluation results to the `outputs/` directory

> ✅ GPU will be used by default if available, otherwise the CPU will be used.

---

### 3️⃣ Reproduce Reported Results

To directly reproduce the experimental results from the paper using pretrained models, run:

```bash
# Test on Cresci15
python test_cresci15.py

# Test on TwiBot-20
python test_twibot20.py

# Test on MGTAB
python test_mgtab.py
```

These scripts will:

- Load pretrained models and community label files
- Output evaluation metrics on the test set (e.g., Accuracy, F1-score, Recall)

## Citation

If you use CAGCL in your research, please cite our paper:

```bibtex
@article{cagcl2025,
  title={CAGCL: Community-Aware Graph Contrastive Learning for Social Bot Detection},
  author={Wei, Kaihang and Teng, Min and Du, Haotong and Wang, Songxin and Zhao, Jinhe and Gao, Chao},
  journal={Proceedings of the 34nd ACM International Conference on Information and Knowledge Management},
  year={2025}
}

