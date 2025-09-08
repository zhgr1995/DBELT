# DBELT
code for paper (DBELT: Difficulty-Driven Dual-Branch Expert Framework for Long-Tailed Recognition)

DBELT is a dual-branch mixture-of-experts framework designed for **long-tailed recognition**.  
It dynamically allocates computation based on sample difficulty (using prediction entropy as cognitive load),  
combining a **Universal branch** (uniform sampling) and a **Resampling branch** (difficulty-aware sampling).  
A Barlow Twins loss reduces redundancy, and a ridge regression fusion adaptively integrates both branches at inference.

---

## Datasets

We evaluate DBELT on four standard **long-tailed benchmarks**:

- **CIFAR-10-LT** (imbalance factors: 10, 50, 100)  
- **CIFAR-100-LT** (imbalance factors: 10, 50, 100)  
- **ImageNet-LT** (1,000 classes)  
- **Places-LT** (365 classes)

These datasets cover both small-scale (CIFAR) and large-scale (ImageNet/Places) long-tailed recognition.

---

## Baseline Methods for Comparison

DBELT is benchmarked against multiple representative methods across four categories:

- **Multi-branch models**: BBN, RIDE  
- **Fusion-based methods**: Logit Adjustment (LA)  
- **Sampling / reweighting**: LDAM-DRW  
- **Uncertainty & routing**: RIDE, V-MoE

---

## Experimental Platform

All experiments were conducted on a **single workstation** with:

- GPU: NVIDIA GeForce **RTX 3060 (12 GB)**  
- CPU: Intel **Core i5-12490F**  
- RAM: 32 GB  
- Software: Python 3.12, PyTorch 2.1, CUDA 12.2, cuDNN 8.9  
- Mixed Precision: AMP enabled during training

This setup shows DBELT can be trained efficiently without large-scale clusters.

---

## Code Release

⚠️ **Note**: Since the paper is still under review and not officially published,  
we currently release only the **main experimental code** (core framework and training scripts).  
Additional modules (e.g., ablation scripts, dataset preprocessing details) will be updated gradually.

Stay tuned for updates in this repository.

---



# DBELT: 基于难度驱动的双分支专家网络

DBELT 是一个面向 **长尾识别** 的双分支专家框架。  
它利用预测熵作为“认知负荷”指标，在有限计算预算下根据样本难度自适应分配计算量。  
模型包含 **U 分支**（均匀采样训练）和 **R 分支**（难例重采样训练），通过 Barlow Twins 损失减少冗余，并在推理阶段使用 **岭回归融合**自适应整合两个分支的输出。

---

## 使用的数据集

论文在四个典型 **长尾基准数据集** 上进行评估：

- **CIFAR-10-LT** （不平衡因子 10、50、100）  
- **CIFAR-100-LT** （不平衡因子 10、50、100）  
- **ImageNet-LT** （1000 类）  
- **Places-LT** （365 类）

覆盖了小规模（CIFAR）和大规模（ImageNet/Places）的长尾识别场景。

---

## 对比方法

DBELT 与多种代表性方法进行了对比，主要包括四类：

- **多分支模型**：BBN, RIDE  
- **特征/输出融合方法**：Logit Adjustment (LA)  
- **采样与重加权**：LDAM-DRW  
- **不确定性与路由**：RIDE-routing, V-MoE

---

## 实验平台配置

所有实验均在一台**单机工作站**上完成：

- GPU：NVIDIA GeForce **RTX 3060 (12 GB)**  
- CPU：Intel **Core i5-12490F**  
- 内存：32 GB  
- 软件环境：Python 3.12, PyTorch 2.1, CUDA 12.2, cuDNN 8.9  
- 训练开启自动混合精度（AMP）

表明 DBELT 在中等硬件资源下即可高效训练，无需大型集群。

---

## 代码发布说明

⚠️ **注意**：由于论文尚未正式发表，目前仅开源了 **主实验代码**（核心框架与训练脚本）。  
其他模块（如消融实验脚本、数据集预处理细节）将会在后续逐步更新。  

请持续关注本仓库的更新。
