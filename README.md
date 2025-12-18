# Single-Agent Reinforcement Learning for Personalized Federated Learning (pFL)

This repository implements a **single-agent reinforcement learning (RL) framework** to control **personalization and training intensity** in a **federated learning (FL)** system under **client heterogeneity**.

The key idea is to use **one centralized RL agent at the server** to dynamically decide, **for each participating client**,  
1. **which model layers should be personalized**, and  
2. **how many local training epochs should be performed**,  

subject to constraints such as **data heterogeneity**, **computation limits**, and **communication efficiency**.

This work is motivated by the observation that **static personalization strategies** (e.g., fixed local heads, fixed local epochs) are suboptimal when clients differ significantly in data distribution and system resources.

---

## 📌 Key Contributions

- **Single-agent RL formulation** for personalized federated learning  
- **Client-specific layer personalization**, including middle-layer personalization  
- **Adaptive control of local training intensity (epochs)**  
- Explicit modeling of **client heterogeneity** (data, compute, bandwidth)  
- Clean separation between:
  - **FL optimization (cross-entropy loss)**
  - **RL policy optimization (policy gradient loss)**  
- Designed to be extensible to **non-RL baselines** (FedAvg, FedProx, FedPer, LG-FedAvg)

---

## 🧠 High-Level Architecture




- **One RL agent** shared across all clients  
- Each client receives a **personalized action**  
- Clients train locally using standard supervised learning  
- The RL agent is trained using **policy gradient methods**

---

## 📂 Repository Structure

single_agent_RL_for_pFL/
│
├── models/
│ ├── resnet18.py # CIFAR-style ResNet model
│ ├── RL_model.py # Policy network for RL agent
│
├── data_loaders/
│ ├── cifar_10_dataloader.py # Dirichlet non-IID client splits
│
├── utils/
│ ├── aggregation.py # FedAvg-style aggregation
│ ├── evaluation.py # Global & client evaluation
│
├── train_rl_federated.py # Main FL + RL training loop
├── baseline_federated.py # Non-RL FL baselines
├── config.py # Hyperparameters & settings
│
├── requirements.txt
└── README.md



---

## 🔬 Problem Setup

### Dataset
- **CIFAR-10**
  - 50,000 training images → split among clients (non-IID)
  - 10,000 test images → **used only for global evaluation**

### Non-IID Data Split
- Dirichlet distribution with concentration parameter `α`
- Smaller `α` → higher data heterogeneity

### Client Heterogeneity
Each client is assigned:
- Dataset size
- Computational capability
- Bandwidth level
- Participation frequency

These factors are included in the **RL state representation**.

---

## 🎯 RL Formulation

### State (per client)
A compact vector encoding:
- Compute capacity
- Bandwidth
- Local dataset size
- Previous validation accuracy
- Training progress

### Action (per client)
- **Layer personalization decision**
  - Which blocks are personalized vs shared
- **Training intensity**
  - Number of local epochs

### Reward
A scalar reward computed at the server level:

---

## 🔬 Problem Setup

### Dataset
- **CIFAR-10**
  - 50,000 training images → split among clients (non-IID)
  - 10,000 test images → **used only for global evaluation**

### Non-IID Data Split
- Dirichlet distribution with concentration parameter `α`
- Smaller `α` → higher data heterogeneity

### Client Heterogeneity
Each client is assigned:
- Dataset size
- Computational capability
- Bandwidth level
- Participation frequency

These factors are included in the **RL state representation**.

---

## 🎯 RL Formulation

### State (per client)
A compact vector encoding:
- Compute capacity
- Bandwidth
- Local dataset size
- Previous validation accuracy
- Training progress

### Action (per client)
- **Layer personalization decision**
  - Which blocks are personalized vs shared
- **Training intensity**
  - Number of local epochs

### Reward
A scalar reward computed at the server level:

reward = α · Δ(global test accuracy)
− β · client accuracy variance
− γ · compute / communication cost


This encourages:
- Better generalization
- Fairness across clients
- Efficient resource usage

---

## 📉 Loss Functions

### Client-side (Supervised Learning)
Standard cross-entropy loss:
\[
\mathcal{L}_{client} = \text{CrossEntropy}(y, f(x))
\]

### RL Agent (Policy Gradient)
REINFORCE-style objective:
\[
\mathcal{L}_{RL} = - \sum_{t,k} \log \pi_\theta(a_{k,t} \mid s_{k,t}) \cdot G_t
\]

where \( G_t \) is the discounted return.

---

## 🧪 Baselines Supported

The framework supports easy comparison with classical FL and pFL methods:

- **FedAvg**
- **FedProx**
- **FedPer**
- **LG-FedAvg**
- **Static personalization (fixed layers + fixed epochs)**

These baselines do **not** use RL and serve as strong reference points.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt


2. Train RL-controlled pFL
python train_rl_federated.py


3. Run baselines
python baseline_federated.py


