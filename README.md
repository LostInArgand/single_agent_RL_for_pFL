# Single-Agent Reinforcement Learning for Personalized Federated Learning (pFL)

This repository implements a **single-agent reinforcement learning (RL) framework** for **personalized federated learning (pFL)** under **data and system heterogeneity**.

The core idea is to use **one centralized RL agent at the server** to dynamically control, **for each participating client**:

1. **Which model layers are shared vs personalized**
2. **How much local training to perform (number of local epochs)**

This allows the system to adapt to heterogeneous clients with different data distributions, computational capabilities, and bandwidth constraints, outperforming static personalization strategies.

---

## 🔍 Motivation

Traditional federated learning methods (e.g., FedAvg) assume:
- homogeneous clients,
- fixed local training schedules,
- static personalization strategies.

In practice, clients vary significantly in:
- data distribution (non-IID),
- compute speed,
- communication bandwidth.

This project explores whether **a single learning agent** can make **adaptive, client-specific decisions** that improve:
- global generalization,
- fairness across clients,
- computational efficiency.

---

## 🧠 Key Idea

- There is **only one RL agent** (centralized at the server).
- The agent observes **client states** and outputs **per-client actions**.
- The same policy network is shared across all clients.

```
               ┌────────────────────────────┐
               │   Single RL Agent (Server) │
               │   πθ(s) → a                │
               └────────────┬───────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
     Client 1             Client 2             Client K
   (state s₁)           (state s₂)           (state sₖ)
  action a₁             action a₂             action aₖ
(layer choice,       (layer choice,       (layer choice,
 local epochs)        local epochs)        local epochs)
```

---

## 📂 Repository Structure

```
single_agent_RL_for_pFL/
│
├── models/
│   ├── resnet18.py              # CIFAR-style ResNet
│   ├── RL_model.py              # RL policy network
│
├── data_loaders/
│   ├── cifar_10_dataloader.py   # Dirichlet non-IID splits
│
├── utils/
│   ├── aggregation.py           # FedAvg-style aggregation
│   ├── evaluation.py            # Global & client evaluation
│
├── train_rl_federated.py         # Main FL + RL training loop
├── baseline_federated.py         # Non-RL baselines
├── config.py                     # Hyperparameters
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **CIFAR-10**
  - 50,000 training images → split among clients (non-IID)
  - 10,000 test images → **used only for global evaluation**

> The CIFAR-10 **test set is never used for training**.

---

## 🔬 Non-IID Data Partitioning

- Training data is split across clients using a **Dirichlet distribution**.
- Smaller concentration parameter `α` → higher heterogeneity.
- Each client receives a different label distribution and dataset size.

---

## ⚙️ Client Heterogeneity

Each client is characterized by:
- local dataset size,
- compute capability,
- bandwidth level,
- participation frequency,
- previous model performance.

These attributes are encoded into the **RL state**.

---

## 🎯 RL Formulation

### State (per client)

A compact vector encoding:
- normalized compute capacity,
- normalized bandwidth,
- local dataset size,
- previous validation accuracy,
- training progress.

### Action (per client)

The RL agent outputs:
- **Layer personalization decision**
  - which blocks are personalized vs shared
  - including support for *middle-layer personalization*
- **Training intensity**
  - number of local epochs

### Reward

A scalar reward computed at the server after each communication round:

```
reward = α · Δ(global test accuracy)
       − β · variance(client accuracies)
       − γ · compute / communication cost
```

This encourages:
- good global generalization,
- fairness across clients,
- efficient use of resources.

---

## 📉 Loss Functions

### Client-Side Learning

Each client trains its local model using standard supervised learning:

\[
\mathcal{L}_{client} = \text{CrossEntropy}(y, f(x))
\]

### RL Agent Loss (Policy Gradient)

The RL agent is trained using a REINFORCE-style objective:

\[
\mathcal{L}_{RL} = - \sum_{t,k}
\log \pi_\theta(a_{k,t} \mid s_{k,t}) \cdot G_t
\]

where \( G_t \) is the discounted return.

The RL loss is **completely separate** from the client training loss.


## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run RL-based personalized FL
```bash
python train_rl_federated.py
```

### 3. Run non-RL baselines
```bash
python baseline_federated.py
```

---

## 📈 Evaluation Metrics

- **Global accuracy** (CIFAR-10 test set)
- **Client fairness**
  - mean and variance of client accuracies
- **Efficiency**
  - number of local updates
  - communication cost
- **Policy behavior**
  - distribution of selected layers and local epochs

---

## 📄 Project Report (LaTeX / PDF)

A full technical report (written in LaTeX) describing:
- formal problem formulation,
- RL state–action–reward definitions,
- algorithm details,
- experimental results and ablations,

can be included in a `docs/` directory as a compiled PDF.

---

## 👤 Author

**Praditha Alwis**  
PhD Student, Electrical & Computer Engineering  
Purdue University

**Kavindu Herath**  
PhD Student, Electrical & Computer Engineering  
Purdue University

**Nethmi Hewa Withthige**  
PhD Student, Electrical & Computer Engineering  
Purdue University

**Lakshika Karunaratne**  
PhD Student, Electrical & Computer Engineering  
Purdue University

---

## 📜 License

MIT License
