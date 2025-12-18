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
├
├── docs/
|   ├── project_report.pdf      # Project Report
│
├── experiments/
│   ├── Layer Selection Experiment.ipynb              # Experiments to find a proper action space for layer selection agent
│   ├── Training Intensity.ipynb                      # Experiments to find a proper action space for training intensity 
deciding agent
|
├── layer_selection_agent/
│   ├── data
│   ├── plots
|   ├── results
│   ├── src
│
├── combined_RL_agent_main.ipynb
└── README.md
```

---

## 📊 Dataset

- **CIFAR-10**
- **MNIST**


## 🚀 How to Run

### 1. Use the jupyter notebook to run combined RL-Agent (Main Program)
```bash
combined_RL_agent_main.ipynb
```

### 2. Run the notebooks in experiments folder to run our experiments
```bash
Layer Selection Experiment.ipynb
Training Intensity.ipynb

```
### 3. To run the layer selection RL agent. Go to layer_selection_agent/src and run.
```bash
python train_RL.py

```

### 4. To run the layer selection baseline. Go to layer_selection_agent/src and run.
```bash
python train_baseline.py

```

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
