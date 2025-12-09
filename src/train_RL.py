import os
import math
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


from models.RL_model import PolicyNet
from models.resnet18 import ResNet18CIFAR
from models.simple_CNN_for_CIFAR_10 import CIFAR10CNN
from data_loaders.cifar_10_dataloader import get_cifar10_dirichlet_clients


# ============================================================
# 0. Utils: set seed
# ============================================================

def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_trainable_depth(model: CIFAR10CNN, depth: int):
    """
    depth in [1, model.num_layers_for_rl]
    Layers [0, depth-1] are trainable; others are frozen.
    Layers = [block1, block2, block3, classifier]
    """
    assert 1 <= depth <= model.num_layers_for_rl
    layers = list(model.blocks) + [model.classifier]

    for i, layer in enumerate(layers):
        trainable = (i < depth)
        for p in layer.parameters():
            p.requires_grad = trainable
# ============================================================

def select_action(policy_net, client_id, num_clients, device):
    """
    Returns:
        action (int, depth in [1..num_layers])
        logprob (tensor)
    """
    state = F.one_hot(torch.tensor([client_id], device=device),
                      num_classes=num_clients).float()  # shape [1, num_clients]
    logits = policy_net(state)
    dist = torch.distributions.Categorical(logits=logits)
    action_idx = dist.sample()         # 0 .. num_actions-1
    logprob = dist.log_prob(action_idx)
    return (action_idx.item() + 1), logprob   # depth = idx + 1


# ============================================================
# 4. Federated helpers: FedAvg & local training
# ============================================================

def local_train(model,
                dataloader,
                device,
                epochs=1,
                lr=0.01,
                momentum=0.9):
    """
    Train the (possibly partially frozen) client model locally.
    Returns:
        model (updated)
        accuracy on this client's data (same dataloader used for eval)
    """
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum
    )

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    # Evaluate local accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total if total > 0 else 0.0
    return model, acc


def fedavg(global_model, client_models, client_idcs):
    """
    FedAvg: weighted average of client models by number of samples.
    """
    # initialize with zeros
    global_model_cpu = global_model.to("cpu")
    global_state = global_model_cpu.state_dict()

    avg_state = {k: torch.zeros_like(v, device="cpu") for k, v in global_state.items()}

    total_samples = sum(len(idcs) for idcs in client_idcs.values())

    for cid, client_model in client_models.items():
        weight = len(client_idcs[cid]) / total_samples
        # get client weights on CPU
        client_state = {k: v.detach().cpu() for k, v in client_model.state_dict().items()}
        for k in avg_state.keys():
            if not torch.is_floating_point(avg_state[k]):
                # skip non-floating tensors
                continue
            avg_state[k] += client_state[k] * weight

    global_model_cpu.load_state_dict(avg_state)
    return global_model_cpu


def model_to_vector(model: nn.Module) -> torch.Tensor:
    """
    Flatten all params into a single 1D tensor (on CPU).
    """
    return torch.cat([p.detach().cpu().view(-1) for p in model.parameters()])


# ============================================================
# 5. RL + Federated training loop
# ============================================================

def train_federated_rl(
    data_root="/local/scratch/a/dalwis/single_agent_RL_for_pFL/data",
    num_clients=5,
    alpha=5.0,
    batch_size=64,
    num_rounds=10,
    local_epochs=1,
    lr_local=0.01,
    momentum_local=0.9,
    rl_lr=1e-3,
    lambda_dist=1e-4,
    seed=123,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- Data ----------
    trainset, client_idcs, client_loaders = get_cifar10_dirichlet_clients(
        data_root=data_root,
        num_clients=num_clients,
        batch_size=batch_size,
        seed=seed,
    )

    testset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    testloader = DataLoader(testset, batch_size=128, shuffle=False)


    # ---------- Models ----------
    global_model = ResNet18CIFAR(num_classes=10)
    num_actions = global_model.num_layers_for_rl  # depths 1..L

    # RL policy
    rl_device = device
    policy_net = PolicyNet(num_clients=num_clients,
                           num_actions=num_actions,
                           hidden_dim=32).to(rl_device)
    policy_opt = torch.optim.Adam(policy_net.parameters(), lr=rl_lr)

    print(f"RL will choose depth in [1, {num_actions}] for each client.")

    # ---------- Training ----------
    for global_round in range(num_rounds):
        print(f"\n=== Global Round {global_round + 1}/{num_rounds} ===")

        client_models = {}
        client_accuracies = {}
        client_logprobs = {}
        client_vectors = {}

        # 1) For each client: RL chooses depth, client trains locally
        for cid in range(num_clients):
            # copy global model to client
            client_model = ResNet18CIFAR(num_classes=10)
            client_model.load_state_dict(global_model.state_dict())

            # RL chooses depth
            depth, logprob = select_action(
                policy_net=policy_net,
                client_id=cid,
                num_clients=num_clients,
                device=rl_device,
            )
            client_logprobs[cid] = logprob

            # set layer trainability according to depth
            set_trainable_depth(client_model, depth=depth)

            # local training
            client_model, acc = local_train(
                model=client_model,
                dataloader=client_loaders[cid],
                device=device,
                epochs=local_epochs,
                lr=lr_local,
                momentum=momentum_local,
            )

            client_models[cid] = client_model
            client_accuracies[cid] = acc
            client_vectors[cid] = model_to_vector(client_model)

            print(f"  Client {cid}: depth={depth}, local_acc={acc:.4f}")

        # 2) Compute pairwise distances & rewards
        rewards = {}
        for cid in range(num_clients):
            vec_c = client_vectors[cid]
            dists = []
            for other_cid in range(num_clients):
                if other_cid == cid:
                    continue
                vec_o = client_vectors[other_cid]
                d = torch.norm(vec_c - vec_o, p=2).item()
                dists.append(d)
            mean_dist = np.mean(dists) if len(dists) > 0 else 0.0

            reward = client_accuracies[cid] - lambda_dist * mean_dist
            rewards[cid] = reward

            print(f"  Client {cid}: mean_dist={mean_dist:.2f}, "
                  f"reward={reward:.4f}")

        # 3) RL policy update (REINFORCE with per-round baseline)
        baseline = np.mean(list(rewards.values()))
        policy_opt.zero_grad()
        policy_loss = 0.0

        for cid in range(num_clients):
            advantage = rewards[cid] - baseline
            # negative for gradient ascent
            policy_loss = policy_loss - client_logprobs[cid] * advantage

        policy_loss = policy_loss / num_clients
        policy_loss.backward()
        policy_opt.step()

        print(f"  RL policy loss: {policy_loss.item():.6f}")

        # FedAvg to update global model
        global_model = fedavg(global_model, client_models, client_idcs)

        # quick global evaluation on union of all data (optional)
        global_acc = evaluate_global(global_model, testloader, device)
        print(f"Global model accuracy (on all client data): {global_acc:.4f}")

    print("\nTraining finished.")

    # ================================
    # Save final global + client models
    # ================================
    save_models(
        global_model=global_model,
        client_models=client_models,   # <-- last-round client models used in FedAvg
        policy_net=policy_net,
        save_dir="/local/scratch/a/dalwis/single_agent_RL_for_pFL/src/weights/split_2_clients_resnet18"
    )

    return global_model, policy_net


# ============================================================
# 6. Evaluation helper
# ============================================================

def evaluate_global(global_model, testloader, device):
    global_model.to(device)
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            logits = global_model(x)   # Use ONLY global model, no personal heads
            _, pred = torch.max(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def save_models(global_model, client_models, policy_net, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)

    # --- Save global model ---
    global_path = os.path.join(save_dir, "global_model.pt")
    torch.save(global_model.state_dict(), global_path)
    print(f"Saved global model → {global_path}")

    # --- Save client models ---
    for cid, model in client_models.items():
        client_path = os.path.join(save_dir, f"client_{cid}_model.pt")
        torch.save(model.state_dict(), client_path)
        print(f"Saved client {cid} model → {client_path}")

    # --- Save RL policy (optional) ---
    policy_path = os.path.join(save_dir, "policy_net.pt")
    torch.save(policy_net.state_dict(), policy_path)
    print(f"Saved RL policy network → {policy_path}")



# ============================================================
# 7. Run
# ============================================================

if __name__ == "__main__":
    # You can change these hyperparameters as needed
    final_global_model, final_policy = train_federated_rl(
        data_root="/local/scratch/a/dalwis/single_agent_RL_for_pFL/data/cifar_10/split_2_clients",
        num_clients=2,
        batch_size=64,
        num_rounds=50,
        local_epochs=4,
        lr_local=0.01,
        momentum_local=0.9,
        rl_lr=1e-3,
        lambda_dist=1e-2,   # penalty weight for Euclidean distance
        seed=123,
    )
