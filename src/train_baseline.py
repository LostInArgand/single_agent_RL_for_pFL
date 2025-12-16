import os
import random
import numpy as np

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.RL_model import PolicyNet
from models.resnet18 import ResNet18CIFAR, ResNet18MNIST
from data_loaders.MNIST_dataloader import get_mnist_dirichlet_clients
from data_loaders.cifar_10_dataloader import get_cifar10_dirichlet_clients


import argparse

def get_config():
    parser = argparse.ArgumentParser()

    # dataset / model
    parser.add_argument('--model_dataset', type=str, default='CIFAR10',
                        choices=['MNIST', 'CIFAR10'])

    # training
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)

    # RL / FL
    parser.add_argument('--communication_budget', type=int, default=3)
    parser.add_argument('--seed', type=int, default=123)

    return parser.parse_args()

config = get_config()

# ============================================================
# 0. Utils: set seed
# ============================================================

def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_trainable_depth(model: nn.Module, depth: int):
    """
    depth in [1, model.num_layers_for_rl]
    Layers [0, depth-1] are trainable; others are frozen.
    Assumes:
        model.blocks is an iterable of blocks
        model.classifier is the final head
    """
    assert 1 <= depth <= model.num_layers_for_rl
    layers = list(model.blocks) + [model.classifier]

    for i, layer in enumerate(layers):
        trainable = (i < depth)
        for p in layer.parameters():
            p.requires_grad = trainable


# ============================================================
# 1. RL action selection
# ============================================================

def select_action(policy_net, state_vec, device):
    """
    Args:
        state_vec: list/np array of length state_dim
    Returns:
        action (int)  in [1..num_actions]  = number of layers to share
        logprob (tensor)
    """
    state = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)  # [1, state_dim]
    logits = policy_net(state)
    dist = torch.distributions.Categorical(logits=logits)
    action_idx = dist.sample()  # 0..num_actions-1
    logprob = dist.log_prob(action_idx)
    return (action_idx.item() + 1), logprob   # map to [1..num_actions]


# ============================================================
# 2. Federated helpers: local training
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


# ============================================================
# 3. Layer mapping and partial FedAvg
# ============================================================

def get_layer_param_names(model: nn.Module):
    """
    Returns:
        layer_param_names: list of lists.
        layer_param_names[i] = list of param names belonging to layer i.
    Assumes:
        model.blocks is an nn.ModuleList (or similar)
        model.classifier exists.
    Layers are [blocks[0], blocks[1], ..., classifier]
    """
    layer_param_names = []

    num_blocks = len(list(model.blocks))
    for layer_idx in range(num_blocks + 1):
        names_for_layer = []
        for name, _ in model.named_parameters():
            if layer_idx < num_blocks:
                prefix = f"blocks.{layer_idx}."
                if name.startswith(prefix):
                    names_for_layer.append(name)
            else:
                # last layer: classifier
                if name.startswith("classifier."):
                    names_for_layer.append(name)
        layer_param_names.append(names_for_layer)

    return layer_param_names


def fedavg_partial(global_model,
                   client_models,
                   client_idcs,
                   client_share_depths):
    """
    FedAvg only over layers that clients actually share.

    Args:
        global_model: nn.Module
        client_models: dict[cid] -> nn.Module
        client_idcs: dict[cid] -> indices of samples
        client_share_depths: dict[cid] -> int (how many layers client c shared)
    Returns:
        updated global_model (on CPU)
    """
    global_model_cpu = global_model.to("cpu")
    global_state = global_model_cpu.state_dict()

    # Start from current global params
    new_state = {k: v.clone() for k, v in global_state.items()}

    # Get param names per layer
    layer_param_names = get_layer_param_names(global_model_cpu)

    # For each client, compute which param names they share
    client_shared_params = {}
    for cid, depth in client_share_depths.items():
        shared_names = []
        # depth in [1..num_layers]; share layers [0..depth-1]
        for li in range(depth):
            shared_names.extend(layer_param_names[li])
        client_shared_params[cid] = set(shared_names)

    # For each parameter, average only over contributing clients
    for name, param in global_state.items():
        if not torch.is_floating_point(param):
            continue

        contributors = [
            cid for cid in client_models.keys()
            if name in client_shared_params[cid]
        ]

        if len(contributors) == 0:
            # Nobody shared this param: keep old global value
            continue

        # Weighted by number of samples of contributing clients
        total_samples = sum(len(client_idcs[cid]) for cid in contributors)
        avg_param = torch.zeros_like(param)

        for cid in contributors:
            weight = len(client_idcs[cid]) / total_samples
            client_param = client_models[cid].state_dict()[name].detach().cpu()
            avg_param += weight * client_param

        new_state[name] = avg_param

    global_model_cpu.load_state_dict(new_state)
    return global_model_cpu


def update_clients_with_global(global_model,
                               client_models,
                               client_share_depths):
    """
    Overwrite only the shared layers of each client model
    with the new global parameters.
    """
    global_cpu = global_model.to("cpu")
    global_state = global_cpu.state_dict()
    layer_param_names = get_layer_param_names(global_cpu)

    for cid, client_model in client_models.items():
        depth = client_share_depths[cid]
        # Which params belong to layers [0..depth-1] for this client
        shared_names = []
        for li in range(depth):
            shared_names.extend(layer_param_names[li])
        shared_names = set(shared_names)

        client_state = client_model.state_dict()
        for name in shared_names:
            if name in client_state:
                client_state[name] = global_state[name].clone()

        client_model.load_state_dict(client_state)


def model_to_vector(model: nn.Module) -> torch.Tensor:
    """
    Flatten all params into a single 1D tensor (on CPU).
    """
    return torch.cat([p.detach().cpu().view(-1) for p in model.parameters()])

def model_to_vectors(model: nn.Module):
    """
    Returns 6 separate flattened vectors (on CPU), one per RL layer.

    Layer order matches your RL/fedavg code:
      [blocks[0], blocks[1], blocks[2], blocks[3], blocks[4], classifier]

    Returns:
        vectors: list[torch.Tensor] length = model.num_layers_for_rl (typically 6)
                 each tensor is 1D on CPU
    """
    # The same "layer" definition you use elsewhere
    layers = list(model.blocks) + [model.classifier]

    vectors = []
    for layer in layers:
        params = [p.detach().cpu().view(-1) for p in layer.parameters()]
        if len(params) == 0:
            vectors.append(torch.empty(0, dtype=torch.float32))
        else:
            vectors.append(torch.cat(params, dim=0))

    return vectors

# ============================================================
# 4. RL + Federated training loop
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
    lambda_comm=1e-2,               # comm violation penalty weight
    communication_budgets=None,     # list/array of ints per client
    seed=123,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- Models & communication budgets ----------
    if config.model_dataset == 'MNIST':
        global_model = ResNet18MNIST(num_classes=10)
    else:
        global_model = ResNet18CIFAR(num_classes=10)  # default
    num_actions = global_model.num_layers_for_rl  # depths / layers 1..L

    if communication_budgets is None:
        communication_budgets = [num_actions] * num_clients
    else:
        assert len(communication_budgets) == num_clients

    # Track last accuracy per client (for RL state input)
    client_last_accs = [0.0 for _ in range(num_clients)]

    # ---------- Data ----------
    if config.model_dataset == 'MNIST':
        trainset, client_idcs, client_loaders = get_mnist_dirichlet_clients(
            data_root=data_root,
            num_clients=num_clients,
            batch_size=batch_size,
            seed=seed,
        )

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        testset = datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=transform_test
        )


    else:
        trainset, client_idcs, client_loaders = get_cifar10_dirichlet_clients(
            data_root=data_root,
            num_clients=num_clients,
            batch_size=batch_size,
            seed=seed,
        )

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

        testset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )


    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    # ---------- RL policy ----------
    # rl_device = device
    # state_dim = 3  # [communication_budget, last_accuracy, client_id]
    # policy_net = PolicyNet(state_dim=state_dim,
    #                        num_actions=num_actions,
    #                        hidden_dim=32).to(rl_device)
    # policy_opt = torch.optim.Adam(policy_net.parameters(), lr=rl_lr)

    # print(f"RL will choose number of shared layers in [1, {num_actions}] for each client.")

    # Initialize client models (personalized copy per client)
    if config.model_dataset == 'MNIST':
        client_models = {
            cid: ResNet18MNIST(num_classes=10) for cid in range(num_clients)
        }
    else:
        client_models = {
            cid: ResNet18CIFAR(num_classes=10) for cid in range(num_clients)
        }

    # ---------- Training ----------
    for global_round in range(num_rounds):
        print(f"\n=== Global Round {global_round + 1}/{num_rounds} ===")

        client_accuracies = {}
        # client_logprobs = {}
        client_vectors = {}
        client_share_depths = {}

        # 1) For each client: RL chooses number of layers to share, client trains locally
        for cid in range(num_clients):
            client_model = client_models[cid]  # persistent model

            # ----- Build RL state [budget, last_acc, client_id] -----
            budget = communication_budgets[cid]
            last_acc = client_last_accs[cid]
            # simple normalization
            budget_norm = budget / float(num_actions)
            last_acc_norm = last_acc          # already in [0,1]
            client_id_norm = cid / max(1, (num_clients - 1))

            state_vec = [budget_norm, last_acc_norm, client_id_norm]

            # RL chooses how many layers to share
            layers_to_share, logprob = 3, None
            # client_logprobs[cid] = logprob
            client_share_depths[cid] = layers_to_share

            # ------ Local training ------
            # Optionally couple training depth with sharing:
            # set_trainable_depth(client_model, depth=layers_to_share)

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
            client_last_accs[cid] = acc  # update history

            print(f"  Client {cid}: layers_shared={layers_to_share}, local_acc={acc:.4f}")

        # 2) Compute pairwise distances & rewards (with comm penalty)
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

            base_reward = client_accuracies[cid] - lambda_dist * mean_dist

            # Communication violation penalty: if layers_to_share > budget
            layers_sh = client_share_depths[cid]
            budget = communication_budgets[cid]
            violation = max(0, layers_sh - budget)
            comm_penalty = lambda_comm * violation

            reward = base_reward - comm_penalty
            rewards[cid] = reward

            print(f"  Client {cid}: mean_dist={mean_dist:.2f}, "
                  f"base_reward={base_reward:.4f}, "
                  f"violation={violation}, "
                  f"comm_penalty={comm_penalty:.4f}, "
                  f"reward={reward:.4f}")

        # 3) RL policy update (REINFORCE with per-round baseline)
        # baseline = np.mean(list(rewards.values()))
        # policy_opt.zero_grad()
        # policy_loss = 0.0

        # for cid in range(num_clients):
        #     advantage = rewards[cid] - baseline
            # policy_loss = policy_loss - client_logprobs[cid] * advantage

        # policy_loss = policy_loss / num_clients
        # policy_loss.backward()
        # policy_opt.step()

        # print(f"  RL policy loss: {policy_loss.item():.6f}")

        # 4) FedAvg only on shared layers
        global_model = fedavg_partial(global_model,
                                      client_models,
                                      client_idcs,
                                      client_share_depths)

        # 5) Push updated shared layers back to each client model
        update_clients_with_global(global_model,
                                   client_models,
                                   client_share_depths)

        # 6) Evaluate global model and each client on the SAME test set
        global_acc = evaluate_model(global_model, testloader, device)
        print(f"Global model accuracy (test set): {global_acc:.4f}")

        for cid, cmodel in client_models.items():
            c_acc = evaluate_model(cmodel, testloader, device)
            print(f"  Client {cid} model accuracy on test set: {c_acc:.4f}")

    print("\nTraining finished.")

    # ================================
    # Save final global + client models
    # ================================
    # save_models(
    #     global_model=global_model,
    #     client_models=client_models,   # last-round client models
    #     policy_net=policy_net,
    #     save_dir="/local/scratch/a/dalwis/single_agent_RL_for_pFL/src/weights/split_2_clients_resnet18"
    # )

    return global_model, None


# ============================================================
# 5. Evaluation & saving
# ============================================================

def evaluate_model(model, testloader, device):
    """
    Generic evaluator: runs any model on the CIFAR-10 test set.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
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
# 6. Run
# ============================================================
if __name__ == "__main__":
    final_global_model, final_policy = train_federated_rl(
        data_root="/local/scratch/a/dalwis/single_agent_RL_for_pFL/data/MNIST/split_10_clients_alpha_5",
        num_clients=10,
        batch_size=64,
        num_rounds=100,
        local_epochs=1,
        lr_local=0.01,
        momentum_local=0.9,
        rl_lr=1e-3,
        lambda_dist=1e-2,   # penalty weight for Euclidean distance
        lambda_comm=1,   # penalty for exceeding comm budget
        communication_budgets=[3, 5, 1, 2, 4, 6, 1, 6, 2, 5],  # example: client0 <=3, client1 <=4
        seed=123,
    )
