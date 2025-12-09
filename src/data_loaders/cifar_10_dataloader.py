import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CIFAR10ClientDataset(Dataset):
    def __init__(self, data_path, target_path, transform=None):
        self.data = np.load(data_path)          # shape: (N, 32, 32, 3), uint8
        self.targets = np.load(target_path)     # shape: (N,)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]                   # (H, W, C), uint8
        img = img.astype("uint8")

        if self.transform:
            img = self.transform(img)

        return img, int(self.targets[idx])


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cifar10_dirichlet_clients(
    data_root,
    num_clients,
    batch_size,
    seed=123
):
    """
    Load non-IID CIFAR-10 client splits from pre-saved numpy files.

    Expected directory structure:
        data_root/
            client_0/
                data.npy
                targets.npy
            client_1/
                data.npy
                targets.npy
            ...
            client_{num_clients-1}/
                data.npy
                targets.npy

    Returns:
        trainset: None (no global CIFAR10 object needed here)
        client_idcs: dict {client_id: [0..num_client_samples-1]} (local indices)
        client_loaders: dict {client_id: DataLoader}
    """
    set_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    client_loaders = {}
    client_idcs = {}

    for cid in range(num_clients):
        cid_dir = os.path.join(data_root, f"client_{cid}")
        data_path = os.path.join(cid_dir, "data.npy")
        target_path = os.path.join(cid_dir, "targets.npy")

        if not (os.path.exists(data_path) and os.path.exists(target_path)):
            raise FileNotFoundError(
                f"Missing data/targets for client {cid} in {cid_dir}"
            )

        dataset = CIFAR10ClientDataset(
            data_path=data_path,
            target_path=target_path,
            transform=transform,
        )

        # local indices within each client's own dataset
        client_idcs[cid] = list(range(len(dataset)))

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        client_loaders[cid] = loader

    trainset = None  # we’re working purely with the saved splits

    return trainset, client_idcs, client_loaders


if __name__ == "__main__":
    NUM_CLIENTS = 2
    BATCH_SIZE = 64

    trainset, client_idcs, client_loaders = get_cifar10_dirichlet_clients(
        data_root="/local/scratch/a/dalwis/single_agent_RL_for_pFL/data/cifar_10/split_2_clients",
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        seed=123,
    )

    for cid, loader in client_loaders.items():
        print(f"Client {cid}: {len(loader.dataset)} samples")


