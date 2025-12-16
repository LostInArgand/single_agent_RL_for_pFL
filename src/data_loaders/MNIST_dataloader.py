import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MNISTClientDataset(Dataset):
    """
    Expects:
      - data.npy:   (N, 28, 28)      uint8  OR (N, 28, 28, 1) uint8
      - targets.npy:(N,)             int64/int32/etc.
    """
    def __init__(self, data_path, target_path, transform=None):
        self.data = np.load(data_path)       # (N,28,28) or (N,28,28,1)
        self.targets = np.load(target_path)  # (N,)
        self.transform = transform

        if self.data.dtype != np.uint8:
            # keep it consistent with torchvision transforms expecting uint8 PIL/ndarray
            self.data = self.data.astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]

        # Handle (28,28,1) -> (28,28)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]

        # Now img should be (H,W) uint8
        if self.transform:
            img = self.transform(img)

        return img, int(self.targets[idx])


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_mnist_dirichlet_clients(
    data_root,
    num_clients,
    batch_size,
    seed=123,
    normalize=True,
):
    """
    Load non-IID MNIST client splits from pre-saved numpy files.

    Expected directory structure:
        data_root/
            client_0/
                data.npy
                targets.npy
            ...
            client_{num_clients-1}/
                data.npy
                targets.npy

    Returns:
        trainset: None
        client_idcs: dict {client_id: [0..num_client_samples-1]} (local indices)
        client_loaders: dict {client_id: DataLoader}
    """
    set_seed(seed)

    if normalize:
        # Standard MNIST normalization
        transform = transforms.Compose([
            transforms.ToTensor(),  # (H,W) uint8 -> (1,H,W) float in [0,1]
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    client_loaders = {}
    client_idcs = {}

    for cid in range(num_clients):
        cid_dir = os.path.join(data_root, f"client_{cid}")
        data_path = os.path.join(cid_dir, "data.npy")
        target_path = os.path.join(cid_dir, "targets.npy")

        if not (os.path.exists(data_path) and os.path.exists(target_path)):
            raise FileNotFoundError(f"Missing data/targets for client {cid} in {cid_dir}")

        dataset = MNISTClientDataset(
            data_path=data_path,
            target_path=target_path,
            transform=transform,
        )

        client_idcs[cid] = list(range(len(dataset)))

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        client_loaders[cid] = loader

    trainset = None
    return trainset, client_idcs, client_loaders


if __name__ == "__main__":
    NUM_CLIENTS = 10
    BATCH_SIZE = 64

    trainset, client_idcs, client_loaders = get_mnist_dirichlet_clients(
        data_root="/local/scratch/a/dalwis/single_agent_RL_for_pFL/data/MNIST/split_10_clients_alpha_5",
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        seed=123,
    )

    for cid, loader in client_loaders.items():
        x, y = next(iter(loader))
        print(f"Client {cid}: {len(loader.dataset)} samples | batch x: {x.shape} y: {y.shape}")
        # x should be [B, 1, 28, 28]
