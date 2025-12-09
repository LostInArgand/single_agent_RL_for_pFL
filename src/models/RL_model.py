import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """
    Simple policy network:
    - Input: one-hot client id (size = num_clients)
    - Output: logits over actions = possible depths {1, ..., num_layers}
    """

    def __init__(self, num_clients: int, num_actions: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(num_clients, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        # x: (batch_size, num_clients)
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits


if __name__ == "__main__":
    num_clients = 2
    num_actions = 4     # depths
    batch_size = 3

    print("Creating PolicyNet...")
    policy = PolicyNet(num_clients=num_clients, num_actions=num_actions)
    print(policy)

    # Create dummy one-hot encoded client IDs
    dummy_input = torch.zeros(batch_size, num_clients)
    dummy_input[0, 1] = 1   # client 1
    # dummy_input[1, 3] = 1   # client 3
    dummy_input[2, 0] = 1   # client 0

    print("\nDummy one-hot input:")
    print(dummy_input)

    # Forward pass
    logits = policy(dummy_input)

    print("\nOutput logits:")
    print(logits)
    print("Output shape:", logits.shape)

    # Assertions
    assert logits.shape == (batch_size, num_actions), "Incorrect output shape!"
    print("\n PolicyNet forward pass OK!")