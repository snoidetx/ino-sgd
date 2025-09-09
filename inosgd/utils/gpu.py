import torch


def connect_to_(index=0):
    if torch.cuda.is_available():
        device = torch.device("cuda", index=index)
    else:
        device = torch.device("cpu")

    print(f"Connected to {device}")
    return device
