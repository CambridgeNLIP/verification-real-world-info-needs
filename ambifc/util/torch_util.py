import torch


def get_device(require_cuda: bool = True) -> torch.device:
    if require_cuda:
        assert torch.cuda.is_available()
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
