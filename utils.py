import torch

def get_device():
    """
    Set the device to 'cuda' or 'mps' or 'cpu' depending on the availability of a GPU.
    """
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'