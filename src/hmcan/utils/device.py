"""Device management utilities."""

import torch


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get the appropriate torch device.

    Args:
        device_str: Device specification. Options:
            - "auto": Automatically select best available device
            - "cpu": Force CPU
            - "cuda": Force CUDA (will fail if unavailable)
            - "cuda:0", "cuda:1", etc.: Specific CUDA device
            - "mps": Force MPS (Apple Silicon)

    Returns:
        torch.device: The selected device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device_str)


def get_device_info(device: torch.device) -> str:
    """
    Get information about the device.

    Args:
        device: The torch device

    Returns:
        String with device information
    """
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return (
            f"CUDA Device: {props.name}\n"
            f"  Memory: {props.total_memory / 1024**3:.1f} GB\n"
            f"  Compute Capability: {props.major}.{props.minor}"
        )
    elif device.type == "mps":
        return "Apple MPS (Metal Performance Shaders)"
    else:
        return "CPU"
