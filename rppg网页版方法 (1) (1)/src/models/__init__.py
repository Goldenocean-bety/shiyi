"""
Deep Learning Models for rPPG
"""

try:
    from .physnet import (
        PhysNet3D,
        PhysNetLite,
        PhysNetPreprocessor,
        PhysNetInference,
        TORCH_AVAILABLE
    )
    from .tscan import (
        TSCAutoencoder,
        TSCAutoencoderLite,
        TSCANPreprocessor,
        TSCANInference
    )
except ImportError as e:
    TORCH_AVAILABLE = False
    PhysNet3D = None
    PhysNetLite = None
    PhysNetPreprocessor = None
    PhysNetInference = None
    TSCAutoencoder = None
    TSCAutoencoderLite = None
    TSCANPreprocessor = None
    TSCANInference = None

__all__ = [
    "PhysNet3D",
    "PhysNetLite", 
    "PhysNetPreprocessor",
    "PhysNetInference",
    "TSCAutoencoder",
    "TSCAutoencoderLite",
    "TSCANPreprocessor",
    "TSCANInference",
    "TORCH_AVAILABLE"
]
