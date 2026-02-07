"""
TS-CAN (Temporal Shift Convolutional Attention Network)
Based on: "TS-CAN: Temporal Shift Convolutional Attention Network for Remote Photoplethysmography"

This model extracts rPPG signals from facial video using temporal shift convolution
and attention mechanisms to capture spatio-temporal patterns of blood flow.
"""

import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. TS-CAN will not be available.")


if TORCH_AVAILABLE:
    
    class TemporalShift(nn.Module):
        """
        Temporal Shift Module
        Shifts feature channels temporally to capture temporal information
        without increasing computational complexity
        """
        
        def __init__(self, n_div: int = 8):
            super().__init__()
            self.n_div = n_div
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass
            Args:
                x: Input tensor (B, C, T, H, W)
            
            Returns:
                Tensor with temporal shifted features
            """
            B, C, T, H, W = x.shape
            c_out = C // self.n_div
            
            # Split channels
            x1 = x[:, :c_out, :, :, :]
            x2 = x[:, c_out:2*c_out, :, :, :]
            x3 = x[:, 2*c_out:, :, :, :]
            
            # Shift temporally
            x1 = torch.roll(x1, shifts=1, dims=2)
            x2 = torch.roll(x2, shifts=-1, dims=2)
            
            # Pad edges
            x1[:, :, 0, :, :] = 0
            x2[:, :, -1, :, :] = 0
            
            # Concatenate
            out = torch.cat([x1, x2, x3], dim=1)
            
            return out
    
    
    class ChannelAttention(nn.Module):
        """
        Channel-wise Attention Module
        """
        
        def __init__(self, in_channels: int, reduction: int = 8):
            super().__init__()
            
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, in_channels),
                nn.Sigmoid()
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, T, H, W = x.shape
            y = self.avg_pool(x).view(B, C)
            y = self.fc(y).view(B, C, 1, 1, 1)
            return x * y
    
    
    class SpatialAttention(nn.Module):
        """
        Spatial Attention Module
        """
        
        def __init__(self, kernel_size: int = 7):
            super().__init__()
            
            self.conv = nn.Conv3d(
                2, 1, kernel_size, padding=kernel_size//2, bias=False
            )
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            out = torch.cat([avg_out, max_out], dim=1)
            out = self.conv(out)
            out = self.sigmoid(out)
            return x * out
    
    
    class TSCANBlock(nn.Module):
        """
        TS-CAN Block with temporal shift and attention
        """
        
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            
            self.temporal_shift = TemporalShift()
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            )
            self.bn = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.channel_attn = ChannelAttention(out_channels)
            self.spatial_attn = SpatialAttention()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.temporal_shift(x)
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.channel_attn(x)
            x = self.spatial_attn(x)
            return x
    
    
    class TSCAutoencoder(nn.Module):
        """
        TS-CAN Autoencoder for rPPG signal extraction
        """
        
        def __init__(self, frames: int = 64):
            super().__init__()
            self.frames = frames
            
            # Encoder
            self.encoder1 = TSCANBlock(3, 16)
            self.pool1 = nn.MaxPool3d((1, 2, 2))
            
            self.encoder2 = TSCANBlock(16, 32)
            self.pool2 = nn.MaxPool3d((1, 2, 2))
            
            self.encoder3 = TSCANBlock(32, 64)
            self.pool3 = nn.MaxPool3d((1, 2, 2))
            
            # Bottleneck
            self.bottleneck = nn.Sequential(
                TemporalShift(),
                nn.Conv3d(64, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True)
            )
            
            # Decoder
            self.decoder1 = TSCANBlock(64, 32)
            self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
            
            self.decoder2 = TSCANBlock(32, 16)
            self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
            
            self.decoder3 = TSCANBlock(16, 8)
            self.up3 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
            
            # Output
            self.output = nn.Conv3d(8, 1, kernel_size=1)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass
            Args:
                x: Input tensor (B, 3, T, H, W)
            
            Returns:
                rPPG signal (B, T)
            """
            # Encoder
            x1 = self.encoder1(x)
            x1_pool = self.pool1(x1)
            
            x2 = self.encoder2(x1_pool)
            x2_pool = self.pool2(x2)
            
            x3 = self.encoder3(x2_pool)
            x3_pool = self.pool3(x3)
            
            # Bottleneck
            x_bn = self.bottleneck(x3_pool)
            
            # Decoder
            x_d1 = self.decoder1(x_bn)
            x_d1_up = self.up1(x_d1)
            
            x_d2 = self.decoder2(x_d1_up)
            x_d2_up = self.up2(x_d2)
            
            x_d3 = self.decoder3(x_d2_up)
            x_d3_up = self.up3(x_d3)
            
            # Output
            x_out = self.output(x_d3_up)
            
            # Global average over spatial dimensions
            x_out = x_out.mean(dim=[-2, -1])  # (B, 1, T)
            x_out = x_out.squeeze(1)  # (B, T)
            
            return x_out
    
    
    class TSCAutoencoderLite(nn.Module):
        """
        Lightweight TS-CAN for faster inference
        ONNX-compatible version
        """
        
        def __init__(self, frames: int = 64):
            super().__init__()
            self.frames = frames
            
            # Encoder with explicit padding
            self.conv1 = nn.Conv3d(3, 16, (1, 5, 5), padding=(0, 2, 2))
            self.bn1 = nn.BatchNorm3d(16)
            self.ts1 = TemporalShift()
            self.pool1 = nn.MaxPool3d((1, 2, 2))
            
            self.conv2 = nn.Conv3d(16, 32, (3, 3, 3), padding=(1, 1, 1))
            self.bn2 = nn.BatchNorm3d(32)
            self.ts2 = TemporalShift()
            self.pool2 = nn.MaxPool3d((1, 2, 2))
            
            self.conv3 = nn.Conv3d(32, 32, (3, 3, 3), padding=(1, 1, 1))
            self.bn3 = nn.BatchNorm3d(32)
            self.ts3 = TemporalShift()
            self.pool3 = nn.MaxPool3d((1, 2, 2))
            
            # Channel attention
            self.attn = ChannelAttention(32)
            
            # Temporal processing
            self.temporal = nn.Conv3d(32, 16, (3, 1, 1), padding=(1, 0, 0))
            self.bn_t = nn.BatchNorm3d(16)
            
            # Output
            self.output = nn.Conv3d(16, 1, 1)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 3, T, 64, 64)
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.ts1(x)
            x = self.pool1(x)  # (B, 16, T, 32, 32)
            
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.ts2(x)
            x = self.pool2(x)  # (B, 32, T, 16, 16)
            
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.ts3(x)
            x = self.pool3(x)  # (B, 32, T, 8, 8)
            
            x = self.attn(x)
            
            x = F.relu(self.bn_t(self.temporal(x)))  # (B, 16, T, 8, 8)
            
            x = self.output(x)  # (B, 1, T, 8, 8)
            
            # Global average over spatial dimensions
            x = x.mean(dim=[-2, -1])  # (B, 1, T)
            x = x.squeeze(1)  # (B, T)
            
            return x


class TSCANPreprocessor:
    """
    Preprocesses video frames for TS-CAN input.
    """
    
    def __init__(self,
                 input_size: Tuple[int, int] = (64, 64),
                 clip_length: int = 64,
                 normalize: bool = True):
        self.input_size = input_size
        self.clip_length = clip_length
        self.normalize = normalize
        
        # Normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Frame buffer
        self.frames = []
    
    def add_frame(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int] = None):
        """
        Add a frame to the buffer.
        """
        import cv2
        
        # Crop face if bbox provided
        if face_bbox is not None:
            x, y, w, h = face_bbox
            frame = frame[y:y+h, x:x+w]
        
        # Resize
        frame = cv2.resize(frame, self.input_size)
        
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # To float [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Normalize
        if self.normalize:
            frame = (frame - self.mean) / self.std
            
        self.frames.append(frame)
    
    def get_clip(self) -> Optional[np.ndarray]:
        """
        Get a clip ready for model input.
        """
        if len(self.frames) < self.clip_length:
            return None
            
        # Get last clip_length frames
        clip = np.array(self.frames[-self.clip_length:])
        
        # (T, H, W, C) -> (C, T, H, W)
        clip = clip.transpose(3, 0, 1, 2)
        
        # Add batch dimension
        clip = clip[np.newaxis, ...]
        
        return clip.astype(np.float32)
    
    def clear(self):
        """Clear frame buffer."""
        self.frames = []


class TSCANInference:
    """
    High-level interface for TS-CAN inference.
    """
    
    def __init__(self,
                 model_path: str = None,
                 device: str = 'auto',
                 use_lite: bool = True):
        """
        Initialize TS-CAN inference.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Run: pip install torch")
            
        # Select device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"TS-CAN using device: {self.device}")
        
        # Initialize model
        if use_lite:
            self.model = TSCAutoencoderLite()
        else:
            self.model = TSCAutoencoder()
            
        self.model.to(self.device)
        self.model.eval()
        
        # Load weights if provided
        if model_path and Path(model_path).exists():
            self.load_weights(model_path)
            
        # Preprocessor
        self.preprocessor = TSCANPreprocessor()
        
        # Signal buffer for post-processing
        self.signal_buffer = []
    
    def load_weights(self, path: str):
        """Load pre-trained weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Loaded weights from: {path}")
    
    def save_weights(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"Saved weights to: {path}")
    
    def export_onnx(self, path: str, clip_length: int = 64):
        """
        Export model to ONNX format.
        """
        dummy_input = torch.randn(1, 3, clip_length, 64, 64).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            input_names=['video_clip'],
            output_names=['rppg_signal'],
            dynamic_axes={
                'video_clip': {0: 'batch', 2: 'time'},
                'rppg_signal': {0: 'batch', 1: 'time'}
            },
            opset_version=14
        )
        print(f"Exported ONNX model to: {path}")
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        face_bbox: Tuple[int, int, int, int] = None
    ) -> Optional[np.ndarray]:
        """
        Process a single frame and return rPPG signal if ready.
        """
        # Add frame
        self.preprocessor.add_frame(frame, face_bbox)
        
        # Get clip
        clip = self.preprocessor.get_clip()
        
        if clip is None:
            return None
            
        # Inference
        with torch.no_grad():
            clip_tensor = torch.from_numpy(clip).to(self.device)
            signal = self.model(clip_tensor)
            signal = signal.cpu().numpy()[0]  # Remove batch dim
            
        return signal
    
    def reset(self):
        """Reset buffers."""
        self.preprocessor.clear()
        self.signal_buffer = []


# Quick test
if __name__ == "__main__":
    print("TS-CAN Module")
    print("=" * 50)
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    
    if TORCH_AVAILABLE:
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Test model
        model = TSCAutoencoderLite()
        print(f"\nTSCAutoencoderLite parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        dummy = torch.randn(1, 3, 64, 64, 64)
        output = model(dummy)
        print(f"Input shape: {dummy.shape}")
        print(f"Output shape: {output.shape}")
