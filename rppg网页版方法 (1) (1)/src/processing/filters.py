"""
Advanced Signal Processing Methods for rPPG
Implements research-grade algorithms: CHROM, POS, and signal quality metrics

References:
- CHROM: De Haan & Jeanne, 2013 - "Robust Pulse Rate From Chrominance-Based rPPG"
- POS: Wang et al., 2017 - "Algorithmic Principles of Remote PPG"
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, List
from collections import deque


class BandpassFilter:
    """
    Butterworth bandpass filter for isolating heart rate frequencies.
    
    Heart rate range: 54-138 BPM = 0.9-2.3 Hz
    """
    
    def __init__(
        self, 
        lowcut: float = 0.9,
        highcut: float = 2.3,
        fps: float = 30.0,
        order: int = 5  # Higher order for better filtering
    ):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fps = fps
        self.order = order
        self._design_filter()
        
    def _design_filter(self):
        """Design the Butterworth bandpass filter."""
        nyquist = self.fps / 2.0
        low = max(0.01, min(self.lowcut / nyquist, 0.99))
        high = max(low + 0.01, min(self.highcut / nyquist, 0.99))
        self.b, self.a = signal.butter(self.order, [low, high], btype='band')
        
    def filter(self, data: np.ndarray) -> np.ndarray:
        """Apply zero-phase bandpass filter."""
        if len(data) < 3 * self.order + 1:
            return data
        try:
            return signal.filtfilt(self.b, self.a, data, padlen=3 * self.order)
        except ValueError:
            return data


def detrend_signal(signal_data: np.ndarray, lambda_val: float = 10) -> np.ndarray:
    """
    Remove slow trends from signal using smoothness priors approach.
    This removes baseline wander while preserving pulse signal.
    """
    n = len(signal_data)
    if n < 3:
        return signal_data
        
    # Simple detrending using moving average subtraction
    window = min(n // 2, int(n * 0.1)) or 1
    if window < 3:
        return signal_data - np.mean(signal_data)
    
    # Use a simple high-pass effect
    smoothed = np.convolve(signal_data, np.ones(window) / window, mode='same')
    return signal_data - smoothed


class CHROMMethod:
    """
    CHROM (Chrominance-based) Method
    
    From: De Haan & Jeanne (2013)
    "Robust Pulse Rate From Chrominance-Based rPPG"
    
    Key idea: Use chrominance signals (color differences) that are
    less sensitive to luminance changes from motion.
    """
    
    @staticmethod
    def extract_pulse(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Extract pulse signal using CHROM method.
        
        Args:
            r, g, b: Time series of mean R, G, B values from ROI
            
        Returns:
            Pulse signal
        """
        if len(r) < 3:
            return np.array([])
            
        # Normalize by temporal mean
        r_mean = np.mean(r)
        g_mean = np.mean(g)
        b_mean = np.mean(b)
        
        if r_mean < 1e-8 or g_mean < 1e-8 or b_mean < 1e-8:
            return np.array([])
        
        r_n = r / r_mean
        g_n = g / g_mean
        b_n = b / b_mean
        
        # CHROM projection
        # These coefficients are derived from skin reflectance analysis
        Xs = 3 * r_n - 2 * g_n
        Ys = 1.5 * r_n + g_n - 1.5 * b_n
        
        # Combine using standard deviation ratio
        std_Xs = np.std(Xs)
        std_Ys = np.std(Ys)
        
        if std_Ys < 1e-8:
            return Xs - np.mean(Xs)
            
        alpha = std_Xs / std_Ys
        pulse = Xs - alpha * Ys
        
        # Remove mean
        pulse = pulse - np.mean(pulse)
        
        return pulse


class CHROME_DEHAANMethod:
    """
    Enhanced CHROM Method from rPPG-Toolbox
    
    From: De Haan & Jeanne (2013)
    "Robust Pulse Rate From Chrominance-Based rPPG"
    
    Key improvements:
    - Sliding window approach with overlap
    - Hanning window for spectral leakage reduction
    - Improved normalization and filtering
    """
    
    @staticmethod
    def extract_pulse(r: np.ndarray, g: np.ndarray, b: np.ndarray, fps: float = 30.0) -> np.ndarray:
        """
        Extract pulse signal using enhanced CHROME_DEHAAN method.
        
        Args:
            r, g, b: Time series of mean R, G, B values from ROI
            fps: Frames per second
            
        Returns:
            Pulse signal
        """
        if len(r) < 30:
            return np.array([])
            
        LPF = 0.9  # Matches our bandpass filter
        HPF = 2.3  # Matches our bandpass filter
        WinSec = 2.0  # Longer window for more stable signal
        
        RGB = np.vstack((r, g, b)).T
        FN = RGB.shape[0]
        NyquistF = 0.5 * fps
        
        # Design bandpass filter
        from scipy import signal
        B, A = signal.butter(4, [LPF/NyquistF, HPF/NyquistF], 'bandpass')  # Higher order
        
        WinL = int(np.ceil(WinSec * fps))
        if WinL % 2 == 1:
            WinL += 1
        
        NWin = int(np.floor((FN - WinL//2) / (WinL//2)))
        WinS = 0
        WinM = int(WinS + WinL//2)
        WinE = WinS + WinL
        totallen = (WinL//2) * (NWin + 1)
        S = np.zeros(totallen)
        
        for i in range(NWin):
            # Normalize within window
            RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)
            RGBNorm = np.zeros((WinE-WinS, 3))
            
            for temp in range(WinS, WinE):
                RGBNorm[temp-WinS] = RGB[temp] / RGBBase
            
            # CHROM projection
            Xs = 3 * RGBNorm[:, 0] - 2 * RGBNorm[:, 1]
            Ys = 1.5 * RGBNorm[:, 0] + RGBNorm[:, 1] - 1.5 * RGBNorm[:, 2]
            
            # Apply bandpass filter
            Xf = signal.filtfilt(B, A, Xs, axis=0)
            Yf = signal.filtfilt(B, A, Ys)
            
            # Combine signals using standard deviation ratio
            Alpha = np.std(Xf) / np.std(Yf) if np.std(Yf) > 1e-8 else 1.0
            SWin = Xf - Alpha * Yf
            
            # Apply Hanning window to reduce spectral leakage
            SWin = SWin * signal.windows.hann(WinL)
            
            # Overlap-add technique
            if i == 0:
                S[WinS:WinM] = SWin[:WinL//2]
            else:
                S[WinS:WinM] += SWin[:WinL//2]
            
            S[WinM:WinE] = SWin[WinL//2:]
            
            # Move window
            WinS = WinM
            WinM = WinS + WinL//2
            WinE = WinS + WinL
        
        # If we have less data than expected, resize
        if len(S) > len(r):
            S = S[:len(r)]
        elif len(S) < len(r):
            S = np.pad(S, (0, len(r) - len(S)), mode='constant')
        
        # Remove mean and normalize
        S = S - np.mean(S)
        if np.std(S) > 1e-8:
            S = S / np.std(S)
        
        return S


class POSMethod:
    """
    POS (Plane-Orthogonal-to-Skin) Method
    
    From: Wang et al. (2017)
    "Algorithmic Principles of Remote PPG"
    
    Key idea: Project signal onto a plane orthogonal to skin-tone
    to maximize pulse signal and minimize motion artifacts.
    """
    
    @staticmethod
    def extract_pulse(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Extract pulse signal using POS method.
        
        Args:
            r, g, b: Time series of mean R, G, B values from ROI
            
        Returns:
            Pulse signal
        """
        if len(r) < 3:
            return np.array([])
            
        # Normalize by temporal mean
        r_mean = np.mean(r)
        g_mean = np.mean(g)
        b_mean = np.mean(b)
        
        if r_mean < 1e-8 or g_mean < 1e-8 or b_mean < 1e-8:
            return np.array([])
        
        r_n = r / r_mean
        g_n = g / g_mean
        b_n = b / b_mean
        
        # POS projection vectors
        # These are derived to be orthogonal to skin tone variations
        Xs = g_n - b_n
        Ys = -2 * r_n + g_n + b_n
        
        # Combine using standard deviation ratio
        std_Xs = np.std(Xs)
        std_Ys = np.std(Ys)
        
        if std_Ys < 1e-8:
            return Xs - np.mean(Xs)
            
        alpha = std_Xs / std_Ys
        pulse = Xs + alpha * Ys
        
        # Remove mean
        pulse = pulse - np.mean(pulse)
        
        return pulse


class GreenChannelMethod:
    """
    Simple green channel method (baseline).
    Uses only the green channel which has strongest pulse absorption.
    """
    
    @staticmethod
    def extract_pulse(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Extract pulse from green channel only."""
        if len(g) < 3:
            return np.array([])
        
        g_mean = np.mean(g)
        if g_mean < 1e-8:
            return np.array([])
            
        pulse = g / g_mean - 1.0
        return pulse - np.mean(pulse)


class SignalQuality:
    """
    Assess quality of extracted pulse signal.
    Higher quality = more reliable heart rate estimation.
    """
    
    @staticmethod
    def calculate_snr(signal_data: np.ndarray, fps: float, 
                      hr_estimate: float = None) -> float:
        """
        Calculate Signal-to-Noise Ratio.
        
        Args:
            signal_data: The pulse signal
            fps: Frames per second
            hr_estimate: Estimated heart rate in BPM (optional)
            
        Returns:
            SNR value (higher = better quality)
        """
        if len(signal_data) < 30:
            return 0.0
            
        # Compute power spectrum
        n = len(signal_data)
        freqs = np.fft.rfftfreq(n, 1.0 / fps)
        fft = np.fft.rfft(signal_data)
        power = np.abs(fft) ** 2
        
        # Define HR frequency range (0.7-3.0 Hz = 42-180 BPM)
        hr_range = (freqs >= 0.7) & (freqs <= 3.0)
        
        if not np.any(hr_range):
            return 0.0
            
        # Find peak in HR range
        hr_power = power[hr_range]
        hr_freqs = freqs[hr_range]
        
        if len(hr_power) == 0:
            return 0.0
            
        peak_idx = np.argmax(hr_power)
        peak_power = hr_power[peak_idx]
        
        # Signal power: peak and harmonics (within small window)
        peak_freq = hr_freqs[peak_idx]
        signal_mask = np.abs(hr_freqs - peak_freq) < 0.2
        signal_power = np.sum(hr_power[signal_mask])
        
        # Noise power: everything else in HR range
        noise_power = np.sum(hr_power[~signal_mask]) + 1e-8
        
        snr = 10 * np.log10(signal_power / noise_power)
        
        return max(0.0, min(snr, 30.0))  # Clamp to reasonable range
    
    @staticmethod
    def calculate_confidence(signal_data: np.ndarray, fps: float) -> float:
        """
        Calculate confidence score (0-1) for the signal quality.
        
        Returns:
            Confidence between 0 (poor) and 1 (excellent)
        """
        if len(signal_data) < 30:
            return 0.0
            
        snr = SignalQuality.calculate_snr(signal_data, fps)
        
        # Map SNR to confidence (0-1)
        # SNR of 5 dB = 0.5 confidence, 10 dB = 0.8, 15 dB = ~1.0
        confidence = 1 / (1 + np.exp(-(snr - 5) / 3))
        
        return float(confidence)


class SimpleSignalProcessor:
    """
    Simple signal processor for single-value or RGB signals.
    
    Includes buffering, detrending, smoothing, and normalization.
    """
    
    def __init__(self):
        self.buffer = []
        self.bufferSize = 150  # 约 5 秒数据 (按 30fps 算)
        
        # 用于简单的低通滤波 (平滑)
        self.lastValue = 0
        self.alpha = 0.2  # 平滑系数，越小越平滑，但延迟越高
    
    def process(self, rawValue):
        # 1. 缓冲数据
        self.buffer.append(rawValue)
        if len(self.buffer) > self.bufferSize:
            self.buffer.pop(0)
        
        # 数据太少时不处理
        if len(self.buffer) < 30:
            return 0
        
        # 2. 去基线 (Detrending) - 关键步骤！
        # 原始信号会随光照变化上下漂移，必须减去均值
        # 计算最近 N 帧的平均值作为基线
        sum_val = sum(self.buffer)
        mean = sum_val / len(self.buffer)
        detrended = rawValue - mean
        
        # 3. 简单的带通滤波模拟 (去除极高频和极低频)
        # 这里用一个简单的指数移动平均 (EMA) 来做低通滤波（去毛刺）
        # y[n] = α * x[n] + (1 - α) * y[n-1]
        smoothed = self.alpha * detrended + (1 - self.alpha) * self.lastValue
        self.lastValue = smoothed
        
        # 4. 标准化 (Z-Score Normalization) - 让波形幅度统一
        # 这样可以防止波形忽大忽小
        variance = sum((x - mean) ** 2 for x in self.buffer) / len(self.buffer)
        stdDev = variance ** 0.5
        if stdDev < 0.0001:
            stdDev = 1  # 防止除以0
        
        finalWave = smoothed / stdDev
        
        # 限制幅度，防止突然的爆音（如眨眼、转头引起的尖峰）
        if finalWave > 3:
            finalWave = 3
        if finalWave < -3:
            finalWave = -3
        
        return finalWave
    
    def process_rgb(self, rgb):
        """
        Process RGB signal by extracting the green channel.
        
        Args:
            rgb: Tuple of (R, G, B) values
            
        Returns:
            Processed signal value
        """
        # Extract green channel (strongest pulse signal)
        g_value = rgb[1]
        return self.process(g_value)
    
    def clear(self):
        """
        Clear the buffer.
        """
        self.buffer = []
        self.lastValue = 0


class SignalProcessor:
    """
    Advanced signal processing pipeline for rPPG.
    
    Uses multiple methods and selects the best one based on signal quality.
    """
    
    METHODS = {
        'chrom': CHROMMethod,
        'pos': POSMethod,
        'green': GreenChannelMethod,
        'chrome_dehaan': CHROME_DEHAANMethod
    }
    
    def __init__(
        self, 
        buffer_size: int = 180,  # 6 seconds at 30 FPS for better FFT resolution
        fps: float = 30.0,
        method: str = 'chrom'
    ):
        """
        Initialize signal processor.
        
        Args:
            buffer_size: Number of frames to keep in buffer
            fps: Video frames per second
            method: 'chrom', 'pos', or 'green'
        """
        self.buffer_size = buffer_size
        self.fps = fps
        self.method = method
        
        # Circular buffers for each color channel
        self.r_buffer = deque(maxlen=buffer_size)
        self.g_buffer = deque(maxlen=buffer_size)
        self.b_buffer = deque(maxlen=buffer_size)
        
        # Bandpass filter
        self.bandpass = BandpassFilter(fps=fps)
        
        # Method selector
        if method not in self.METHODS:
            raise ValueError(f"Method must be one of: {list(self.METHODS.keys())}")
        self.pulse_extractor = self.METHODS[method]
        
    def add_sample(self, rgb: Tuple[float, float, float]):
        """Add new RGB sample to buffer."""
        r, g, b = rgb
        self.r_buffer.append(r)
        self.g_buffer.append(g)
        self.b_buffer.append(b)
        
    def get_raw_signal(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get raw RGB signals."""
        return (
            np.array(self.r_buffer),
            np.array(self.g_buffer),
            np.array(self.b_buffer)
        )
        
    def get_pulse_signal(self) -> np.ndarray:
        """
        Get processed pulse signal using selected method.
        """
        r, g, b = self.get_raw_signal()
        
        if len(r) < 30:
            return np.array([])
            
        # Extract pulse using selected method
        if self.method == 'chrome_dehaan':
            # chrome_dehaan handles its own filtering
            pulse = self.pulse_extractor.extract_pulse(r, g, b, self.fps)
        else:
            pulse = self.pulse_extractor.extract_pulse(r, g, b)
            
            if len(pulse) == 0:
                return np.array([])
                
            # Detrend
            pulse = detrend_signal(pulse)
            
            # Bandpass filter
            pulse = self.bandpass.filter(pulse)
        
        return pulse
    
    def get_best_signal(self) -> Tuple[np.ndarray, str, float]:
        """
        Try all methods and return the one with best signal quality.
        
        Returns:
            Tuple of (pulse_signal, method_name, confidence)
        """
        r, g, b = self.get_raw_signal()
        
        if len(r) < 30:
            return (np.array([]), 'none', 0.0)
            
        best_signal = np.array([])
        best_method = 'none'
        best_confidence = 0.0
        
        for method_name, method_class in self.METHODS.items():
            if method_name == 'chrome_dehaan':
                # chrome_dehaan handles its own filtering
                pulse = method_class.extract_pulse(r, g, b, self.fps)
            else:
                pulse = method_class.extract_pulse(r, g, b)
                
                if len(pulse) == 0:
                    continue
                    
                pulse = detrend_signal(pulse)
                pulse = self.bandpass.filter(pulse)
            
            if len(pulse) == 0:
                continue
                
            confidence = SignalQuality.calculate_confidence(pulse, self.fps)
            
            if confidence > best_confidence:
                best_signal = pulse
                best_method = method_name
                best_confidence = confidence
                
        return (best_signal, best_method, best_confidence)
    
    # Backwards compatibility
    def get_combined_signal(self) -> np.ndarray:
        """Alias for get_pulse_signal for backwards compatibility."""
        return self.get_pulse_signal()
    
    def get_signal(self, channel: str = 'green') -> np.ndarray:
        """Get single channel signal (for debugging)."""
        if channel == 'red':
            raw = np.array(self.r_buffer)
        elif channel == 'green':
            raw = np.array(self.g_buffer)
        elif channel == 'blue':
            raw = np.array(self.b_buffer)
        else:
            raise ValueError("Channel must be 'red', 'green', or 'blue'")
            
        if len(raw) < 10:
            return raw
            
        raw = (raw - np.mean(raw)) / (np.std(raw) + 1e-8)
        return self.bandpass.filter(raw)
    
    def get_signal_quality(self) -> float:
        """Get quality score for current signal."""
        pulse = self.get_pulse_signal()
        return SignalQuality.calculate_confidence(pulse, self.fps)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough samples for analysis."""
        return len(self.g_buffer) >= self.buffer_size // 2
    
    def clear(self):
        """Clear all buffers."""
        self.r_buffer.clear()
        self.g_buffer.clear()
        self.b_buffer.clear()


# Quick test
if __name__ == "__main__":
    print("Advanced Signal Processing Module")
    print("=" * 50)
    print("\nAvailable methods:")
    print("  - CHROM: Chrominance-based (De Haan & Jeanne, 2013)")
    print("  - POS: Plane-Orthogonal-to-Skin (Wang et al., 2017)")
    print("  - Green: Simple green channel baseline")
    print("\nUsage:")
    print("  processor = SignalProcessor(buffer_size=180, fps=30, method='chrom')")
    print("  processor.add_sample((r, g, b))")
    print("  pulse = processor.get_pulse_signal()")
    print("  signal, method, confidence = processor.get_best_signal()")
    
    # Demo with synthetic data
    print("\n--- Testing with synthetic pulse ---")
    np.random.seed(42)
    fps = 30
    duration = 6
    hr_hz = 1.2  # 72 BPM
    
    t = np.arange(0, duration, 1/fps)
    
    # Simulate RGB with pulse (green channel has strongest signal)
    pulse = 0.02 * np.sin(2 * np.pi * hr_hz * t)
    r = 150 + 0.5 * pulse + 0.02 * np.random.randn(len(t))
    g = 120 + pulse + 0.02 * np.random.randn(len(t))
    b = 100 + 0.3 * pulse + 0.02 * np.random.randn(len(t))
    
    processor = SignalProcessor(fps=fps, method='chrom')
    
    for i in range(len(t)):
        processor.add_sample((r[i], g[i], b[i]))
    
    signal_out, method, conf = processor.get_best_signal()
    print(f"Best method: {method}")
    print(f"Confidence: {conf:.2f}")
    print(f"Signal length: {len(signal_out)}")
