"""
Heart Rate Monitor with TS-CAN Support
Integrates TS-CAN deep learning model for rPPG signal extraction
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from collections import deque

from ..detection.face_detector import FaceDetector
from ..processing import SignalProcessor, FFTAnalyzer
from ..processing.kalman_filter import KalmanHRFilter
from ..processing.low_light import LowLightEnhancer
from ..models import TSCANInference, TORCH_AVAILABLE


class HeartRateMonitorTSCAN:
    """
    Heart rate monitor using TS-CAN deep learning model.
    
    Combines facial detection with TS-CAN model for rPPG signal extraction.
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        buffer_seconds: float = 6.0,
        use_lite: bool = True,
        enable_low_light: bool = True
    ):
        """
        Initialize heart rate monitor with TS-CAN.
        
        Args:
            fps: Video frames per second
            buffer_seconds: Seconds of data to buffer
            use_lite: Use lightweight TS-CAN model for faster inference
            enable_low_light: Enable low-light frame enhancement
        """
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)
        self.use_lite = use_lite
        self.enable_low_light = enable_low_light
        
        # Check if PyTorch is available
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install PyTorch for TS-CAN support.")
        
        # OpenCV face detector
        self.detector = FaceDetector()
        
        # TS-CAN inference
        self.tscan_inference = TSCANInference(
            device='auto',
            use_lite=use_lite
        )
        
        # Signal processing for post-processing
        self.fft_analyzer = FFTAnalyzer(fps=fps)
        
        # Kalman filter for smoothing
        self.kalman_filter = KalmanHRFilter(
            initial_hr=70.0,
            process_noise=0.5,
            measurement_noise=20.0,
            spike_threshold=10.0,
            hard_clamp_range=15.0
        )
        
        # Low-light enhancer
        if enable_low_light:
            self.low_light = LowLightEnhancer(
                clip_limit=2.5,
                brightness_threshold=70.0,
                target_brightness=110.0
            )
        else:
            self.low_light = None
        
        # Smoothing buffers
        self.hr_buffer = deque(maxlen=12)
        self.confidence_buffer = deque(maxlen=12)
        
        # State
        self.frame_count = 0
        self.last_detection = None
        self.last_valid_hr = 0.0
        
        # Signal buffer for FFT
        self.signal_buffer = deque(maxlen=self.buffer_size)
        
        # Quality thresholds
        self.min_confidence = 0.30
        self.hr_change_threshold = 15
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single video frame using TS-CAN model.
        
        Args:
            frame: BGR image from video
            
        Returns:
            Dictionary with results
        """
        self.frame_count += 1
        
        # Apply low-light enhancement if enabled
        if self.low_light is not None:
            brightness, needs_enhancement = self.low_light.analyze_brightness(frame)
            if needs_enhancement:
                frame = self.low_light.enhance(frame)
        
        # Detect face
        detection = self.detector.detect(frame)
        
        if detection is None:
            return self._create_result(
                frame, None, 0.0, 0.0, False, 'No face detected'
            )
            
        self.last_detection = detection
        
        # Get face bounding box
        bbox = detection['bbox']
        
        # Process frame with TS-CAN
        signal = self.tscan_inference.process_frame(frame, bbox)
        
        if signal is None:
            return self._create_result(
                frame, detection, 0.0, 0.0, True,
                f'Collecting data... {len(self.tscan_inference.preprocessor.frames)}/64'
            )
        
        # Add signal to buffer
        self.signal_buffer.extend(signal)
        
        # Check if we have enough data for FFT
        if len(self.signal_buffer) < self.buffer_size // 2:
            buffer_pct = len(self.signal_buffer) / self.buffer_size * 100
            return self._create_result(
                frame, detection, 0.0, 0.0, True,
                f'Processing signal... {buffer_pct:.0f}%'
            )
        
        # Extract heart rate using FFT
        heart_rate, fft_confidence = self.fft_analyzer.get_heart_rate(np.array(self.signal_buffer))
        
        # Apply outlier rejection and smoothing
        smoothed_hr = self._smooth_heart_rate(heart_rate, fft_confidence)
        
        # Create result
        status = 'TS-CAN Deep Learning'
        
        return self._create_result(
            frame, detection, smoothed_hr, fft_confidence, True, status,
            raw_signal=np.array(self.signal_buffer),
            method='tscan'
        )
    
    def _smooth_heart_rate(self, hr: float, confidence: float) -> float:
        """
        Apply Kalman filtering with outlier rejection.
        """
        
        # Reject very low confidence readings entirely
        if confidence < self.min_confidence:
            return self.last_valid_hr if self.last_valid_hr > 0 else 0.0
        
        # Apply Kalman filter
        kalman_hr = self.kalman_filter.update(hr, confidence)
        
        # Add to buffer for weighted averaging
        self.hr_buffer.append(kalman_hr)
        self.confidence_buffer.append(confidence)
        
        if len(self.hr_buffer) == 0:
            return 0.0
        
        # Weighted average based on confidence
        hrs = np.array(self.hr_buffer)
        confs = np.array(self.confidence_buffer)
        
        # Remove obvious outliers
        if len(hrs) >= 3:
            median = np.median(hrs)
            mad = np.median(np.abs(hrs - median))
            valid_mask = np.abs(hrs - median) < 3 * (mad + 5)
            hrs = hrs[valid_mask]
            confs = confs[valid_mask]
        
        if len(hrs) == 0:
            return self.last_valid_hr if self.last_valid_hr > 0 else 0.0
        
        # Weighted average
        smoothed_hr = np.average(hrs, weights=confs)
        
        self.last_valid_hr = smoothed_hr
        
        return smoothed_hr

    def _create_result(
        self,
        frame: np.ndarray,
        detection: Optional[dict],
        heart_rate: float,
        confidence: float,
        face_detected: bool,
        status: str,
        raw_signal: np.ndarray = None,
        method: str = None
    ) -> Dict[str, Any]:
        """Create result dictionary with annotated frame."""
        
        annotated = frame.copy()
        
        # Draw face detection
        if detection is not None:
            annotated = self.detector.draw_detection(annotated, detection)
        
        # Add HR text
        if heart_rate > 0:
            hr_text = f"HR: {heart_rate:.0f} BPM"
            conf_text = f"Conf: {confidence:.2f}"
            method_text = f"Method: {method.upper()}"
            
            # Color based on confidence
            if confidence >= 0.7:
                color = (0, 255, 0)  # Green
            elif confidence >= 0.4:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            cv2.putText(annotated, hr_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(annotated, conf_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(annotated, method_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        result = {
            'heart_rate': heart_rate,
            'confidence': confidence,
            'face_detected': face_detected,
            'status': status,
            'frame_annotated': annotated
        }
        
        if raw_signal is not None:
            result['raw_signal'] = raw_signal
        if method is not None:
            result['method'] = method
            
        return result

    def reset(self):
        """
        Reset the monitor state and TS-CAN buffers.
        """
        # Reset TS-CAN inference
        self.tscan_inference.reset()
        
        # Reset signal buffer
        self.signal_buffer.clear()
        
        # Reset smoothing buffers
        self.hr_buffer.clear()
        self.confidence_buffer.clear()
        
        # Reset state
        self.frame_count = 0
        self.last_detection = None
        self.last_valid_hr = 0.0


# Quick test
if __name__ == "__main__":
    print("Heart Rate Monitor - TS-CAN Edition")
    print("=" * 50)
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    
    if TORCH_AVAILABLE:
        print("Features:")
        print("  - TS-CAN deep learning model")
        print("  - Temporal shift convolution with attention")
        print("  - Real-time face detection")
        print("  - Kalman filtering for stable readings")
