"""
Heart Rate Monitor - Production Grade
Uses MediaPipe Face Mesh for accurate forehead ROI detection
With Kalman filtering, spike rejection, and low-light compensation
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from collections import deque

from ..detection.face_detector import FaceDetector
from ..processing import SignalProcessor, SimpleSignalProcessor, FFTAnalyzer
from ..processing.kalman_filter import KalmanHRFilter
from ..processing.low_light import LowLightEnhancer


class HeartRateMonitor:
    """
    Production-grade heart rate monitoring from video.
    
    Uses MediaPipe Face Mesh for accurate landmark-based ROI detection.
    Includes Kalman filtering for smooth, stable readings.
    Low-light enhancement for webcam in poor conditions.
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        buffer_seconds: float = 6.0,
        roi_region: str = 'forehead',
        smoothing_window: int = 12,
        method: str = 'chrome_dehaan',  # Updated default to chrome_dehaan
        enable_low_light: bool = True
    ):
        """
        Initialize heart rate monitor.
        
        Args:
            fps: Video frames per second
            buffer_seconds: Seconds of data to buffer
            roi_region: Face region to use (forehead, left_cheek, right_cheek)
            smoothing_window: Number of readings to smooth over
            method: Signal extraction method ('chrom', 'pos', 'chrome_dehaan', 'auto')
            enable_low_light: Enable low-light frame enhancement
        """
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)
        self.method = method
        self.roi_region = roi_region
        self.enable_low_light = enable_low_light
        
        # OpenCV face detector
        self.detector = FaceDetector()
        
        # Signal processing
        if method == 'simple':
            # Use SimpleSignalProcessor for basic processing
            self.signal_processor = SimpleSignalProcessor()
        else:
            # Use advanced SignalProcessor
            self.signal_processor = SignalProcessor(
                buffer_size=self.buffer_size,
                fps=fps,
                method='chrome_dehaan' if method != 'auto' else 'chrome_dehaan'
            )
        self.fft_analyzer = FFTAnalyzer(fps=fps)
        self.use_simple_processor = (method == 'simple')
        
        # Kalman filter with aggressive spike rejection
        self.kalman_filter = KalmanHRFilter(
            initial_hr=70.0,
            process_noise=0.2,        # Very low for maximum stability
            measurement_noise=25.0,   # Very high for maximum noise rejection
            spike_threshold=8.0,      # Very tight threshold for outliers
            hard_clamp_range=15.0     # Never deviate more than ±15 BPM
        )
        
        # Low-light enhancer
        if enable_low_light:
            self.low_light = LowLightEnhancer(
                clip_limit=2.5,
                brightness_threshold=70.0,  # Enhance if darker than this
                target_brightness=110.0
            )
        else:
            self.low_light = None
        
        # Smoothing buffers
        self.hr_buffer = deque(maxlen=smoothing_window)
        self.confidence_buffer = deque(maxlen=smoothing_window)
        
        # State
        self.frame_count = 0
        self.last_detection = None
        self.last_valid_hr = 0.0
        
        # Quality thresholds - optimized for noise reduction
        self.min_confidence = 0.20   # Higher for better signal quality
        self.hr_change_threshold = 12  # Tighter control for stability
        self.confidence_gate = 0.60   # Higher for more reliable readings
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single video frame using MediaPipe landmarks.
        
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
        
        # Detect face with MediaPipe
        detection = self.detector.detect(frame)
        
        if detection is None:
            return self._create_result(
                frame, None, 0.0, 0.0, False, 'No face detected'
            )
            
        self.last_detection = detection
        
        # Get mean RGB from face bounding box
        x, y, w, h = detection['bbox']
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return self._create_result(
                frame, detection, 0.0, 0.0, True, 'ROI extraction failed'
            )
        
        # Calculate mean RGB
        mean_rgb = cv2.mean(face_roi)[:3]
        
        if mean_rgb == (0.0, 0.0, 0.0):
            return self._create_result(
                frame, detection, 0.0, 0.0, True, 'ROI extraction failed'
            )



        
        # Add to signal processor
        if self.use_simple_processor:
            # Use SimpleSignalProcessor's process_rgb method
            processed_value = self.signal_processor.process_rgb(mean_rgb)
        else:
            # Use advanced SignalProcessor
            self.signal_processor.add_sample(mean_rgb)
        
        # Check if we have enough data
        if self.use_simple_processor:
            # Simple processor needs at least 30 samples
            if len(self.signal_processor.buffer) < 30:
                buffer_pct = len(self.signal_processor.buffer) / 30 * 100
                return self._create_result(
                    frame, detection, 0.0, 0.0, True,
                    f'Collecting data... {buffer_pct:.0f}%'
                )
        else:
            # Advanced processor check
            if not self.signal_processor.is_ready():
                buffer_pct = len(self.signal_processor.g_buffer) / (self.buffer_size // 2) * 100
                return self._create_result(
                    frame, detection, 0.0, 0.0, True,
                    f'Collecting data... {buffer_pct:.0f}%'
                )
        
        # Get pulse signal
        if self.use_simple_processor:
            # For simple processor, build a signal array from buffer
            signal = np.array(self.signal_processor.buffer)
            method_used = 'simple'
            # Calculate simple confidence based on signal variation
            if len(signal) > 0:
                sig_std = np.std(signal)
                sig_quality = min(1.0, max(0.0, sig_std * 10))
            else:
                sig_quality = 0.0
        else:
            # Use advanced SignalProcessor
            if self.method == 'auto':
                signal, method_used, sig_quality = self.signal_processor.get_best_signal()
            else:
                signal = self.signal_processor.get_pulse_signal()
                method_used = self.method
                sig_quality = self.signal_processor.get_signal_quality()
        
        if len(signal) == 0:
            return self._create_result(
                frame, detection, 0.0, 0.0, True, 'Signal extraction failed'
            )
        
        # Extract heart rate
        heart_rate, fft_confidence = self.fft_analyzer.get_heart_rate(signal)
        
        # Combine confidences
        confidence = (sig_quality + fft_confidence) / 2
        
        # Apply outlier rejection and smoothing
        smoothed_hr = self._smooth_heart_rate(heart_rate, confidence)
        
        # Create result
        status = f'Measuring ({method_used.upper()})'
        
        return self._create_result(
            frame, detection, smoothed_hr, confidence, True, status,
            raw_signal=signal,
            method=method_used
        )
    
    def _smooth_heart_rate(self, hr: float, confidence: float) -> float:
        """
        Apply Kalman filtering with outlier rejection.
        
        The Kalman filter provides:
        - Smooth, stable readings
        - Automatic spike rejection
        - Confidence-adaptive noise model
        """
        
        # Reject very low confidence readings entirely
        if confidence < self.min_confidence:
            return self.last_valid_hr if self.last_valid_hr > 0 else 0.0
        
        # Apply Kalman filter (handles spike rejection internally)
        kalman_hr = self.kalman_filter.update(hr, confidence)
        
        # Add to buffer for weighted averaging
        self.hr_buffer.append(kalman_hr)
        self.confidence_buffer.append(confidence)
        
        if len(self.hr_buffer) == 0:
            return 0.0
        
        # Weighted average based on confidence for extra stability
        hrs = np.array(self.hr_buffer)
        confs = np.array(self.confidence_buffer)
        
        # Remove obvious outliers (median-based)
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
        
        # Draw face detection with OpenCV
        if detection is not None:
            annotated = self.detector.draw_detection(annotated, detection)
        
        # Add HR text
        if heart_rate > 0:
            hr_text = f"HR: {heart_rate:.0f} BPM"
            conf_text = f"Conf: {confidence:.2f}"
            
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
        """Reset the monitor."""
        self.signal_processor.clear()
        self.hr_buffer.clear()
        self.confidence_buffer.clear()
        self.frame_count = 0
        self.last_valid_hr = 0.0
        
    def get_signal_plot_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get data for plotting the signal."""
        r, g, b = self.signal_processor.get_raw_signal()
        processed = self.signal_processor.get_pulse_signal()
        
        time_raw = np.arange(len(g)) / self.fps
        time_proc = np.arange(len(processed)) / self.fps if len(processed) > 0 else np.array([])
            
        return (time_raw, g, processed)


# Quick test
if __name__ == "__main__":
    print("Heart Rate Monitor - MediaPipe Edition")
    print("=" * 50)
    print("Features:")
    print("  - MediaPipe Face Mesh for precise detection")
    print("  - Landmark-based forehead ROI extraction")
    print("  - CHROM/POS signal extraction methods")
    print("  - Outlier rejection and robust smoothing")

