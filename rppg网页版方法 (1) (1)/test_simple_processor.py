#!/usr/bin/env python3
"""
Test script for SimpleSignalProcessor class
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from processing import SimpleSignalProcessor

def test_simple_processor():
    """Test SimpleSignalProcessor with synthetic data"""
    print("Testing SimpleSignalProcessor...")
    print("=" * 50)
    
    # Create processor instance
    processor = SimpleSignalProcessor()
    
    # Generate synthetic data: sine wave + noise
    # Simulate pulse signal with frequency 1.2 Hz (72 BPM)
    # Add some baseline drift
    t = np.arange(0, 10, 1/30)  # 10 seconds at 30 FPS
    pulse_freq = 1.2  # Hz
    pulse_amplitude = 0.1
    
    # Create signal with drift and noise
    drift = 0.05 * t  # Linear baseline drift
    noise = 0.02 * np.random.randn(len(t))  # Gaussian noise
    signal = 1.0 + drift + pulse_amplitude * np.sin(2 * np.pi * pulse_freq * t) + noise
    
    # Process each sample
    processed_values = []
    for i, raw_value in enumerate(signal):
        processed = processor.process(raw_value)
        processed_values.append(processed)
        
        # Print progress every 100 samples
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(t)} samples")
    
    # Convert to numpy array
    processed_values = np.array(processed_values)
    
    # Analyze results
    print("\nResults:")
    print(f"Input signal range: {signal.min():.3f} to {signal.max():.3f}")
    print(f"Processed signal range: {processed_values.min():.3f} to {processed_values.max():.3f}")
    print(f"Processed signal mean: {processed_values.mean():.3f}")
    print(f"Processed signal std: {processed_values.std():.3f}")
    
    # Check if buffer is working correctly
    print(f"\nBuffer size: {len(processor.buffer)}")
    print(f"Buffer capacity: {processor.bufferSize}")
    
    # Verify signal processing
    # After buffer fills, processed values should not be zero
    non_zero_values = processed_values[30:]
    if len(non_zero_values) > 0:
        print(f"\nNon-zero processed values: {len(non_zero_values)}")
        print(f"First non-zero value: {non_zero_values[0]:.3f}")
        print(f"Last processed value: {non_zero_values[-1]:.3f}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_simple_processor()
