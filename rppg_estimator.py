"""
rPPG (Remote Photoplethysmography) Heart Rate Estimator

This module implements heart rate estimation from video frames using the
green channel method and FFT-based frequency analysis.

Reference: Verkruysse et al., "Remote plethysmographic imaging using ambient light"
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from collections import deque
from typing import Optional, Tuple


class rPPGEstimator:
    """
    Heart rate estimation using remote photoplethysmography (rPPG).
    
    Method:
    1. Extract face ROI from video frames
    2. Compute mean green channel value (most sensitive to blood volume changes)
    3. Buffer signal over time window (typically 10 seconds)
    4. Apply bandpass filter to remove noise
    5. Use FFT to find dominant frequency in physiological range (40-240 BPM)
    """
    
    def __init__(self, 
                 fps: int = 30,
                 window_size: int = 300,
                 min_bpm: int = 40,
                 max_bpm: int = 240):
        """
        Initialize rPPG estimator.
        
        Args:
            fps: Frames per second of input video
            window_size: Number of frames to buffer (default: 300 = 10 seconds at 30 FPS)
            min_bpm: Minimum heart rate in BPM
            max_bpm: Maximum heart rate in BPM
        """
        self.fps = fps
        self.window_size = window_size
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        
        # Signal buffer (stores green channel mean values)
        self.signal_buffer = deque(maxlen=window_size)
        
        # Bandpass filter parameters
        # Convert BPM to Hz: BPM / 60
        self.lowcut = min_bpm / 60.0
        self.highcut = max_bpm / 60.0
        
        # Design Butterworth bandpass filter
        self.filter_order = 4
        self.sos = signal.butter(
            self.filter_order, 
            [self.lowcut, self.highcut], 
            btype='bandpass', 
            fs=fps, 
            output='sos'
        )
        
    def process_frame(self, face_roi: np.ndarray) -> None:
        """
        Extract green channel mean from face ROI and add to buffer.
        
        Args:
            face_roi: Face region of interest (BGR image from OpenCV)
        """
        if face_roi is None or face_roi.size == 0:
            return
        
        # Extract green channel (index 1 in BGR)
        green_channel = face_roi[:, :, 1]
        
        # Compute mean value
        green_mean = np.mean(green_channel)
        
        # Add to buffer
        self.signal_buffer.append(green_mean)
    
    def estimate_heart_rate(self) -> Optional[Tuple[float, float]]:
        """
        Estimate heart rate using FFT on buffered signal.
        
        Returns:
            Tuple of (heart_rate_bpm, confidence) or None if insufficient data
            - heart_rate_bpm: Estimated heart rate in beats per minute
            - confidence: Signal quality metric (0-1, higher is better)
        """
        # Need at least half window for reliable estimation
        if len(self.signal_buffer) < self.window_size // 2:
            return None
        
        # Convert buffer to numpy array
        raw_signal = np.array(self.signal_buffer)
        
        # Detrend signal (remove DC component and linear trend)
        detrended = signal.detrend(raw_signal)
        
        # Apply bandpass filter
        filtered_signal = signal.sosfilt(self.sos, detrended)
        
        # Compute FFT
        n = len(filtered_signal)
        yf = fft(filtered_signal)
        xf = fftfreq(n, 1 / self.fps)
        
        # Take only positive frequencies
        positive_freqs = xf[:n // 2]
        positive_fft = np.abs(yf[:n // 2])
        
        # Find frequencies in physiological range
        freq_mask = (positive_freqs >= self.lowcut) & (positive_freqs <= self.highcut)
        valid_freqs = positive_freqs[freq_mask]
        valid_fft = positive_fft[freq_mask]
        
        if len(valid_fft) == 0:
            return None
        
        # Find peak frequency (dominant heart rate)
        peak_idx = np.argmax(valid_fft)
        peak_freq = valid_freqs[peak_idx]
        
        # Convert Hz to BPM
        heart_rate_bpm = peak_freq * 60.0
        
        # Compute confidence metric
        # Use ratio of peak power to total power in valid range
        peak_power = valid_fft[peak_idx]
        total_power = np.sum(valid_fft)
        confidence = peak_power / total_power if total_power > 0 else 0.0
        
        return heart_rate_bpm, confidence
    
    def get_signal_quality(self) -> float:
        """
        Compute signal quality metric based on signal-to-noise ratio.
        
        Returns:
            Quality score (0-1, higher is better)
        """
        if len(self.signal_buffer) < 30:  # Need at least 1 second
            return 0.0
        
        raw_signal = np.array(self.signal_buffer)
        
        # Compute signal variance
        signal_var = np.var(raw_signal)
        
        # Estimate noise variance (high-frequency components)
        # Use difference between consecutive samples
        noise = np.diff(raw_signal)
        noise_var = np.var(noise)
        
        # SNR in dB
        if noise_var > 0:
            snr = 10 * np.log10(signal_var / noise_var)
            # Normalize to 0-1 (assume SNR range of 0-30 dB)
            quality = np.clip(snr / 30.0, 0.0, 1.0)
        else:
            quality = 1.0
        
        return quality
    
    def reset(self):
        """Clear the signal buffer."""
        self.signal_buffer.clear()
    
    def get_raw_signal(self) -> np.ndarray:
        """
        Get the raw buffered signal for visualization.
        
        Returns:
            Numpy array of buffered green channel values
        """
        return np.array(self.signal_buffer)
    
    def get_filtered_signal(self) -> Optional[np.ndarray]:
        """
        Get the filtered signal for visualization.
        
        Returns:
            Numpy array of filtered signal or None if insufficient data
        """
        if len(self.signal_buffer) < 30:
            return None
        
        raw_signal = np.array(self.signal_buffer)
        detrended = signal.detrend(raw_signal)
        filtered_signal = signal.sosfilt(self.sos, detrended)
        
        return filtered_signal


if __name__ == "__main__":
    # Test the rPPG estimator
    print("Testing rPPGEstimator...")
    
    import cv2
    from face_detector import FaceDetector
    
    # Initialize
    face_detector = FaceDetector()
    rppg_estimator = rPPGEstimator(fps=30, window_size=300)
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    
    print(f"Camera FPS: {fps}")
    print("Press 'q' to quit, 'r' to reset buffer")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get face ROI
        face_data = face_detector.get_face_roi(frame)
        
        if face_data is not None:
            face_roi, bbox = face_data
            
            # Process frame
            rppg_estimator.process_frame(face_roi)
            
            # Estimate heart rate every 30 frames (1 second)
            if frame_count % 30 == 0:
                hr_result = rppg_estimator.estimate_heart_rate()
                quality = rppg_estimator.get_signal_quality()
                
                if hr_result is not None:
                    hr_bpm, confidence = hr_result
                    print(f"Heart Rate: {hr_bpm:.1f} BPM | Confidence: {confidence:.2f} | Quality: {quality:.2f}")
                else:
                    print("Buffering... (need more frames)")
            
            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display buffer status
            buffer_pct = len(rppg_estimator.signal_buffer) / rppg_estimator.window_size * 100
            cv2.putText(frame, f"Buffer: {buffer_pct:.0f}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('rPPG Heart Rate Estimation', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            rppg_estimator.reset()
            print("Buffer reset")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test complete!")
