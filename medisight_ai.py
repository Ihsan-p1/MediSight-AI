"""
MediSight-AI: Unified Multi-Modal Health Monitoring System

This module integrates all MediSight-AI models into a single inference pipeline:
- Face Detection & Landmarks (MediaPipe)
- Emotion Recognition (EmotionNet)
- Fatigue/Drowsiness Detection (DrowsinessNet)
- Pain Detection (PainNet)

"""

import torch
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import time

from models import EmotionNet, DrowsinessNet, PainNet
from face_detector import FaceDetector

from torchvision import transforms


class MediSightAI:
    """
    Unified inference system for multi-modal health monitoring.
    
    Features:
    - Real-time face detection and landmark extraction
    - Emotion recognition (7 classes)
    - Fatigue/drowsiness detection (2 classes)
    - Pain detection (3 classes)

    """
    
    def __init__(self, device: str = 'cuda', checkpoint_dir: str = 'checkpoints'):
        """
        Initialize MediSight-AI system.
        
        Args:
            device: Device to run models on ('cuda' or 'cpu')
            checkpoint_dir: Directory containing trained model checkpoints
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"MediSight-AI initialized on device: {self.device}")
        
        # Initialize face detector
        self.face_detector = FaceDetector()

        
        # Load trained models
        print("Loading trained models...")
        self.emotion_model = self._load_model(
            'emotion', EmotionNet, 7, checkpoint_dir
        )
        self.fatigue_model = self._load_model(
            'fatigue', DrowsinessNet, 2, checkpoint_dir
        )
        self.pain_model = self._load_model(
            'pain', PainNet, 3, checkpoint_dir
        )
        
        # Class labels
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.fatigue_labels = ['Drowsy', 'Non Drowsy']
        self.pain_labels = ['disgust', 'sadness', 'surprise']
        
        # Transforms for each model
        self.emotion_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.fatigue_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.pain_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        print("MediSight-AI ready!")
    
    def _load_model(self, name: str, model_class, num_classes: int, 
                    checkpoint_dir: str) -> torch.nn.Module:
        """Load a trained model from checkpoint."""
        import os
        
        model = model_class(num_classes=num_classes)
        checkpoint_path = os.path.join(checkpoint_dir, f"{name}_best.pth")
        
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"  ✓ Loaded {name} model from {checkpoint_path}")
        else:
            print(f"  ⚠ Warning: {checkpoint_path} not found, using untrained model")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def process_frame(self, frame: np.ndarray, 
                     enable_emotion: bool = True,
                     enable_fatigue: bool = True,
                     enable_pain: bool = True) -> Dict:
        """
        Process a single video frame and return all predictions.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            enable_emotion: Whether to run emotion recognition
            enable_fatigue: Whether to run fatigue detection
            enable_pain: Whether to run pain detection

            
        Returns:
            Dictionary containing:
            - 'face_detected': bool
            - 'bbox': (x, y, w, h) or None
            - 'landmarks': numpy array (468, 3) or None
            - 'emotion': {'label': str, 'confidence': float, 'probabilities': dict} or None
            - 'fatigue': {'label': str, 'confidence': float, 'probabilities': dict} or None
            - 'pain': {'label': str, 'confidence': float, 'probabilities': dict} or None

            - 'inference_time': float (milliseconds)
        """
        start_time = time.time()
        
        results = {
            'face_detected': False,
            'bbox': None,
            'landmarks': None,
            'emotion': None,
            'fatigue': None,
            'pain': None,

            'inference_time': 0.0
        }
        
        # Detect face and get ROI
        face_data = self.face_detector.get_face_roi(frame)
        
        if face_data is None:
            results['inference_time'] = (time.time() - start_time) * 1000
            return results
        
        face_roi, bbox = face_data
        results['face_detected'] = True
        results['bbox'] = bbox
        
        # Get landmarks
        landmarks = self.face_detector.get_landmarks(frame)
        results['landmarks'] = landmarks
        
        # Run emotion recognition
        if enable_emotion:
            results['emotion'] = self._predict_emotion(face_roi)
        
        # Run fatigue detection
        if enable_fatigue:
            results['fatigue'] = self._predict_fatigue(face_roi)
        
        # Run pain detection
        if enable_pain:
            results['pain'] = self._predict_pain(face_roi)
        

        
        results['inference_time'] = (time.time() - start_time) * 1000
        
        return results
    
    def _predict_emotion(self, face_roi: np.ndarray) -> Dict:
        """Run emotion recognition on face ROI."""
        # Preprocess
        input_tensor = self.emotion_transform(face_roi).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.emotion_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
        
        # Convert to dict
        prob_dict = {
            label: float(probabilities[i])
            for i, label in enumerate(self.emotion_labels)
        }
        
        return {
            'label': self.emotion_labels[predicted.item()],
            'confidence': float(confidence.item()),
            'probabilities': prob_dict
        }
    
    def _predict_fatigue(self, face_roi: np.ndarray) -> Dict:
        """Run fatigue detection on face ROI."""
        # Preprocess
        input_tensor = self.fatigue_transform(face_roi).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.fatigue_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
        
        # Convert to dict
        prob_dict = {
            label: float(probabilities[i])
            for i, label in enumerate(self.fatigue_labels)
        }
        
        return {
            'label': self.fatigue_labels[predicted.item()],
            'confidence': float(confidence.item()),
            'probabilities': prob_dict
        }
    
    def _predict_pain(self, face_roi: np.ndarray) -> Dict:
        """Run pain detection on face ROI."""
        # Preprocess
        input_tensor = self.pain_transform(face_roi).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.pain_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
        
        # Convert to dict
        prob_dict = {
            label: float(probabilities[i])
            for i, label in enumerate(self.pain_labels)
        }
        
        return {
            'label': self.pain_labels[predicted.item()],
            'confidence': float(confidence.item()),
            'probabilities': prob_dict
        }
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     display: bool = True, max_frames: Optional[int] = None) -> None:
        """
        Process a video file and optionally save annotated output.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save annotated video (optional)
            display: Whether to display video while processing
            max_frames: Maximum number of frames to process (None = all)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            results = self.process_frame(frame)
            
            # Annotate frame
            annotated = self._annotate_frame(frame, results)
            
            # Write to output
            if writer:
                writer.write(annotated)
            
            # Display
            if display:
                cv2.imshow('MediSight-AI', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames...")
            
            # Max frames limit
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        print(f"Processing complete! Processed {frame_count} frames.")
    
    def _annotate_frame(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw results on frame."""
        annotated = frame.copy()
        
        if not results['face_detected']:
            cv2.putText(annotated, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated
        
        # Draw bounding box
        if results['bbox']:
            x, y, w, h = results['bbox']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw results
        y_offset = 30
        
        if results['emotion']:
            text = f"Emotion: {results['emotion']['label']} ({results['emotion']['confidence']:.2f})"
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 30
        
        if results['fatigue']:
            text = f"Fatigue: {results['fatigue']['label']} ({results['fatigue']['confidence']:.2f})"
            color = (0, 0, 255) if results['fatigue']['label'] == 'Drowsy' else (0, 255, 0)
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
        
        if results['pain']:
            text = f"Pain: {results['pain']['label']} ({results['pain']['confidence']:.2f})"
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            y_offset += 30
        
        # Inference time
        cv2.putText(annotated, f"Inference: {results['inference_time']:.1f}ms", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return annotated


if __name__ == "__main__":
    print("Testing MediSightAI...")
    
    # Initialize system
    medisight = MediSightAI(device='cuda')
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    print("\nPress 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results = medisight.process_frame(frame)
        
        # Annotate and display
        annotated = medisight._annotate_frame(frame, results)
        cv2.imshow('MediSight-AI Test', annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()
    print("Test complete!")
