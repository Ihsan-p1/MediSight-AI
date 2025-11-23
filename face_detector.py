"""
Face Detection and Landmark Extraction using MediaPipe

This module provides a wrapper around MediaPipe's face detection and face mesh
solutions for easy integration into the MediSight-AI system.
"""

import mediapipe as mp
import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """
    Face detection and landmark extraction using MediaPipe.
    
    Features:
    - Fast face detection (>30 FPS on GPU)
    - 468-point facial landmarks
    - Bounding box extraction
    - ROI extraction for downstream tasks
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe face detection and face mesh.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face detection (for bounding boxes)
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full range (better for varying distances)
            min_detection_confidence=min_detection_confidence
        )
        
        # Face mesh (for landmarks)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,  # False for video
            max_num_faces=1,  # Process only one face for efficiency
            refine_landmarks=True,  # Include iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the image and return bounding boxes.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            List of bounding boxes as (x, y, w, h) tuples
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.face_detection.process(image_rgb)
        
        bboxes = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                bboxes.append((x, y, width, height))
                
        return bboxes
    
    def get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 468 facial landmarks from the image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Numpy array of shape (468, 3) with normalized x, y, z coordinates,
            or None if no face detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert to numpy array
            landmarks = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in face_landmarks.landmark
            ])
            
            return landmarks
        
        return None
    
    def get_face_roi(self, image: np.ndarray, 
                     padding: float = 0.1) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract face region of interest (ROI) with padding.
        
        Args:
            image: Input image (BGR format from OpenCV)
            padding: Padding ratio around the face (0.1 = 10% padding)
            
        Returns:
            Tuple of (face_roi, bbox) where:
            - face_roi: Cropped face image
            - bbox: Bounding box as (x, y, w, h)
            Returns None if no face detected
        """
        bboxes = self.detect_faces(image)
        
        if not bboxes:
            return None
        
        # Get first face
        x, y, w, h = bboxes[0]
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x_start = max(0, x - pad_w)
        y_start = max(0, y - pad_h)
        x_end = min(image.shape[1], x + w + pad_w)
        y_end = min(image.shape[0], y + h + pad_h)
        
        # Extract ROI
        face_roi = image[y_start:y_end, x_start:x_end]
        bbox = (x_start, y_start, x_end - x_start, y_end - y_start)
        
        return face_roi, bbox
    
    def draw_landmarks(self, image: np.ndarray, 
                      draw_detection: bool = True,
                      draw_mesh: bool = True) -> np.ndarray:
        """
        Draw face detection and landmarks on the image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            draw_detection: Whether to draw face detection bounding box
            draw_mesh: Whether to draw face mesh landmarks
            
        Returns:
            Image with drawings
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated_image = image.copy()
        
        # Draw face detection
        if draw_detection:
            detection_results = self.face_detection.process(image_rgb)
            if detection_results.detections:
                for detection in detection_results.detections:
                    self.mp_drawing.draw_detection(annotated_image, detection)
        
        # Draw face mesh
        if draw_mesh:
            mesh_results = self.face_mesh.process(image_rgb)
            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                    )
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                    )
        
        return annotated_image
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        self.face_detection.close()
        self.face_mesh.close()


if __name__ == "__main__":
    # Test the face detector
    print("Testing FaceDetector...")
    
    detector = FaceDetector()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit, 'd' to toggle detection, 'm' to toggle mesh")
    draw_detection = True
    draw_mesh = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get landmarks
        landmarks = detector.get_landmarks(frame)
        if landmarks is not None:
            print(f"Detected {len(landmarks)} landmarks")
        
        # Draw on frame
        annotated = detector.draw_landmarks(frame, draw_detection, draw_mesh)
        
        # Display
        cv2.imshow('Face Detection & Landmarks', annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            draw_detection = not draw_detection
        elif key == ord('m'):
            draw_mesh = not draw_mesh
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test complete!")
