"""
MediSight-AI Real-Time Demo Application

Interactive webcam demo showcasing all MediSight-AI capabilities:
- Face detection and landmarks
- Emotion recognition
- Fatigue/drowsiness detection
- Pain detection
- Heart rate estimation (rPPG)
"""

import cv2
import numpy as np
from medisight_ai import MediSightAI
import argparse


def draw_ui_panel(frame, results, fps):
    """Draw a professional UI panel with all results."""
    h, w = frame.shape[:2]
    
    # Create semi-transparent overlay for UI panel
    overlay = frame.copy()
    panel_height = 200
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Title
    cv2.putText(frame, "MediSight-AI", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if not results['face_detected']:
        cv2.putText(frame, "No face detected", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame
    
    # Draw bounding box
    if results['bbox']:
        x, y, w_box, h_box = results['bbox']
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
    
    # Results panel
    y_offset = 70
    line_height = 30
    
    # Emotion
    if results['emotion']:
        emotion = results['emotion']
        text = f"Emotion: {emotion['label'].upper()}"
        conf_text = f"{emotion['confidence']*100:.1f}%"
        
        # Color based on emotion
        color_map = {
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'angry': (0, 0, 255),
            'fear': (128, 0, 128),
            'disgust': (0, 128, 128),
            'surprise': (255, 255, 0),
            'neutral': (200, 200, 200)
        }
        color = color_map.get(emotion['label'], (255, 255, 255))
        
        cv2.putText(frame, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, conf_text, (300, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += line_height
    
    # Fatigue
    if results['fatigue']:
        fatigue = results['fatigue']
        text = f"Fatigue: {fatigue['label'].upper()}"
        conf_text = f"{fatigue['confidence']*100:.1f}%"
        
        color = (0, 0, 255) if fatigue['label'] == 'Drowsy' else (0, 255, 0)
        
        cv2.putText(frame, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, conf_text, (300, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += line_height
    
    # Pain
    if results['pain']:
        pain = results['pain']
        text = f"Pain: {pain['label'].upper()}"
        conf_text = f"{pain['confidence']*100:.1f}%"
        
        cv2.putText(frame, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame, conf_text, (300, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        y_offset += line_height
    
    # Heart Rate
    if results['heart_rate']:
        hr = results['heart_rate']
        text = f"Heart Rate: {hr['bpm']:.0f} BPM"
        quality_text = f"Quality: {hr['quality']*100:.0f}%"
        
        # Color based on quality
        if hr['quality'] > 0.7:
            color = (0, 255, 0)
        elif hr['quality'] > 0.4:
            color = (0, 255, 255)
        else:
            color = (0, 165, 255)
        
        cv2.putText(frame, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, quality_text, (300, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += line_height
    else:
        cv2.putText(frame, "Heart Rate: Buffering...", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        y_offset += line_height
    
    # Inference time
    cv2.putText(frame, f"Inference: {results['inference_time']:.1f}ms", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description="MediSight-AI Real-Time Demo")
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'], help='Device to run on')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=1280, 
                       help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720, 
                       help='Camera height (default: 720)')
    parser.add_argument('--no-emotion', action='store_true', 
                       help='Disable emotion recognition')
    parser.add_argument('--no-fatigue', action='store_true', 
                       help='Disable fatigue detection')
    parser.add_argument('--no-pain', action='store_true', 
                       help='Disable pain detection')
    parser.add_argument('--no-rppg', action='store_true', 
                       help='Disable heart rate estimation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MediSight-AI Real-Time Demo")
    print("=" * 60)
    
    # Initialize MediSight-AI
    print("\nInitializing MediSight-AI...")
    medisight = MediSightAI(device=args.device)
    
    # Open camera
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    
    print("\n" + "=" * 60)
    print("Controls:")
    print("  Q - Quit")
    print("  R - Reset rPPG buffer")
    print("  E - Toggle emotion recognition")
    print("  F - Toggle fatigue detection")
    print("  P - Toggle pain detection")
    print("  H - Toggle heart rate estimation")
    print("=" * 60)
    
    # State
    enable_emotion = not args.no_emotion
    enable_fatigue = not args.no_fatigue
    enable_pain = not args.no_pain
    enable_rppg = not args.no_rppg
    
    # FPS calculation
    import time
    fps_buffer = []
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Process frame
        results = medisight.process_frame(
            frame,
            enable_emotion=enable_emotion,
            enable_fatigue=enable_fatigue,
            enable_pain=enable_pain,
            enable_rppg=enable_rppg
        )
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        fps_buffer.append(fps)
        if len(fps_buffer) > 30:
            fps_buffer.pop(0)
        avg_fps = np.mean(fps_buffer)
        
        # Draw UI
        annotated = draw_ui_panel(frame, results, avg_fps)
        
        # Display
        cv2.imshow('MediSight-AI Demo', annotated)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('r'):
            medisight.reset_rppg()
            print("rPPG buffer reset")
        elif key == ord('e'):
            enable_emotion = not enable_emotion
            print(f"Emotion recognition: {'ON' if enable_emotion else 'OFF'}")
        elif key == ord('f'):
            enable_fatigue = not enable_fatigue
            print(f"Fatigue detection: {'ON' if enable_fatigue else 'OFF'}")
        elif key == ord('p'):
            enable_pain = not enable_pain
            print(f"Pain detection: {'ON' if enable_pain else 'OFF'}")
        elif key == ord('h'):
            enable_rppg = not enable_rppg
            print(f"Heart rate estimation: {'ON' if enable_rppg else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Demo complete!")


if __name__ == "__main__":
    main()
