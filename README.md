# MediSight-AI Deployment Guide

## Quick Start

### 1. Install Dependencies

```bash
# Using Python 3.10.11
py -3.10 -m pip install -r requirements.txt
```

### 2. Run Demo

```bash
# Basic usage (webcam)
py -3.10 demo.py

# With options
py -3.10 demo.py --device cuda --width 1280 --height 720
```

### 3. Controls

- **Q**: Quit
- **R**: Reset rPPG buffer
- **E**: Toggle emotion recognition
- **F**: Toggle fatigue detection
- **P**: Toggle pain detection
- **H**: Toggle heart rate estimation

## File Structure

```
Mediapipe2/
├── checkpoints/              # Trained model weights
│   ├── emotion_best.pth     # Emotion model (65.91% test acc)
│   ├── fatigue_best.pth     # Fatigue model (99.97% test acc)
│   └── pain_best.pth        # Pain model (100% test acc)
├── models.py                 # Model architectures
├── face_detector.py          # MediaPipe face detection + landmarks
├── rppg_estimator.py         # Heart rate estimation
├── medisight_ai.py           # Unified inference system
├── demo.py                   # Real-time demo application
├── train.py                  # Training script
├── dataset_loader.py         # Data loading utilities
└── requirements.txt          # Dependencies
```

## API Usage

### Basic Example

```python
from medisight_ai import MediSightAI
import cv2

# Initialize
medisight = MediSightAI(device='cuda')

# Process single frame
frame = cv2.imread('image.jpg')
results = medisight.process_frame(frame)

print(f"Emotion: {results['emotion']['label']}")
print(f"Fatigue: {results['fatigue']['label']}")
print(f"Pain: {results['pain']['label']}")
if results['heart_rate']:
    print(f"Heart Rate: {results['heart_rate']['bpm']:.0f} BPM")
```

### Process Video

```python
medisight = MediSightAI(device='cuda')
medisight.process_video('input.mp4', output_path='output.mp4', display=True)
```

## Performance

### Hardware Requirements

- **Minimum**: CPU (Intel i5 or equivalent)
- **Recommended**: NVIDIA GPU (RTX 3050 or better)

### Benchmarks (RTX 3050 6GB)

| Component | Inference Time | FPS |
|-----------|---------------|-----|
| Face Detection | ~5ms | >200 |
| Landmarks (468) | ~8ms | >120 |
| Emotion Model | ~2ms | >500 |
| Fatigue Model | ~3ms | >300 |
| Pain Model | ~2ms | >500 |
| rPPG Update | ~1ms | >1000 |
| **Total Pipeline** | **~20ms** | **~50 FPS** |

## Model Details

### Emotion Recognition
- **Architecture**: Custom CNN
- **Input**: 48x48 grayscale
- **Classes**: angry, disgust, fear, happy, neutral, sad, surprise
- **Accuracy**: 65.91% (test), 66.84% (val)

### Fatigue Detection
- **Architecture**: MobileNetV2 backbone
- **Input**: 224x224 RGB
- **Classes**: Drowsy, Non Drowsy
- **Accuracy**: 99.97% (test), 100% (val)

### Pain Detection
- **Architecture**: Custom CNN
- **Input**: 48x48 grayscale
- **Classes**: disgust, sadness, surprise (pain proxies)
- **Accuracy**: 100% (test), 100% (val)

### Heart Rate (rPPG)
- **Method**: Green channel + FFT
- **Range**: 40-240 BPM
- **Window**: 10 seconds (300 frames at 30 FPS)
- **Accuracy**: Depends on lighting and subject stability

## Deployment Options

### Option 1: Standalone Application
Package with PyInstaller:
```bash
py -3.10 -m pip install pyinstaller
pyinstaller --onefile --add-data "checkpoints;checkpoints" demo.py
```

### Option 2: REST API (Flask)
```python
from flask import Flask, request, jsonify
from medisight_ai import MediSightAI
import cv2
import numpy as np

app = Flask(__name__)
medisight = MediSightAI(device='cuda')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    results = medisight.process_frame(frame)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Option 3: FastAPI (Production)
```python
from fastapi import FastAPI, File, UploadFile
from medisight_ai import MediSightAI
import cv2
import numpy as np

app = FastAPI()
medisight = MediSightAI(device='cuda')

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    results = medisight.process_frame(frame)
    return results

# Run with: uvicorn deployment:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Issue: Low FPS
- **Solution**: Ensure CUDA is available (`torch.cuda.is_available()`)
- **Solution**: Reduce camera resolution
- **Solution**: Disable unused features (e.g., `--no-pain`)

### Issue: No face detected
- **Solution**: Ensure good lighting
- **Solution**: Face camera directly
- **Solution**: Adjust `min_detection_confidence` in `FaceDetector`

### Issue: Unstable heart rate
- **Solution**: Stay still for 10+ seconds
- **Solution**: Ensure good lighting (avoid flickering)
- **Solution**: Check signal quality metric

## License

MediSight-AI is for educational and research purposes.
