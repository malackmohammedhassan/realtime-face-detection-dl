# Model Weights Directory

This directory is reserved for storing face detection model weights.

## Usage

When implementing face detection models (MTCNN, RetinaFace, etc.), place model weights here:

```
models/
├── mtcnn_weights/
│   ├── pnet.pth
│   ├── rnet.pth
│   └── onet.pth
├── retinaface/
│   └── mobilenet025.pth
└── yolov8/
    └── yolov8m.pt
```

## Auto-Download

Most libraries will automatically download and cache weights:

```python
# PyTorch Hub
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

# Torchvision models
from torchvision.models import resnet50
model = resnet50(pretrained=True)

# Specialized libraries (may cache locally)
# Verify documentation for your chosen detection model
```

## Git Ignore

This directory is git-ignored (see `.gitignore`) to avoid committing large binary files. Store models locally or in cloud storage (e.g., S3, Hugging Face Hub).
