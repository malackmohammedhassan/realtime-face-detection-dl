# Dataset Preparation for Scratch CNN Training

## Quick Start

Create the following directory structure and add your images:

```
data/
├── train/
│   ├── face/      # 500+ face images (any format)
│   └── non_face/  # 500+ non-face images
└── val/
    ├── face/      # 100+ face images
    └── non_face/  # 100+ non-face images
```

## Dataset Requirements

### Image Format

- **Supported formats**: .jpg, .png, .bmp, .webp
- **Recommended**: .jpg for compatibility
- **Size**: Any size (model automatically resizes to 64×64)
- **Quality**: Good quality preferred, but lower quality works too

### Face Images

- **Clear** facial features (eyes, nose, mouth visible)
- **Variety**: Different angles, lighting, ages, ethnicities
- **No**: People too far away (must be recognizable as face)

### Non-Face Images

- **Absolutely NO visible human faces**
- **Examples**: Landscapes, buildings, animals, objects, textures, backgrounds
- **Variety**: Different environments and lighting conditions

### Dataset Balance

- **Equal numbers**: Same count of faces and non-faces (e.g., 500 + 500)
- **Why**: Model learns equally well from both classes
- **Minimum**: 100 of each class (better: 500+)

## Dataset Sources

### Free Face Image Collections

| Source                                                        | Size        | Quality | License   |
| ------------------------------------------------------------- | ----------- | ------- | --------- |
| [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) | 13K images  | High    | Public    |
| [WIDER-FACE](http://shuoyang1213.me/WIDERFACE/)               | 32K images  | Varied  | CC-BY 4.0 |
| [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)   | 200K images | High    | CC-BY 3.0 |
| [UTKFace](https://susanqq.github.io/UTKFace/)                 | 20K images  | Medium  | Public    |
| [Asian Face Database](https://pan.baidu.com/)                 | 10K images  | High    | Academic  |

### Free Non-Face (Background) Collections

| Source                                   | Size                     | Quality | License   |
| ---------------------------------------- | ------------------------ | ------- | --------- |
| [COCO Dataset](https://cocodataset.org/) | 330K images              | High    | CC-BY 4.0 |
| [ImageNet](http://www.image-net.org/)    | 14M images               | Varied  | Research  |
| [Unsplash](https://unsplash.com/)        | 1M+ images               | High    | Free      |
| [Pexels](https://www.pexels.com/)        | High quality free images | High    | Free      |
| [Pixabay](https://pixabay.com/)          | High quality free images | High    | Free      |

## Preparation Steps

### Option 1: Use Your Own Photos (Quickest)

```powershell
# Create directories
mkdir -p data\train\face
mkdir -p data\train\non_face
mkdir -p data\val\face
mkdir -p data\val\non_face

# Copy face photos (selfies, portraits, etc.)
copy my_face_photo1.jpg data\train\face\
copy my_face_photo2.jpg data\train\face\
copy my_selfies\*.jpg data\train\face\

# About 20% to validation
move data\train\face\*_val.jpg data\val\face\
# Or manually select 20% files

# Copy background images (landscapes, objects, etc.)
copy background1.jpg data\train\non_face\
copy background2.jpg data\train\non_face\
copy nature_photos\*.jpg data\train\non_face\

# Move ~20% to validation
move data\train\non_face\*_val.jpg data\val\non_face\
```

### Option 2: Download from LFW Dataset

```powershell
# Create directories
mkdir -p data\train\face data\val\face

# Download LFW tar.gz (13K face images)
# Link: http://vis-www.cs.umass.edu/lfw/lfw.tgz
# Then extract and organize

# For non-faces, download subset from COCO or Unsplash
mkdir -p data\train\non_face data\val\non_face
# Download urban/landscape photos (~1000 images)
```

### Option 3: Generate from Web Search

```powershell
# Usage: Download images using Bing Image Search (with permission)
# Tools:
# - Bing Image Search API
# - Google Images with Download All extension
# - Flickr API for free licensed images

# Manual steps:
# 1. Search "face portrait" on Unsplash/Pexels
# 2. Download 500+ face images
# 3. Search "landscape background"
# 4. Download 500+ non-face images
# 5. Organize into data/train/ and data/val/
```

## Validation Checklist

Before training, verify:

```powershell
# Count images in each directory
[Object[]]$face_train = Get-ChildItem data\train\face\ -Include *.jpg,*.png
[Object[]]$nonface_train = Get-ChildItem data\train\non_face\ -Include *.jpg,*.png
[Object[]]$face_val = Get-ChildItem data\val\face\ -Include *.jpg,*.png
[Object[]]$nonface_val = Get-ChildItem data\val\non_face\ -Include *.jpg,*.png

Write-Host "Training: $($face_train.Count) faces, $($nonface_train.Count) non-faces"
Write-Host "Validation: $($face_val.Count) faces, $($nonface_val.Count) non-faces"
```

Expected output:

```
Training: 500 faces, 500 non-faces
Validation: 100 faces, 100 non-faces
```

## Important Notes

### ⚠️ Non-Face Dataset Warning

**CRITICAL**: Non-face images **must not contain human faces**. If they do:

- Model will learn incorrectly
- Accuracy will be poor (50%)
- Training metrics will be misleading

Check your non-face images visually!

### 📊 Minimum Dataset Sizes

| Size                      | Training Time | Expected Accuracy |
| ------------------------- | ------------- | ----------------- |
| 100 faces + 100 non-faces | ~1 minute     | 65-75%            |
| 300 + 300                 | ~5 minutes    | 75-85%            |
| 500 + 500                 | ~10 minutes   | 85-92%            |
| 1000 + 1000               | ~30 minutes   | 92-96%            |
| 2000+ + 2000+             | ~90 minutes   | 96%+              |

### 🎨 Data Quality vs. Quantity

**Better to have:**

- 500 good quality, diverse images than
- 2000 low quality, repetitive images

Focus on **variety** over quantity.

## Common Mistakes to Avoid

❌ **Wrong**: All faces from same angle/lighting
✅ **Right**: Faces at different angles, lighting, expressions

❌ **Wrong**: Non-face images with partially visible faces
✅ **Right**: Non-face images with zero human faces

❌ **Wrong**: Very small faces (< 20 pixels)
✅ **Right**: Reasonably sized, recognizable faces

❌ **Wrong**: Only frontal faces
✅ **Right**: Frontal, profile, and angled faces

## After Dataset Preparation

Once you've organized your data:

```powershell
# Activate environment
.\venv\Scripts\activate

# Start training
python scripts/train_scratch_cnn.py

# Monitor progress
# Expected: ~5 minutes for 500+500 images on CPU
```

## Dataset Citation and Licensing

When using public datasets, always:

1. **Cite the source** in your academic report
2. **Respect the license** (CC-BY, CC-BY-4.0, etc.)
3. **Document where images came from**
4. **Mention any preprocessing** (resizing, cropping, etc.)

Example for your report:

> "Face images sourced from Labeled Faces in the Wild [1], non-face images from COCO dataset [2]. Total: 1000 face images, 1000 non-face images, split 80/20 train/val."

---

**Questions?** See [SCRATCH_CNN_TRAINING.md](SCRATCH_CNN_TRAINING.md) for training instructions.
