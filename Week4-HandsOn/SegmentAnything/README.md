# Segment Anything Model (SAM) Baseline

# Segment Anything Model (SAM) Baseline

## Download the SAM checkpoint

Before running the demo, fetch the model weights:

```bash
cd Week4-HandsOn/SegmentAnything
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -o sam_vit_b.pth

## Model & Source
- **Name:** Segment Anything Model (SAM), ViT-B variant  
- **Source:** Barsellotti, Luca, et al. “Personalized Instance-Based Navigation toward User-Specific Objects in Realistic Environments.” NeurIPS 2024  
- **Code:** https://github.com/facebookresearch/segment-anything

## Environment & Installation
- **OS:** macOS  
- **Python:** 3.9 (inside `.venv`)  
- **Packages:**  
  - torch, torchvision  
  - pillow  
  - segment-anything (git+https://github.com/facebookresearch/segment-anything.git)

```bash
# from repo root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision pillow
pip install git+https://github.com/facebookresearch/segment-anything.git

