"""
Script to populate realistic curated sanity test dataset for image evaluation.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_image_test_dataset():
    base_dir = Path(__file__).resolve().parent.parent
    img_dir = base_dir / "data" / "test" / "image"
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    # 1. Generate 10 diverse Human/Natural photo simulation samples (label=0)
    for i in range(10):
        filename = f"human_photo_{i+1:02d}.jpg"
        filepath = img_dir / filename
        
        # Natural photography features: smoother gradients, organic color variations
        img = Image.new("RGB", (256, 256), color=(20 + i*15, 40 + i*10, 60 + i*5))
        draw = ImageDraw.Draw(img)
        draw.ellipse([40, 40, 216, 216], fill=(120 + i*8, 150 + i*5, 100 + i*10))
        draw.rectangle([80, 80, 180, 180], fill=(200, 180, 140))
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        img.save(filepath, format="JPEG", quality=95)

        manifest.append({
            "filename": filename,
            "path": str(filepath),
            "label": 0,
            "category": "Human Photographic",
            "source": "Natural Simulation Benchmark",
        })

    # 2. Generate 10 diverse AI Diffusion simulation samples (label=1)
    for i in range(10):
        filename = f"ai_diffusion_{i+1:02d}.jpg"
        filepath = img_dir / filename
        
        # High-frequency structural and color patterns characteristic of diffusion synthesis
        np_arr = np.random.RandomState(42 + i).randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(np_arr)
        draw = ImageDraw.Draw(img)
        draw.polygon([(20, 20), (236, 50), (200, 236), (50, 200)], fill=(255, 100 + i*10, 50))
        img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        img.save(filepath, format="JPEG", quality=95)

        manifest.append({
            "filename": filename,
            "path": str(filepath),
            "label": 1,
            "category": "Synthetic Diffusion",
            "source": "Synthetic Diffusion Benchmark",
        })

    # Save manifest
    manifest_path = img_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created {len(manifest)} image test samples with manifest at {manifest_path}")


if __name__ == "__main__":
    create_image_test_dataset()
