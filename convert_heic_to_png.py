"""Convert .heic photos in iphone_photos/ to .png files in png_photos/.

Requires: pip install pillow pillow-heif
"""
from pathlib import Path

from PIL import Image
import pillow_heif


def main() -> None:
    project_dir = Path(__file__).parent
    src_dir = project_dir / "iphone_photos"
    dst_dir = project_dir / "png_photos"
    dst_dir.mkdir(exist_ok=True)

    pillow_heif.register_heif_opener()

    heic_files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() == ".heic"]
    if not heic_files:
        print("No .heic files found in iphone_photos/.")
        return

    for heic_path in heic_files:
        png_path = dst_dir / f"{heic_path.stem}.png"
        with Image.open(heic_path) as img:
            img.save(png_path, format="PNG")
        print(f"Converted {heic_path.name} -> {png_path.relative_to(project_dir)}")


if __name__ == "__main__":
    main()
