import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img

def save_image(path: Path, img, jpeg_quality=None):
    ensure_dir(path.parent)
    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"] and jpeg_quality is not None:
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:
        cv2.imwrite(str(path), img)

def list_images_recursive(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and is_image_file(p):
            files.append(p)
    return sorted(files)

def apply_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    out = img.astype(np.float32) * factor
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def apply_gaussian_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    out = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return out

def apply_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=sigma, size=img_f.shape).astype(np.float32)
    out = img_f + noise
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)
    return out

def apply_jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, encimg = cv2.imencode(".jpg", img, encode_param)
    if not success:
        raise ValueError("JPEG encoding failed.")
    out = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return out

def relative_to_root(file_path: Path, root: Path) -> Path:
    return file_path.relative_to(root)

def generate_single_corruption_dataset(
    input_root: Path,
    output_root: Path,
    corruption_name: str,
    severity,
    seed: int = 42
):
    np.random.seed(seed)
    image_paths = list_images_recursive(input_root)
    if len(image_paths) == 0:
        raise ValueError(f"No images found in: {input_root}")
    corruption_tag = f"{corruption_name}_{str(severity).replace('.', 'p')}"
    target_root = output_root / corruption_tag
    ensure_dir(target_root)
    print(f"\n[INFO] Generating: {corruption_tag}")
    print(f"[INFO] Input : {input_root}")
    print(f"[INFO] Output: {target_root}")
    print(f"[INFO] Number of images: {len(image_paths)}")
    for img_path in tqdm(image_paths, desc=corruption_tag):
        rel_path = relative_to_root(img_path, input_root)
        save_path = target_root / rel_path
        img = read_image(img_path)
        if corruption_name == "brightness":
            out = apply_brightness(img, float(severity))
            save_image(save_path, out)
        elif corruption_name == "blur":
            out = apply_gaussian_blur(img, int(severity))
            save_image(save_path, out)
        elif corruption_name == "noise":
            out = apply_gaussian_noise(img, float(severity))
            save_image(save_path, out)
        elif corruption_name == "jpeg":
            out = apply_jpeg_compression(img, int(severity))
            save_image(save_path, out)
        else:
            raise ValueError(f"Unsupported corruption: {corruption_name}")
    print(f"[DONE] Saved to: {target_root}")

def generate_all_default(input_root: Path, output_root: Path, seed: int = 42):
    corruption_plan = {
        "brightness": [0.9, 0.8, 1.1, 1.2],
        "blur": [3, 5, 7],
        "noise": [0.02, 0.05, 0.08],
        "jpeg": [90, 70, 50]
    }
    for corruption_name, severities in corruption_plan.items():
        for severity in severities:
            generate_single_corruption_dataset(
                input_root=input_root,
                output_root=output_root,
                corruption_name=corruption_name,
                severity=severity,
                seed=seed
            )

def parse_args():
    parser = argparse.ArgumentParser(description="Generate corrupted image datasets with OpenCV.")
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Path to clean test dataset root (ImageFolder-style)."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Directory to save corrupted datasets."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "brightness", "blur", "noise", "jpeg"],
        help="Which corruption to generate."
    )
    parser.add_argument(
        "--severity",
        type=str,
        default=None,
        help=(
            "Single severity for chosen mode. "
            "Examples: brightness->0.9, blur->5, noise->0.02, jpeg->70"
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    ensure_dir(output_root)
    if args.mode == "all":
        generate_all_default(input_root, output_root, seed=args.seed)
    else:
        if args.severity is None:
            raise ValueError(f"--severity is required when mode is '{args.mode}'")
        if args.mode == "brightness":
            sev = float(args.severity)
        elif args.mode == "blur":
            sev = int(args.severity)
        elif args.mode == "noise":
            sev = float(args.severity)
        elif args.mode == "jpeg":
            sev = int(args.severity)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
        generate_single_corruption_dataset(
            input_root=input_root,
            output_root=output_root,
            corruption_name=args.mode,
            severity=sev,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
