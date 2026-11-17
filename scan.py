"""Automatic Document Scanner - command-line entry point.

Examples:
    python scan.py --input samples/page.jpg
    python scan.py --input photos/ --output scans/ --pdf scans/all_pages.pdf
    python scan.py --input samples/page.jpg --debug      # also saves each pipeline stage
"""
import argparse
from pathlib import Path

import cv2
from PIL import Image

from scanner.utils import scan_document

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(path):
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    return [p]


def main():
    ap = argparse.ArgumentParser(description="Turn angled phone photos of documents into flat scans.")
    ap.add_argument("--input", required=True, help="an image file or a folder of images")
    ap.add_argument("--output", default="output", help="folder to write scans into")
    ap.add_argument("--color", action="store_true", help="keep color instead of black-and-white")
    ap.add_argument("--debug", action="store_true", help="also save edges/outline/warped stages")
    ap.add_argument("--pdf", help="combine all scans into one PDF at this path")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_pages = []

    images = collect_images(args.input)
    ok = 0
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[skip] could not read {path}")
            continue
        result = scan_document(img, color=args.color)
        if result is None:
            print(f"[fail] no page found in {path.name}")
            continue

        stem = path.stem
        cv2.imwrite(str(out_dir / f"{stem}_scan.png"), result["final"])
        if args.debug:
            cv2.imwrite(str(out_dir / f"{stem}_1_edges.png"), result["edges"])
            cv2.imwrite(str(out_dir / f"{stem}_2_outline.png"), result["outline"])
            cv2.imwrite(str(out_dir / f"{stem}_3_warped.png"), result["warped"])
        if args.pdf:
            final = result["final"]
            rgb = final if final.ndim == 2 else cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
            pdf_pages.append(Image.fromarray(rgb).convert("RGB"))
        ok += 1
        print(f"[ok]   {path.name} -> {out_dir / (stem + '_scan.png')}")

    if args.pdf and pdf_pages:
        Path(args.pdf).parent.mkdir(parents=True, exist_ok=True)
        pdf_pages[0].save(args.pdf, save_all=True, append_images=pdf_pages[1:])
        print(f"[pdf]  {len(pdf_pages)} page(s) -> {args.pdf}")

    print(f"\nScanned {ok}/{len(images)} image(s).")


if __name__ == "__main__":
    main()
