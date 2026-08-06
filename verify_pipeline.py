"""Smoke-test the two-stage pipeline on a real page and write inspectable artifacts."""

import json
import sys

import cv2

import pipeline

image_path = sys.argv[1] if len(sys.argv) > 1 else "examples/I1ER9510006.jpg"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "verify_out"

import os

os.makedirs(out_dir, exist_ok=True)

image = cv2.imread(image_path)
print(f"loaded {image_path} shape={image.shape}")

# --- stage 1 -------------------------------------------------------------- #
page = pipeline.detect(image)
boxes = pipeline.lines_to_boxes(page.lines)
print(f"detected {len(boxes)} lines, deskew angle {page.angle:.3f}deg")

cv2.imwrite(f"{out_dir}/deskewed.png", page.image)
with open(f"{out_dir}/boxes.json", "w") as fh:
    json.dump(boxes, fh, indent=2)

overlay = page.image.copy()
for i, b in enumerate(boxes):
    cv2.rectangle(overlay, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (0, 0, 255), 3)
    cv2.putText(overlay, str(i), (b["x"] - 40, b["y"] + b["h"]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
cv2.imwrite(f"{out_dir}/overlay.png", overlay)

# --- reconciliation round-trip -------------------------------------------- #
relines = pipeline.boxes_to_lines(boxes, page.lines)
reused = sum(1 for a, b in zip(relines, page.lines) if a is b)
print(f"round-trip: {len(relines)} lines, {reused} reused original contours (want {len(boxes)})")

edited = [dict(b) for b in boxes]
edited[0]["h"] += 10  # simulate a user nudging one box
edited_lines = pipeline.boxes_to_lines(edited, page.lines)
originals = {id(line) for line in page.lines}
reused_after = sum(1 for line in edited_lines if id(line) in originals)
print(f"after editing 1 box: {reused_after}/{len(edited)} contours preserved")

# --- stage 2 -------------------------------------------------------------- #
print(f"available OCR models: {pipeline.available_ocr_models()}")
texts = pipeline.ocr(page.image, relines)
print(f"OCR returned {len(texts)} lines\n")
for i, t in enumerate(texts):
    print(f"{i:>3}| {t}")

with open(f"{out_dir}/transcription.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(texts))
print(f"\nartifacts written to {out_dir}/")
