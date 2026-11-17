"""Generate a synthetic 'phone photo' of a page at an angle, for testing the
scanner before you have your own photos. Writes samples/synthetic_page.jpg."""
from pathlib import Path

import cv2
import numpy as np

page = np.full((1100, 850, 3), 245, np.uint8)
cv2.putText(page, "CSE381 Document Scanner", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3)
for i, y in enumerate(range(220, 1000, 55)):
    cv2.putText(page, f"Line {i + 1}: the quick brown fox jumps over the lazy dog",
                (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)

# Background: dark wooden-ish table with noise
bg = np.full((1500, 2000, 3), (40, 60, 90), np.uint8)
noise = np.random.default_rng(0).integers(0, 40, bg.shape, dtype=np.uint8)
bg = cv2.add(bg, noise)

src = np.float32([[0, 0], [849, 0], [849, 1099], [0, 1099]])
dst = np.float32([[620, 180], [1500, 300], [1380, 1350], [420, 1180]])  # skewed page
H = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(page, H, (bg.shape[1], bg.shape[0]))
mask = cv2.warpPerspective(np.full(page.shape[:2], 255, np.uint8), H, (bg.shape[1], bg.shape[0]))
bg[mask > 0] = warped[mask > 0]

# Uneven lighting gradient, like a real photo
grad = np.tile(np.linspace(0.75, 1.0, bg.shape[1]), (bg.shape[0], 1))[..., None]
photo = (bg * grad).astype(np.uint8)

Path("samples").mkdir(exist_ok=True)
cv2.imwrite("samples/synthetic_page.jpg", photo)
print("wrote samples/synthetic_page.jpg")
