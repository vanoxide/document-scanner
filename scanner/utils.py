"""Core image-processing steps for the document scanner.

Pipeline: resize -> grayscale + blur -> Canny edges -> find the page's 4 corners
-> homography (warpPerspective) -> adaptive threshold for a clean "scanned" look.
"""
import cv2
import numpy as np

WORK_HEIGHT = 800  # images are downscaled to this height for edge/contour detection


def resize_to_height(img, height=WORK_HEIGHT):
    """Downscale an image to a fixed height. Returns (resized, scale) where
    scale converts coordinates in the resized image back to the original."""
    h, w = img.shape[:2]
    scale = h / float(height)
    resized = cv2.resize(img, (int(w / scale), height), interpolation=cv2.INTER_AREA)
    return resized, scale


def detect_edges(img):
    """Grayscale -> Gaussian blur -> Canny -> dilate (closes small gaps in the page outline)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Automatic Canny thresholds based on the median intensity
    median = np.median(blur)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(blur, lower, upper)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def order_corners(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array([
        pts[np.argmin(s)],  # top-left: smallest x+y
        pts[np.argmin(d)],  # top-right: smallest y-x
        pts[np.argmax(s)],  # bottom-right: largest x+y
        pts[np.argmax(d)],  # bottom-left: largest y-x
    ], dtype=np.float32)


def find_page_corners(edges, min_area_ratio=0.15):
    """Find the 4 corners of the largest quadrilateral contour in an edge map.

    Falls back to the minimum-area rectangle around the largest contour if no
    clean 4-sided polygon is found. Returns None if nothing big enough exists.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    img_area = edges.shape[0] * edges.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        if cv2.contourArea(c) < min_area_ratio * img_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx)

    largest = contours[0]
    if cv2.contourArea(largest) < min_area_ratio * img_area:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(largest))
    return order_corners(box)


def warp_to_top_down(img, corners):
    """Compute the homography from the 4 page corners to a flat rectangle and warp."""
    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, H, (width, height))


def scan_look(warped, block_size=21, c=10):
    """Adaptive threshold to give a clean black-on-white 'scanned' look."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, c)


def scan_document(img, color=False):
    """Run the full pipeline on a BGR image.

    Returns a dict with the result and every intermediate stage (handy for the
    README/demo GIF), or None if no page was found.
    """
    small, scale = resize_to_height(img)
    edges = detect_edges(small)
    corners_small = find_page_corners(edges)
    if corners_small is None:
        return None

    corners = corners_small * scale  # map back to full resolution
    warped = warp_to_top_down(img, corners)
    final = warped if color else scan_look(warped)

    outline = small.copy()
    cv2.polylines(outline, [corners_small.astype(np.int32)], True, (0, 255, 0), 3)
    for x, y in corners_small.astype(int):
        cv2.circle(outline, (x, y), 8, (0, 0, 255), -1)

    return {"edges": edges, "outline": outline, "warped": warped,
            "final": final, "corners": corners}
