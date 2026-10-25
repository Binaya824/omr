import cv2
import sys
import os
import json
import statistics
from collections import Counter
import numpy as np


def find_largest_rectangle(img, area_ratio=0.3, target_size=1000):
    """Bounding box of the sheet. Usually one contour already encloses the
    whole sheet, but a photographed sheet (vs. a clean render) can show up as
    several separate same-size block borders instead of one outer border
    (and Canny's edges often have pixel-level gaps that break a single big
    border into fragments) -- dilating closes those gaps, and unioning every
    contour close in size to the largest covers the multi-block case too.

    A high-resolution phone photo carries fine JPEG/moire noise dense enough
    that its raw edge map is one giant connected blob even before dilating
    (the noise bridges the rectangle borders to everything around them, so
    there's no clean contour to find at all). Downscaling to ~target_size
    first averages that per-pixel noise away while the bold rectangle
    borders survive, which is also why the right amount of dilation is a
    fixed amount of *downscaled* pixels, not a fraction of the original
    resolution -- so dilation is tuned once at this normalized scale rather
    than relative to whatever resolution the photo happened to arrive at."""
    h, w = img.shape[:2]
    scale = min(1.0, target_size / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else img

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found")

    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)
    boxes = [cv2.boundingRect(c) for c, a in zip(contours, areas) if a >= area_ratio * max_area]

    x0 = min(bx for bx, by, bw, bh in boxes)
    y0 = min(by for bx, by, bw, bh in boxes)
    x1 = max(bx + bw for bx, by, bw, bh in boxes)
    y1 = max(by + bh for bx, by, bw, bh in boxes)
    return (round(x0 / scale), round(y0 / scale), round((x1 - x0) / scale), round((y1 - y0) / scale))


def _normalized_roi(img, box, target_size=1000):
    """Crop to box and downscale so the longer side is ~target_size. The
    area/radius heuristics below are tuned in absolute pixels at that scale;
    without this, the same sheet photographed at a different resolution
    needs different thresholds (a real bubble can be 3000px^2 in a phone
    photo and 300px^2 in a small render) and fine photo noise that would
    normally average away survives as spurious same-size "circles"."""
    x, y, w, h = box
    roi = img[y:y + h, x:x + w]
    scale = min(1.0, target_size / max(h, w))
    small = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else roi
    return small, scale, x, y


def find_circles(img, box, min_area=15, max_area=400, min_circularity=0.7, dedup_dist=4):
    roi, scale, x, y = _normalized_roi(img, box)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 15, 5)
    contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity < min_circularity:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r <= 2:
            continue
        circles.append((x + cx / scale, y + cy / scale, r / scale))

    # adjacent contours (inner/outer ring of the same drawn circle) collapse to one
    # (threshold scaled back to original-image pixels, same as the circles above)
    circles.sort(key=lambda c: -c[2])
    kept = []
    dedup_dist_orig = dedup_dist / scale
    for cx, cy, r in circles:
        if not any((cx - kx) ** 2 + (cy - ky) ** 2 < dedup_dist_orig ** 2 for kx, ky, _ in kept):
            kept.append((cx, cy, r))
    return kept


def find_circles_hough(img, box):
    """Fallback for noisy/photographed sheets where contour tracing can't find
    clean closed loops (moire patterns, JPEG artifacts, glare). Sweeps a range
    of radii and sensitivities and keeps whichever HoughCircles result builds
    into the most complete grid, since the true bubble size isn't known
    upfront for an arbitrary photo."""
    roi, scale, x, y = _normalized_roi(img, box)
    h = roi.shape[0]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cv2.medianBlur(gray, 5), (5, 5), 0)

    best_circles, best_score = [], 0
    for min_r in range(3, max(5, h // 12), 2):
        max_r = int(min_r * 1.8) + 2
        for p2 in (12, 14, 16, 20):
            found = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=min_r * 1.3,
                                      param1=40, param2=p2, minRadius=min_r, maxRadius=max_r)
            if found is None:
                continue
            cands = [(x + float(cx) / scale, y + float(cy) / scale, float(r) / scale)
                     for cx, cy, r in found[0]]
            grid = build_grid(normalize_radius(filter_similar_size(cands)))
            score = sum(len(row) for col in grid for row in col)
            if score > best_score:
                best_score, best_circles = score, cands
    return best_circles


def filter_similar_size(circles, tolerance=0.2):
    if not circles:
        return circles
    median_r = statistics.median(r for _, _, r in circles)
    return [c for c in circles if abs(c[2] - median_r) <= tolerance * median_r]


def normalize_radius(circles):
    if not circles:
        return circles
    mode_r = Counter(round(r) for _, _, r in circles).most_common(1)[0][0]
    return [(cx, cy, mode_r) for cx, cy, _ in circles]


def cluster_1d(values, tolerance):
    clusters = []
    for v in sorted(values):
        if clusters and v - clusters[-1][-1] <= tolerance:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [statistics.mean(c) for c in clusters]


def build_grid(circles, options_per_row=4):
    """Reconstruct the full row/column grid from detected circles, so a bubble
    missed by detection (e.g. a filled-in answer) still gets its slot: every
    column's spacing and left edge are derived from the circles that WERE
    found, then applied uniformly to every row."""
    if len(circles) < options_per_row:
        return []

    radius = circles[0][2]  # already normalized to one common radius
    row_centers = cluster_1d([c[1] for c in circles], tolerance=radius * 0.8)
    col_centers = cluster_1d([c[0] for c in circles], tolerance=radius * 0.8)
    if len(col_centers) < options_per_row or len(row_centers) < 2:
        return []

    def spacing_of(centers):
        diffs = [round(b - a) for a, b in zip(centers, centers[1:])]
        return Counter(diffs).most_common(1)[0][0]  # most common uniform distance

    def uniform_positions(centers, spacing):
        edge = min(centers) - radius  # first circle touches this edge
        return [edge + radius + k * spacing for k in range(len(centers))]

    col_centers = sorted(col_centers)
    col_spacing = spacing_of(col_centers)
    blocks = [[col_centers[0]]]
    for prev, cur in zip(col_centers, col_centers[1:]):
        if cur - prev > 1.5 * col_spacing:
            blocks.append([])
        blocks[-1].append(cur)

    row_centers = sorted(row_centers)
    ys = uniform_positions(row_centers, spacing_of(row_centers))

    columns = []
    for block in blocks:
        if len(block) != options_per_row:
            continue
        xs = uniform_positions(block, col_spacing)
        columns.append([[(cx, ry, radius) for cx in xs] for ry in ys])
    return columns


def fill_ratio(gray, cx, cy, r, sample_scale=0.85, dark_thresh=127):
    # sample slightly inside the ring so its own outline doesn't count as fill
    r2 = r * sample_scale
    y0, y1 = int(cy - r2), int(cy + r2) + 1
    x0, x1 = int(cx - r2), int(cx + r2) + 1
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
    mask = (xx - (cx - x0)) ** 2 + (yy - (cy - y0)) ** 2 <= r2 * r2
    return (patch[mask] < dark_thresh).mean()


def fill_threshold(ratios, default=0.9, min_gap=0.06):
    """The requested 90% cutoff assumes solid, low-noise ink. A compressed or
    photographed sheet can cap even a genuinely filled bubble's ratio well
    below that, so if nothing clears the default, fall back to splitting at
    the widest gap in the observed ratios -- still "mostly dark vs mostly
    not", just calibrated to this image instead of assumed. But an entirely
    unfilled sheet has no real split, only noise -- its widest gap is tiny,
    so require the gap to be a real separation before trusting it, otherwise
    keep the default (nothing on a blank sheet qualifies as filled)."""
    if not ratios or max(ratios) >= default:
        return default
    s = sorted(ratios)
    gaps = [(b - a, a, b) for a, b in zip(s, s[1:])]
    gap, lo, hi = max(gaps)
    return (lo + hi) / 2 if gap >= min_gap else default


def process(image_path, out_path="output.png", json_path=None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    x, y, w, h = find_largest_rectangle(img)

    # Contour tracing is precise on a clean render but unreliable on a noisy
    # photo (compression/moire breaks the ring shapes) -- and a noisy photo
    # doesn't just fail cleanly, it can produce a small but non-empty grid
    # that looks "done" and is actually wrong. So always compute both and
    # keep whichever reconstructs the more complete grid, rather than only
    # falling back to Hough when contour tracing returns nothing at all.
    contour_circles = normalize_radius(filter_similar_size(find_circles(img, (x, y, w, h))))
    contour_columns = build_grid(contour_circles)
    contour_score = sum(len(row) for col in contour_columns for row in col)

    hough_circles = normalize_radius(filter_similar_size(find_circles_hough(img, (x, y, w, h))))
    hough_columns = build_grid(hough_circles)
    hough_score = sum(len(row) for col in hough_columns for row in col)

    columns = contour_columns if contour_score >= hough_score else hough_columns
    columns.sort(key=lambda col: statistics.mean(c[0] for row in col for c in row))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_circles = [c for col in columns for row in col for c in row]
    threshold = fill_threshold([fill_ratio(gray, cx, cy, r) for cx, cy, r in all_circles])

    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    pad = 3
    row_number = 1
    filled_count = 0
    results = {}
    for i, column in enumerate(columns, start=1):
        flat = [c for row in column for c in row]
        gx0 = int(min(c[0] - c[2] for c in flat)) - pad
        gy0 = int(min(c[1] - c[2] for c in flat)) - pad
        gx1 = int(max(c[0] + c[2] for c in flat)) + pad
        gy1 = int(max(c[1] + c[2] for c in flat)) + pad
        # cv2.rectangle(img, (gx0, gy0), (gx1, gy1), (255, 0, 0), 2)
        # label = str(i)
        # (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # tx = (gx0 + gx1) // 2 - tw // 2
        # cv2.putText(img, label, (tx, gy0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # for row_a, row_b in zip(column, column[1:]):
        #     bottom = max(c[1] + c[2] for c in row_a)
        #     top = min(c[1] - c[2] for c in row_b)
        #     line_y = int((bottom + top) / 2)
        #     cv2.line(img, (gx0, line_y), (gx1, line_y), (255, 0, 0), 1)

        for row in column:
            row_y = int(statistics.mean(c[1] for c in row))
            # rlabel = str(row_number)
            # (rtw, rth), _ = cv2.getTextSize(rlabel, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            # cv2.putText(img, rlabel, (gx0 - rtw - 4, row_y + rth // 2),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            filled_index = 0
            for option_index, (cx, cy, r) in enumerate(row, start=1):
                filled = fill_ratio(gray, cx, cy, r) >= threshold
                filled_count += filled
                if filled and filled_index == 0:
                    filled_index = option_index  # first filled bubble in the row
                if filled:
                    cv2.circle(img, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
            results[row_number] = filled_index
            row_number += 1

    grid_circle_count = sum(len(row) for col in columns for row in col)
    cv2.imwrite(out_path, img)

    json_path = json_path or f"{os.path.splitext(out_path)[0]}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return (x, y, w, h), grid_circle_count, len(columns), filled_count


def demo():
    # synthetic OMR-like grid: 2 rows x 4 circles, evenly spaced, one circle
    # omitted entirely (build_grid must reconstruct it) and one filled solid
    # (fill detection must flag it).
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (180, 180), (255, 255, 255), -1)
    radius, spacing = 8, 24
    skip = (1, 2)  # row 1, col 2 deliberately missing
    filled = (0, 1)  # row 0, col 1 deliberately filled solid
    for row in range(2):
        for col in range(4):
            if (row, col) == skip:
                continue
            cx = 50 + col * spacing
            cy = 70 + row * spacing
            thickness = -1 if (row, col) == filled else 2
            cv2.circle(img, (cx, cy), radius, (0, 0, 0), thickness)
    cv2.imwrite("/tmp/_test_in.png", img)

    box, n, g, f = process("/tmp/_test_in.png", "/tmp/_test_out.png")
    expected_box = (20, 20, 160, 160)
    assert all(abs(a - b) <= 6 for a, b in zip(box, expected_box)), box  # dilation pads edges a few px
    assert g == 1, g  # one column of circles
    assert n == 8, n  # all 8 slots present, including the reconstructed one
    assert f == 1, f  # exactly the one solid-filled circle detected as filled

    with open("/tmp/_test_out.json") as fh:
        results = json.load(fh)
    assert results == {"1": 2, "2": 0}, results  # row 1 option 2 filled, row 2 empty
    print("demo ok:", box, n, "circle(s)", g, "column(s)", f, "filled", results)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(process(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output.png"))
    else:
        demo()
