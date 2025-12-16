#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract embedded images from a PDF into a folder named after the PDF.

Usage:
  python3 extract_figure.py nemotron3.pdf

Output:
  Creates ./nemotron3/ and saves images as 图1.png / 图2.jpg / ...

Notes:
  - Requires PyMuPDF: pip install PyMuPDF
  - Default strategy is caption-driven: locate "Figure 1/2/..." captions to determine
    figure count and approximate positions, then crop-render the visual region near
    each caption as a single exported image (merging subplots).
  - If captions are missing, it can fallback to rect-driven extraction or page rendering.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _require_pymupdf():
    try:
        import fitz  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: PyMuPDF.\n"
            "Install with: pip install PyMuPDF\n"
            f"Original error: {e}"
        )
    return fitz


def _cluster_rects(rects, merge_gap: float):
    """
    Greedy clustering: merge rects if they intersect after expanding by merge_gap.
    Works well for "subplots extracted as multiple images" -> union into one figure box.
    """
    clusters = []
    for r in sorted(rects, key=lambda x: (x.y0, x.x0)):
        merged = False
        for idx, c in enumerate(clusters):
            c_expand = c + (-merge_gap, -merge_gap, merge_gap, merge_gap)
            r_expand = r + (-merge_gap, -merge_gap, merge_gap, merge_gap)
            if c_expand.intersects(r_expand) or c_expand.contains(r_expand) or r_expand.contains(c_expand):
                clusters[idx] = c | r
                merged = True
                break
        if not merged:
            clusters.append(r)

    # One more pass in case clusters became mergeable after unions
    changed = True
    while changed:
        changed = False
        new_clusters = []
        for c in clusters:
            merged_into = False
            for j, nc in enumerate(new_clusters):
                nc_expand = nc + (-merge_gap, -merge_gap, merge_gap, merge_gap)
                c_expand = c + (-merge_gap, -merge_gap, merge_gap, merge_gap)
                if nc_expand.intersects(c_expand) or nc_expand.contains(c_expand) or c_expand.contains(nc_expand):
                    new_clusters[j] = nc | c
                    merged_into = True
                    changed = True
                    break
            if not merged_into:
                new_clusters.append(c)
        clusters = new_clusters

    return clusters


def _norm_block_text(block) -> str:
    parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            t = span.get("text", "")
            if t:
                parts.append(t)
        parts.append("\n")
    return "".join(parts).strip()


_CAPTION_RE = re.compile(
    r"^(?P<label>(Figure|Fig\.?|FIG\.?|图))\s*(?P<num>\d{1,4})\b\s*(?P<sep>[\|\.:：-])?",
    re.IGNORECASE,
)


def _find_caption_blocks(page, header_y: float, footer_y: float):
    """
    Return list of (num:int|None, bbox:fitz.Rect, text:str, lead_font_size:float|None) for caption-like text blocks.
    Heuristics:
      - must match caption regex at the start of a text block (avoids 'see Figure 2' refs)
      - skip header/footer strips
    """
    fitz = _require_pymupdf()
    d = page.get_text("dict")
    out = []
    for b in d.get("blocks", []):
        if b.get("type") != 0 or not b.get("bbox"):
            continue
        bbox = fitz.Rect(b["bbox"])
        if bbox.y1 <= header_y or bbox.y0 >= footer_y:
            continue
        text = _norm_block_text(b)
        if not text:
            continue
        m = _CAPTION_RE.match(text.lstrip())
        if not m:
            continue
        # Additional guard: captions usually start the block; don't accept if it starts with '('
        if text.lstrip().startswith("("):
            continue
        num = None
        try:
            num = int(m.group("num"))
        except Exception:
            num = None

        # capture the leading span font size for potential font-based heuristics
        lead_size = None
        try:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    t = (span.get("text") or "").strip()
                    if t:
                        lead_size = float(span.get("size")) if span.get("size") is not None else None
                        raise StopIteration
        except StopIteration:
            pass
        except Exception:
            lead_size = None
        out.append((num, bbox, text, lead_size))
    # order top-to-bottom
    out.sort(key=lambda x: (x[1].y0, x[1].x0))
    return out


def _collect_visual_rects(page, header_y: float, footer_y: float, min_area_ratio: float):
    """
    Collect rectangles of visual elements (embedded images + vector drawings).
    For vector plots, drawings rects often cover the plot region.
    """
    fitz = _require_pymupdf()
    page_rect = page.rect
    page_area = float(page_rect.width * page_rect.height)

    rects = []
    # Embedded images
    for img in page.get_images(full=True):
        xref = int(img[0])
        try:
            rects.extend(page.get_image_rects(xref))
        except Exception:
            continue

    # Image blocks (some PDFs expose them here)
    try:
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            if b.get("type") == 1 and b.get("bbox"):
                rects.append(fitz.Rect(b["bbox"]))
    except Exception:
        pass

    # Vector drawings (charts often are pure vector)
    try:
        for dr in page.get_drawings():
            r = dr.get("rect")
            if r:
                rects.append(fitz.Rect(r))
    except Exception:
        pass

    out = []
    for r in rects:
        if r.y1 <= header_y or r.y0 >= footer_y:
            continue
        # filter ultra-tiny artifacts (icons, bullets)
        if float(r.get_area()) < page_area * min_area_ratio:
            continue
        out.append(r)
    return out


def extract_figures_by_captions(
    pdf_path: Path,
    out_dir: Path,
    name_prefix: str = "图",
    dpi: int = 300,
    header_cut: float = 0.12,
    footer_cut: float = 0.05,
    merge_gap: float = 10.0,
    min_visual_area_ratio: float = 0.00005,
    min_crop_area_ratio: float = 0.002,
    pad: float = 4.0,
    include_all_subfigures: bool = True,
    subfigure_area_keep_ratio: float = 0.08,
) -> int:
    """
    Caption-driven extraction:
      1) locate 'Figure N ...' caption blocks
      2) collect visual rects on the page (images + drawings)
      3) associate a visual cluster with each caption (usually just above the caption)
      4) crop-render that cluster as a single image
    """
    fitz = _require_pymupdf()
    doc = fitz.open(str(pdf_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0

    used_names: set[str] = set()
    saved = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_rect = page.rect
        page_w, page_h = page_rect.width, page_rect.height
        page_area = float(page_w * page_h)

        header_y = page_h * header_cut
        footer_y = page_h * (1.0 - footer_cut)
        top_y = page_rect.y0
        bottom_y = page_rect.y1

        captions = _find_caption_blocks(page, header_y, footer_y)
        if not captions:
            continue

        visuals = _collect_visual_rects(page, header_y, footer_y, min_area_ratio=min_visual_area_ratio)
        if not visuals:
            # still allow a geometric fallback crop around caption (vector-only without detectable drawings is rare)
            visuals = []

        # For multi-caption pages, use neighbors to limit the search window
        for idx, (num, cap_bbox, cap_text, lead_size) in enumerate(captions):
            # IMPORTANT: do not force prev_y to header_y; top-of-page figures often start above "header-cut".
            prev_y = top_y if idx == 0 else min(max(captions[idx - 1][1].y1 + 2.0, top_y), bottom_y)
            next_y = bottom_y if idx == len(captions) - 1 else max(min(captions[idx + 1][1].y0 - 2.0, bottom_y), top_y)

            # search window ABOVE the caption (primary rule)
            above_window = fitz.Rect(page_rect.x0, prev_y, page_rect.x1, cap_bbox.y0)
            if above_window.y1 <= above_window.y0:
                continue

            # Prefer visuals above caption
            above = []
            for r in visuals:
                # take intersection with the above-window; this avoids a big drawing-rect spanning
                # multiple figures (above + below the caption) from dominating the crop.
                if not r.intersects(above_window):
                    continue
                rr = fitz.Rect(r)
                rr.x0 = max(rr.x0, above_window.x0)
                rr.x1 = min(rr.x1, above_window.x1)
                rr.y0 = max(rr.y0, above_window.y0)
                rr.y1 = min(rr.y1, above_window.y1)
                if rr.y1 > rr.y0 and rr.x1 > rr.x0:
                    above.append(rr)
            above_clusters = _cluster_rects(above, merge_gap=merge_gap) if above else []

            chosen = None
            if above_clusters:
                # For multi-panel figures (a/b/c...), keep ALL significant clusters above the caption
                # so the exported image contains the full figure, not just one panel.
                if include_all_subfigures and len(above_clusters) > 1:
                    biggest = max(above_clusters, key=lambda r: float(r.get_area()))
                    biggest_area = float(biggest.get_area()) or 1.0
                    kept = [r for r in above_clusters if float(r.get_area()) >= biggest_area * subfigure_area_keep_ratio]
                    if kept:
                        u = kept[0]
                        for r in kept[1:]:
                            u = u | r
                        chosen = u
                    else:
                        chosen = biggest
                else:
                    # pick cluster closest to caption, then by area
                    def score(r):
                        gap = max(0.0, cap_bbox.y0 - r.y1)
                        area = float(r.get_area())
                        return (gap, -area)

                    chosen = sorted(above_clusters, key=score)[0]

            # Geometric fallback: crop the band above caption within the window (may include text, but better than missing)
            if chosen is None:
                # Main rule: crop ABOVE the caption (as requested)
                chosen = fitz.Rect(above_window)

                # If the caption is extremely close to the top, it's likely "caption-on-top";
                # in that special case, use a BELOW search window to avoid exporting a near-blank crop.
                if cap_bbox.y0 < page_h * 0.18 and chosen.height < page_h * 0.10:
                    below_window = fitz.Rect(page_rect.x0, cap_bbox.y1, page_rect.x1, next_y)
                    below = []
                    for r in visuals:
                        if not r.intersects(below_window):
                            continue
                        rr = fitz.Rect(r)
                        rr.x0 = max(rr.x0, below_window.x0)
                        rr.x1 = min(rr.x1, below_window.x1)
                        rr.y0 = max(rr.y0, below_window.y0)
                        rr.y1 = min(rr.y1, below_window.y1)
                        if rr.y1 > rr.y0 and rr.x1 > rr.x0:
                            below.append(rr)
                    below_clusters = _cluster_rects(below, merge_gap=merge_gap) if below else []
                    if below_clusters:
                        def score2(r):
                            gap = max(0.0, r.y0 - cap_bbox.y1)
                            area = float(r.get_area())
                            return (gap, -area)

                        chosen = sorted(below_clusters, key=score2)[0]
                    else:
                        chosen = fitz.Rect(page_rect.x0, cap_bbox.y1, page_rect.x1, min(next_y, cap_bbox.y1 + page_h * 0.35))

            if chosen is None:
                continue

            # clamp & pad & enforce body area
            bbox = fitz.Rect(chosen)
            bbox.x0 = max(page_rect.x0, bbox.x0 - pad)
            bbox.x1 = min(page_rect.x1, bbox.x1 + pad)
            # Do NOT clamp to header_y here (top-of-page figures would get chopped).
            bbox.y0 = max(page_rect.y0, bbox.y0 - pad)
            bbox.y1 = min(page_rect.y1, bbox.y1 + pad)

            if bbox.y1 <= bbox.y0 or bbox.x1 <= bbox.x0:
                continue
            if float(bbox.get_area()) < page_area * min_crop_area_ratio:
                # too small: likely an icon
                continue

            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=bbox, alpha=False)
            # prefer using caption number if available
            if num is not None:
                filename = f"{name_prefix}{num}.png"
                if filename in used_names:
                    filename = f"{name_prefix}{saved + 1}.png"
            else:
                filename = f"{name_prefix}{saved + 1}.png"

            used_names.add(filename)
            saved += 1
            (out_dir / filename).write_bytes(pix.tobytes("png"))

    return saved


def extract_figures_by_rects(
    pdf_path: Path,
    out_dir: Path,
    name_prefix: str = "图",
    dpi: int = 300,
    header_cut: float = 0.12,
    footer_cut: float = 0.05,
    merge_gap: float = 6.0,
    min_area_ratio: float = 0.002,
) -> int:
    """
    Extract figures by locating image rectangles on each page, clustering nearby rects,
    and rendering the union bbox. This avoids "one figure -> many images" (subplots),
    and allows skipping header logo strips.
    """
    fitz = _require_pymupdf()
    doc = fitz.open(str(pdf_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    scale = dpi / 72.0

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_rect = page.rect
        page_w, page_h = page_rect.width, page_rect.height
        page_area = float(page_w * page_h)

        header_y = page_h * header_cut
        footer_y = page_h * (1.0 - footer_cut)

        rects = []
        for img in page.get_images(full=True):
            xref = int(img[0])
            try:
                rects.extend(page.get_image_rects(xref))
            except Exception:
                continue

        # Some PDFs expose images as blocks (type=1) even when get_images is sparse.
        try:
            d = page.get_text("dict")
            for b in d.get("blocks", []):
                if b.get("type") == 1 and b.get("bbox"):
                    rects.append(fitz.Rect(b["bbox"]))
        except Exception:
            pass

        if not rects:
            continue

        clusters = _cluster_rects(rects, merge_gap=merge_gap)

        # Sort top-to-bottom, left-to-right for stable numbering
        clusters = sorted(clusters, key=lambda r: (r.y0, r.x0))

        for bbox in clusters:
            # Skip obvious header/footer decorations (logos, page header images)
            if bbox.y1 <= header_y:
                continue
            if bbox.y0 >= footer_y:
                continue

            # Skip tiny artifacts (icons, bullets)
            if float(bbox.get_area()) < page_area * min_area_ratio:
                # But allow large figures even if near header; tiny near header are almost always logos
                continue

            # Clamp to body area to remove header/footer content even if bbox touches them
            bbox = fitz.Rect(bbox)
            bbox.y0 = max(bbox.y0, header_y)
            bbox.y1 = min(bbox.y1, footer_y)
            if bbox.y1 <= bbox.y0 or bbox.x1 <= bbox.x0:
                continue

            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=bbox, alpha=False)
            saved += 1
            out_path = out_dir / f"{name_prefix}{saved}.png"
            pix.save(str(out_path))

    return saved


def render_pages_fallback(
    pdf_path: Path,
    out_dir: Path,
    name_prefix: str = "图",
    dpi: int = 300,
    header_cut: float = 0.12,
    footer_cut: float = 0.05,
) -> int:
    """Fallback: render each page to a PNG, cropping header/footer strips."""
    fitz = _require_pymupdf()
    doc = fitz.open(str(pdf_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    scale = dpi / 72.0
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_rect = page.rect
        page_h = page_rect.height
        header_y = page_h * header_cut
        footer_y = page_h * (1.0 - footer_cut)
        clip = fitz.Rect(page_rect)
        clip.y0 = max(clip.y0, header_y)
        clip.y1 = min(clip.y1, footer_y)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
        saved += 1
        out_path = out_dir / f"{name_prefix}{saved}.png"
        pix.save(str(out_path))
    return saved


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract images from a PDF into a same-name folder.")
    p.add_argument("pdf", help="Input PDF file path")
    p.add_argument(
        "--outdir",
        "-o",
        default=None,
        help="Output directory (default: <pdf_stem>/ next to the PDF)",
    )
    p.add_argument(
        "--prefix",
        default="图",
        help="Filename prefix (default: 图). Output will be 图1.png, 图2.jpg, ...",
    )
    p.add_argument(
        "--mode",
        choices=["caption", "rect", "pages"],
        default="caption",
        help="Extraction mode: caption (default), rect, or pages.",
    )
    p.add_argument("--dpi", type=int, default=300, help="Render DPI when cropping figures/pages (default: 300)")
    p.add_argument(
        "--header-cut",
        type=float,
        default=0.12,
        help="Skip/crop the top header strip by ratio of page height (default: 0.12)",
    )
    p.add_argument(
        "--footer-cut",
        type=float,
        default=0.05,
        help="Skip/crop the bottom footer strip by ratio of page height (default: 0.05)",
    )
    p.add_argument(
        "--merge-gap",
        type=float,
        default=6.0,
        help="Merge nearby image rects within this gap (PDF points) to avoid split subplots (default: 6)",
    )
    p.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.002,
        help="Ignore tiny images smaller than this ratio of page area (default: 0.002)",
    )
    p.add_argument(
        "--min-visual-area-ratio",
        type=float,
        default=0.00005,
        help="(caption mode) Filter tiny visual rects (default: 0.00005)",
    )
    p.add_argument(
        "--no-subfigures",
        action="store_true",
        help="(caption mode) Do not merge multiple panels (a/b/...) into a single figure crop.",
    )
    p.add_argument(
        "--subfigure-keep-ratio",
        type=float,
        default=0.08,
        help="(caption mode) Keep subfigure clusters whose area >= this ratio of the largest cluster (default: 0.08)",
    )
    p.add_argument(
        "--fallback-pages",
        action="store_true",
        help="If no figures were found, render each page (cropping header/footer) as PNGs.",
    )
    args = p.parse_args(argv)

    pdf_path = Path(args.pdf).expanduser()
    if not pdf_path.exists() or not pdf_path.is_file():
        print(f"❌ PDF not found: {pdf_path}", file=sys.stderr)
        return 2
    if pdf_path.suffix.lower() != ".pdf":
        print(f"❌ Not a PDF file: {pdf_path}", file=sys.stderr)
        return 2

    if args.outdir:
        out_dir = Path(args.outdir).expanduser()
    else:
        # same directory as pdf, folder named by stem
        out_dir = pdf_path.with_suffix("")

    try:
        if args.mode == "caption":
            count = extract_figures_by_captions(
                pdf_path,
                out_dir,
                name_prefix=args.prefix,
                dpi=args.dpi,
                header_cut=args.header_cut,
                footer_cut=args.footer_cut,
                merge_gap=args.merge_gap,
                min_visual_area_ratio=args.min_visual_area_ratio,
                min_crop_area_ratio=args.min_area_ratio,
                include_all_subfigures=(not args.no_subfigures),
                subfigure_area_keep_ratio=args.subfigure_keep_ratio,
            )
            if count == 0:
                # fallback to rect mode if captions are missing / not detected
                count = extract_figures_by_rects(
                    pdf_path,
                    out_dir,
                    name_prefix=args.prefix,
                    dpi=args.dpi,
                    header_cut=args.header_cut,
                    footer_cut=args.footer_cut,
                    merge_gap=args.merge_gap,
                    min_area_ratio=args.min_area_ratio,
                )
            if count == 0 and args.fallback_pages:
                count = render_pages_fallback(
                    pdf_path,
                    out_dir,
                    name_prefix=args.prefix,
                    dpi=args.dpi,
                    header_cut=args.header_cut,
                    footer_cut=args.footer_cut,
                )
        elif args.mode == "rect":
            count = extract_figures_by_rects(
                pdf_path,
                out_dir,
                name_prefix=args.prefix,
                dpi=args.dpi,
                header_cut=args.header_cut,
                footer_cut=args.footer_cut,
                merge_gap=args.merge_gap,
                min_area_ratio=args.min_area_ratio,
            )
            if count == 0 and args.fallback_pages:
                count = render_pages_fallback(
                    pdf_path,
                    out_dir,
                    name_prefix=args.prefix,
                    dpi=args.dpi,
                    header_cut=args.header_cut,
                    footer_cut=args.footer_cut,
                )
        else:
            count = render_pages_fallback(
                pdf_path,
                out_dir,
                name_prefix=args.prefix,
                dpi=args.dpi,
                header_cut=args.header_cut,
                footer_cut=args.footer_cut,
            )
    except RuntimeError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    print(f"✅ Extracted {count} image(s) to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


