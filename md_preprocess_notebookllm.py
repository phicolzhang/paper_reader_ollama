#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess Markdown files by inserting images after figure captions.

Usage:
  python3 md_preprocess.py Nemotron3_ch.md

This script:
  1. Backs up the input file (e.g., Nemotron3_ch.md -> Nemotron3_ch.md.bak)
  2. Finds lines starting with "**图 $number"
  3. Inserts image references below those lines (e.g., ![图 1](Nemotron3/图1.png))
  4. Images are expected in a folder named after the markdown file (without "_ch.md")
     For example: Nemotron3_ch.md -> images in Nemotron3/ folder
"""

import argparse
import re
import shutil
from pathlib import Path


def extract_base_name(md_path: Path) -> str:
    """
    Extract base name from markdown filename.
    Example: Nemotron3_ch.md -> Nemotron3
    """
    stem = md_path.stem  # "Nemotron3_ch"
    if stem.endswith('_ch'):
        return stem[:-3]  # Remove "_ch"
    return stem


def find_figure_lines(lines):
    """
    Find all lines starting with "**图 $number" and return their indices and figure numbers.
    Returns: list of (line_index, figure_number)
    """
    pattern = re.compile(r'^\*\*图\s+(\d+)')
    matches = []
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            fig_num = int(m.group(1))
            matches.append((i, fig_num))
    return matches


def insert_images(md_path: Path, dry_run: bool = False):
    """
    Insert image references after figure caption lines.
    """
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    # Extract base name and determine image folder
    base_name = extract_base_name(md_path)
    img_folder = md_path.parent / base_name
    if not img_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {img_folder}")

    # Backup original file
    backup_path = md_path.with_suffix(md_path.suffix + '.bak')
    if not dry_run:
        shutil.copy2(md_path, backup_path)
        print(f"✅ Backed up to: {backup_path}")

    # Read markdown content
    lines = md_path.read_text(encoding='utf-8').splitlines(keepends=True)

    # Find all figure caption lines
    figure_matches = find_figure_lines(lines)
    if not figure_matches:
        print("⚠️  No figure captions found (lines starting with '**图 $number')")
        return

    print(f"Found {len(figure_matches)} figure caption(s)")

    # Insert images (process in reverse to preserve line indices)
    insertions = []
    for line_idx, fig_num in reversed(figure_matches):
        img_filename = f"图{fig_num}.png"
        img_path = img_folder / img_filename

        if not img_path.exists():
            print(f"⚠️  Image not found: {img_path} (skipping Figure {fig_num})")
            continue

        # Create image markdown syntax
        # Use relative path from markdown file to image folder
        rel_img_path = f"{base_name}/{img_filename}"
        img_markdown = f"![图 {fig_num}]({rel_img_path})\n"

        # Insert after the caption line
        insertions.append((line_idx + 1, img_markdown))
        print(f"  → Will insert {img_filename} after line {line_idx + 1} (Figure {fig_num})")

    if not insertions:
        print("⚠️  No images to insert (all images missing)")
        return

    if dry_run:
        print("\n[DRY RUN] Would insert images at the following positions:")
        for line_idx, img_md in insertions:
            print(f"  Line {line_idx}: {img_md.strip()}")
        return

    # Insert images (in reverse order to preserve indices)
    for line_idx, img_markdown in insertions:
        lines.insert(line_idx, img_markdown)

    # Write back
    md_path.write_text(''.join(lines), encoding='utf-8')
    print(f"\n✅ Inserted {len(insertions)} image(s) into: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Insert images into Markdown files after figure captions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 md_preprocess.py Nemotron3_ch.md
  python3 md_preprocess.py paper_ch.md --dry-run
        """
    )
    parser.add_argument(
        'input',
        type=Path,
        help='Input Markdown file (e.g., Nemotron3_ch.md)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without modifying the file'
    )
    args = parser.parse_args()

    try:
        insert_images(args.input, dry_run=args.dry_run)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())

