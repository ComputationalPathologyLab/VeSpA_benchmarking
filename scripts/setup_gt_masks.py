#!/usr/bin/env python
"""
Ground truth masks status checker.

Shows the current organization of ground truth masks across categories.

Usage:
    python scripts/setup_gt_masks.py
    python scripts/setup_gt_masks.py --status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class GTManager:
    """Manages ground truth mask organization across observer and derived categories."""

    def __init__(self, gt_base_dir: Path):
        """
        Initialize GTManager.

        Args:
            gt_base_dir: Base directory for all ground truth masks
        """
        self.base_dir = Path(gt_base_dir)
        self.cat1_dir = self.base_dir / "category_1_observers"
        self.cat2_dir = self.base_dir / "category_2_intersection"
        self.cat3_dir = self.base_dir / "category_3_union"

        # Ensure directories exist
        for d in [self.cat1_dir, self.cat2_dir, self.cat3_dir]:
            d.mkdir(parents=True, exist_ok=True)


def show_status(gt_base_dir: Path) -> None:
    """Show status of GT organization."""
    manager = GTManager(gt_base_dir)

    print("\n" + "=" * 70)
    print("GROUND TRUTH MASKS STATUS")
    print("=" * 70)

    # Category 1: Observers
    print("\n CATEGORY 1 - Observer Masks:")
    cat1_masks = list(manager.cat1_dir.glob("*.png"))
    if cat1_masks:
        print(f"   Found {len(cat1_masks)} masks:")
        for mask in sorted(cat1_masks):
            print(f"   - {mask.name}")
    else:
        print("   No observer masks found")

    # Category 2: Intersection
    print("\n CATEGORY 2 - Intersection Masks:")
    cat2_masks = list(manager.cat2_dir.glob("*.png"))
    if cat2_masks:
        print(f"   Found {len(cat2_masks)} masks:")
        for mask in sorted(cat2_masks):
            print(f"   - {mask.name}")
    else:
        print("   No intersection masks found")

    # Category 3: Union
    print("\nCATEGORY 3 - Union Masks:")
    cat3_masks = list(manager.cat3_dir.glob("*.png"))
    if cat3_masks:
        print(f"   Found {len(cat3_masks)} masks:")
        for mask in sorted(cat3_masks):
            print(f"   - {mask.name}")
    else:
        print("   No union masks found")

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check ground truth mask organization status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current GT mask status
  %(prog)s
  %(prog)s --status
        """,
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status of GT mask organization (default action)",
    )

    args = parser.parse_args()

    gt_base_dir = REPO_ROOT / "datasets" / "groundthruth"

    # Show status (default action)
    show_status(gt_base_dir)


if __name__ == "__main__":
    main()
