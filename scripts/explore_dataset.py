#!/usr/bin/env python3
"""
Exploration rapide du jeu de données Vesuvius Challenge - Surface Detection.

- Résume les ids présents dans les CSV train/test.
- Vérifie la présence des fichiers .tif images/labels et les écarts éventuels.
- Inspecte la métadonnée simple des .tif via la commande `file` (forme, bps).
- Écrit un rapport Markdown pour consigner les observations.

Usage (depuis la racine du projet):
    python3 scripts/explore_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

RE_SHAPE = re.compile(r'shape": \[([0-9 ,]+)\]')
RE_HEIGHT = re.compile(r"height=([0-9]+)")
RE_WIDTH = re.compile(r"width=([0-9]+)")
RE_BPS = re.compile(r"bps=([0-9]+)")


def read_csv_ids(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def list_ids_from_dir(path: Path) -> List[str]:
    return sorted(p.stem for p in path.glob("*.tif"))


def probe_tif_metadata(path: Path) -> Dict[str, Optional[str]]:
    """Interroge la commande `file` pour récupérer forme/dtype simples."""
    res = subprocess.run(["file", str(path)], capture_output=True, text=True, check=True)
    line = res.stdout.strip()
    meta: Dict[str, Optional[str]] = {"path": str(path.name), "raw": line}

    def extract_int(regex: re.Pattern[str]) -> Optional[int]:
        m = regex.search(line)
        return int(m.group(1)) if m else None

    meta["height"] = extract_int(RE_HEIGHT)
    meta["width"] = extract_int(RE_WIDTH)
    meta["bps"] = extract_int(RE_BPS)

    shape = None
    m = RE_SHAPE.search(line)
    if m:
        shape = tuple(int(x.strip()) for x in m.group(1).split(","))
    meta["shape"] = shape
    return meta


def shape_distribution(paths: Iterable[Path]) -> Counter:
    shapes = Counter()
    for p in paths:
        meta = probe_tif_metadata(p)
        shapes[meta.get("shape")] += 1
    return shapes


def compare_sets(csv_ids: Iterable[str], files: Iterable[str]) -> Tuple[List[str], List[str]]:
    csv_set, file_set = set(csv_ids), set(files)
    missing = sorted(csv_set - file_set)
    extra = sorted(file_set - csv_set)
    return missing, extra


def build_report(
    data_root: Path,
    report_path: Path,
    max_files: Optional[int] = None,
) -> str:
    train_csv = data_root / "train.csv"
    test_csv = data_root / "test.csv"
    train_img_dir = data_root / "train_images"
    train_lbl_dir = data_root / "train_labels"
    test_img_dir = data_root / "test_images"

    train_rows = read_csv_ids(train_csv)
    test_rows = read_csv_ids(test_csv)
    train_ids_csv = [row["id"] for row in train_rows]
    test_ids_csv = [row["id"] for row in test_rows]

    train_imgs = list(train_img_dir.glob("*.tif"))
    train_lbls = list(train_lbl_dir.glob("*.tif"))
    test_imgs = list(test_img_dir.glob("*.tif"))

    if max_files:
        train_imgs = train_imgs[:max_files]
        train_lbls = train_lbls[:max_files]
        test_imgs = test_imgs[:max_files]

    missing_train_imgs, extra_train_imgs = compare_sets(train_ids_csv, (p.stem for p in train_imgs))
    missing_train_lbls, extra_train_lbls = compare_sets(train_ids_csv, (p.stem for p in train_lbls))
    missing_test_imgs, extra_test_imgs = compare_sets(test_ids_csv, (p.stem for p in test_imgs))

    train_shapes = shape_distribution(train_imgs)
    label_shapes = shape_distribution(train_lbls)

    sample_inspects = []
    for p in train_imgs[:5]:
        sample_inspects.append(probe_tif_metadata(p))
    for p in train_lbls[:5]:
        sample_inspects.append(probe_tif_metadata(p))

    lines = []
    lines.append(f"# Rapport exploration Vesuvius Challenge")
    lines.append(f"- Racine données : `{data_root}`")
    lines.append(f"- train.csv : {len(train_rows)} lignes | test.csv : {len(test_rows)} lignes")
    lines.append(
        f"- Fichiers : train_images={len(train_imgs)}, train_labels={len(train_lbls)}, test_images={len(test_imgs)}"
    )
    if max_files:
        lines.append(f"- ⚠️ Échantillonnage activé : seulement les {max_files} premiers fichiers inspectés.")

    def fmt_missing(label: str, missing: List[str], extra: List[str]):
        if not missing and not extra:
            lines.append(f"- {label} : correspondance parfaite avec le CSV.")
            return
        if missing:
            lines.append(f"- {label} manquants vs CSV : {len(missing)} (ex: {missing[:5]})")
        if extra:
            lines.append(f"- {label} en trop vs CSV : {len(extra)} (ex: {extra[:5]})")

    fmt_missing("Images train", missing_train_imgs, extra_train_imgs)
    fmt_missing("Labels train", missing_train_lbls, extra_train_lbls)
    fmt_missing("Images test", missing_test_imgs, extra_test_imgs)

    def fmt_shapes(title: str, counter: Counter):
        if not counter:
            lines.append(f"- {title} : aucune forme lue.")
            return
        most_common = counter.most_common(5)
        shape_str = ", ".join([f"{k}: {v}" for k, v in most_common])
        uniq = len(counter)
        lines.append(f"- {title} : {uniq} forme(s) unique(s) (top 5) → {shape_str}")

    fmt_shapes("Formes images train", train_shapes)
    fmt_shapes("Formes labels train", label_shapes)

    lines.append("\n## Exemples d'inspection `file` (images puis labels)")
    for meta in sample_inspects:
        lines.append(f"- {meta['path']}: shape={meta.get('shape')} height={meta.get('height')} width={meta.get('width')} bps={meta.get('bps')}")

    report = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore rapidement le dataset Vesuvius Challenge.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw" / "vesuvius-challenge-surface-detection",
        help="Chemin vers le dossier racine contenant train.csv/test.csv/train_images/train_labels/test_images.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "analysis" / "readme.md",
        help="Chemin du rapport Markdown généré.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limiter le nombre de fichiers inspectés (pour un run rapide).",
    )
    args = parser.parse_args()

    report = build_report(args.data_root, args.report, args.max_files)
    print(report)


if __name__ == "__main__":
    main()
