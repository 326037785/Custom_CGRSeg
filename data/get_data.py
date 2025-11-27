"""Utilities to download segmentation datasets."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Tuple
from urllib.request import urlopen

ADE20K_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
COCO_TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_STUFF_MAP_URL = "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip"
COCO_IMAGE_URL = "http://images.cocodataset.org/{split}2017/{file_name}"
PASCAL_CONTEXT_URL = "http://cs.stanford.edu/~roozbeh/pascal-context/pascal-context-540.zip"


def download_file(url: str, destination: Path) -> Path:
    """Download a URL to a local path if it does not already exist."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"[skip] {destination} already exists")
        return destination

    print(f"[download] {url} -> {destination}")
    with urlopen(url) as response, open(destination, "wb") as output:
        shutil.copyfileobj(response, output)
    return destination


def extract_zip(zip_path, target_dir):
    print(f"[extract] {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        # 如果 ZIP 里有一层 top-level 文件夹，就自动剥掉
        members = z.namelist()
        root = members[0].split("/")[0]

        if all(m.startswith(root + "/") for m in members):
            z.extractall(target_dir)
            # 然后把 target_dir/root/ 下的内容提到 target_dir/
            extracted_root = target_dir / root
            for item in extracted_root.iterdir():
                shutil.move(str(item), target_dir)
            extracted_root.rmdir()
        else:
            z.extractall(target_dir)


def download_ade20k(output_dir: Path) -> None:
    """Download the ADE20K dataset archive."""
    dataset_dir = output_dir / "ade20k"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_file(ADE20K_URL, dataset_dir / "ADEChallengeData2016.zip")
    extract_zip(zip_path, dataset_dir)


def _load_coco_annotations(annotation_dir: Path, split: str) -> Tuple[list, list, list]:
    annotation_path = annotation_dir / f"instances_{split}2017.json"
    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Missing {annotation_path}. Please ensure annotations are downloaded and extracted."
        )

    with open(annotation_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    return data["images"], data["annotations"], data.get("categories", [])


def _subset_annotations(images: list, annotations: list, categories: list, count: int) -> dict:
    selected_images = images[:count]
    selected_ids = {image["id"] for image in selected_images}
    filtered_annotations = [ann for ann in annotations if ann.get("image_id") in selected_ids]

    return {
        "info": {"description": f"Subset of {count} images"},
        "images": selected_images,
        "annotations": filtered_annotations,
        "categories": categories,
    }


def _download_coco_image_set(image_entries: Iterable[dict], split: str, image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for entry in image_entries:
        image_path = image_dir / entry["file_name"]
        if image_path.exists():
            print(f"[skip] {image_path} already exists")
            continue

        url = COCO_IMAGE_URL.format(split=split, file_name=entry["file_name"])
        print(f"[download] {url} -> {image_path}")
        with urlopen(url) as response, open(image_path, "wb") as output:
            shutil.copyfileobj(response, output)


def download_coco_stuff(output_dir: Path, train_count: int | None = None, val_count: int | None = None) -> None:
    """Download COCO-Stuff images, annotations, and stuff maps.

    If ``train_count`` or ``val_count`` are provided, only that number of images
    and their annotations are downloaded for the respective split.
    """

    dataset_dir = output_dir / "coco-stuff"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    annotation_zip = download_file(COCO_ANNOTATIONS_URL, dataset_dir / "annotations_trainval2017.zip")
    annotation_dir = dataset_dir / "annotations"
    extract_zip(annotation_zip, annotation_dir)

    download_file(COCO_STUFF_MAP_URL, dataset_dir / "stuffthingmaps_trainval2017.zip")

    if train_count is None and val_count is None:
        train_zip = download_file(COCO_TRAIN_URL, dataset_dir / "train2017.zip")
        val_zip = download_file(COCO_VAL_URL, dataset_dir / "val2017.zip")
        extract_zip(train_zip, dataset_dir / "train2017")
        extract_zip(val_zip, dataset_dir / "val2017")
        return

    images, annotations, categories = _load_coco_annotations(annotation_dir, "train")
    if train_count:
        print(f"[subset] Downloading first {train_count} training images")
        subset = _subset_annotations(images, annotations, categories, train_count)
        image_dir = dataset_dir / "train2017"
        _download_coco_image_set(subset["images"], "train", image_dir)
        subset_path = annotation_dir / "instances_train2017_subset.json"
        with open(subset_path, "w", encoding="utf-8") as output:
            json.dump(subset, output)

    val_images, val_annotations, val_categories = _load_coco_annotations(annotation_dir, "val")
    if val_count:
        print(f"[subset] Downloading first {val_count} validation images")
        subset = _subset_annotations(val_images, val_annotations, val_categories, val_count)
        image_dir = dataset_dir / "val2017"
        _download_coco_image_set(subset["images"], "val", image_dir)
        subset_path = annotation_dir / "instances_val2017_subset.json"
        with open(subset_path, "w", encoding="utf-8") as output:
            json.dump(subset, output)


def download_pascal_context(output_dir: Path) -> None:
    """Download the Pascal Context dataset archive."""
    dataset_dir = output_dir / "pascal-context"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_file(PASCAL_CONTEXT_URL, dataset_dir / "pascal-context-540.zip")
    extract_zip(zip_path, dataset_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download segmentation datasets")
    parser.add_argument(
        "--dataset",
        choices=["ade20k", "coco-stuff", "pascal-context", "all"],
        default="all",
        help="Dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where datasets will be stored",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Number of COCO-Stuff training images to download (subset mode)",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=None,
        help="Number of COCO-Stuff validation images to download (subset mode)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir

    if args.dataset in ("ade20k", "all"):
        download_ade20k(output_dir)

    if args.dataset in ("coco-stuff", "all"):
        download_coco_stuff(output_dir, train_count=args.train_count, val_count=args.val_count)

    if args.dataset in ("pascal-context", "all"):
        download_pascal_context(output_dir)


if __name__ == "__main__":
    main()
