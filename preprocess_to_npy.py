#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import KFold


def _dtype_from_name(name):
    name = name.lower().strip()
    if name in ("f16", "float16", "half"):
        return np.float16, "f16"
    if name in ("f32", "float32"):
        return np.float32, "f32"
    raise ValueError(f"Unsupported dtype: {name}. Use float16 or float32.")


def _numeric_or_str_sort_key(value):
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def _save_text_list(path, values):
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            f.write(f"{v}\n")


def _resize_oct_volume(volume, out_shape):
    volume = np.asarray(volume).squeeze()
    if volume.ndim != 3:
        raise ValueError(f"OCT volume must be 3D after squeeze, got shape={volume.shape}")
    d, h, w = volume.shape
    out_d, out_h, out_w = out_shape
    scale = [out_d / float(d), out_h / float(h), out_w / float(w)]
    return ndimage.zoom(volume, scale, order=0)


def preprocess_ddr_split(
    images_root,
    split_file,
    output_dir,
    image_size=(224, 224),
    dtype=np.float16,
    dtype_tag="f16",
    normalize=True,
    progress_every=500,
):
    split_stem = Path(split_file).stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = image_size

    image_rel_paths = []
    labels_list = []
    with open(split_file, "r", encoding="utf-8") as fin:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            image_rel_paths.append(parts[0])
            labels_list.append([int(x) for x in parts[1:]])

    n = len(image_rel_paths)
    if n == 0:
        raise ValueError(f"No valid samples found in split file: {split_file}")

    max_votes = max(len(x) for x in labels_list)
    labels_arr = np.full((n, max_votes), -1, dtype=np.int64)
    for i, votes in enumerate(labels_list):
        labels_arr[i, : len(votes)] = np.asarray(votes, dtype=np.int64)

    images_npy = out_dir / f"{split_stem}_images_{h}x{w}_{dtype_tag}.npy"
    labels_npy = out_dir / f"{split_stem}_labels.npy"
    ids_txt = out_dir / f"{split_stem}_filenames.txt"

    print(f"[DDR] split={split_stem}, samples={n}")
    print(f"[DDR] writing images -> {images_npy}")
    images_mm = np.lib.format.open_memmap(
        images_npy,
        mode="w+",
        dtype=dtype,
        shape=(n, 3, h, w),
    )

    if normalize:
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    else:
        mean = None
        std = None

    bad_count = 0
    for idx, rel_path in enumerate(image_rel_paths):
        full_path = os.path.join(images_root, rel_path)
        try:
            with Image.open(full_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize((w, h), resample=Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                if normalize:
                    arr = (arr - mean) / std
                arr = np.transpose(arr, (2, 0, 1))
        except Exception as exc:
            bad_count += 1
            if bad_count <= 10:
                print(f"[DDR] failed to load {full_path}: {exc}")
            arr = np.zeros((3, h, w), dtype=np.float32)

        images_mm[idx] = arr.astype(dtype, copy=False)
        if ((idx + 1) % progress_every == 0) or (idx + 1 == n):
            print(f"[DDR] {split_stem}: {idx + 1}/{n}")

    images_mm.flush()
    np.save(labels_npy, labels_arr)
    _save_text_list(ids_txt, image_rel_paths)
    print(f"[DDR] labels -> {labels_npy}")
    print(f"[DDR] ids    -> {ids_txt}")
    if bad_count > 0:
        print(f"[DDR] warning: {bad_count} images failed and were zero-filled.")


def _build_amd_label_map(label_file):
    df = pd.read_excel(label_file)
    if "data" not in df.columns:
        raise ValueError(f"'data' column not found in {label_file}")
    if len(df.columns) < 2:
        raise ValueError(f"No class columns found in {label_file}")

    label_map = {}
    class_cols = [c for c in df.columns if c != "data"]
    for _, row in df.iterrows():
        raw_id = row["data"]
        if pd.isna(raw_id):
            continue
        try:
            key = str(int(raw_id))
        except Exception:
            key = str(raw_id).strip()
        logits = row[class_cols].to_numpy(dtype=np.float32)
        label_map[key] = int(np.argmax(logits))
    return label_map


def _collect_amd_ids(dataset_root, id_order="sorted"):
    id_order = str(id_order).lower().strip()
    ids = []
    for name in os.listdir(dataset_root):
        full_path = os.path.join(dataset_root, name)
        if os.path.isdir(full_path):
            ids.append(name)

    if id_order == "sorted":
        ids = sorted(ids, key=_numeric_or_str_sort_key)
    elif id_order == "os":
        # Keep filesystem listing order to reproduce pipelines that split on raw os.listdir().
        pass
    else:
        raise ValueError(f"Unsupported AMD id_order: {id_order}, choose from ['sorted', 'os']")

    return ids


def _select_amd_split_ids(all_ids, split_mode="all", fold=0, n_splits=5, seed=10):
    split_mode = split_mode.lower()
    if split_mode == "all":
        return {"all": all_ids}
    if split_mode != "kfold":
        raise ValueError(f"Unsupported AMD split_mode: {split_mode}")

    if len(all_ids) < n_splits:
        raise ValueError(f"Not enough samples for KFold: {len(all_ids)} < {n_splits}")
    if fold < 0 or fold >= n_splits:
        raise ValueError(f"Invalid fold index: {fold}, expected [0, {n_splits - 1}]")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_splits = list(kf.split(np.arange(len(all_ids))))
    train_idx, val_idx = fold_splits[fold]
    all_ids_arr = np.asarray(all_ids)
    return {
        "train": all_ids_arr[train_idx].tolist(),
        "val": all_ids_arr[val_idx].tolist(),
    }


def preprocess_amd_oct(
    dataset_root,
    label_file,
    output_dir,
    split_mode="all",
    id_order="sorted",
    fold=0,
    n_splits=5,
    seed=10,
    out_shape=(96, 96, 96),
    dtype=np.float16,
    dtype_tag="f16",
    progress_every=100,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_map = _build_amd_label_map(label_file)
    all_ids = _collect_amd_ids(dataset_root, id_order=id_order)
    split_to_ids = _select_amd_split_ids(all_ids, split_mode=split_mode, fold=fold, n_splits=n_splits, seed=seed)
    out_d, out_h, out_w = out_shape
    print(f"[AMD] id_order={id_order}, split_mode={split_mode}, fold={fold}, seed={seed}")

    for split_name, ids in split_to_ids.items():
        n = len(ids)
        if n == 0:
            raise ValueError(f"No samples in AMD split '{split_name}'")

        images_npy = out_dir / f"AMD_{split_name}_oct_{out_d}x{out_h}x{out_w}_{dtype_tag}.npy"
        labels_npy = out_dir / f"AMD_{split_name}_labels.npy"
        ids_txt = out_dir / f"AMD_{split_name}_ids.txt"

        print(f"[AMD] split={split_name}, samples={n}")
        print(f"[AMD] writing oct -> {images_npy}")
        images_mm = np.lib.format.open_memmap(
            images_npy,
            mode="w+",
            dtype=dtype,
            shape=(n, 3, out_d, out_h, out_w),
        )
        labels_arr = np.empty((n,), dtype=np.int64)

        skipped = 0
        for idx, sample_id in enumerate(ids):
            try:
                key = str(int(sample_id)) if sample_id.isdigit() else sample_id.strip()
                if key not in label_map:
                    raise KeyError(f"label not found for id={sample_id}")
                labels_arr[idx] = label_map[key]

                nii_path = os.path.join(dataset_root, sample_id, f"processed_data_{sample_id}.nii")
                vol = nib.load(nii_path).get_fdata()
                vol = _resize_oct_volume(vol, out_shape).astype(np.float32, copy=False)
                np.nan_to_num(vol, copy=False)
                vol = np.repeat(vol[None, ...], 3, axis=0)
                images_mm[idx] = vol.astype(dtype, copy=False)
            except Exception as exc:
                skipped += 1
                if skipped <= 10:
                    print(f"[AMD] failed sample {sample_id}: {exc}")
                labels_arr[idx] = -1
                images_mm[idx] = np.zeros((3, out_d, out_h, out_w), dtype=dtype)

            if ((idx + 1) % progress_every == 0) or (idx + 1 == n):
                print(f"[AMD] {split_name}: {idx + 1}/{n}")

        images_mm.flush()
        np.save(labels_npy, labels_arr)
        _save_text_list(ids_txt, ids)
        print(f"[AMD] labels -> {labels_npy}")
        print(f"[AMD] ids    -> {ids_txt}")
        if skipped > 0:
            print(f"[AMD] warning: {skipped} samples failed and were zero-filled (label=-1).")


def preprocess_eyeq_csv(
    csv_file,
    images_root,
    output_dir,
    image_col="image",
    label_col="quality",
    image_size=(224, 224),
    dtype=np.float16,
    dtype_tag="f16",
    normalize=True,
    progress_every=500,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = image_size
    stem = Path(csv_file).stem

    df = pd.read_csv(csv_file)
    if image_col not in df.columns:
        raise ValueError(f"Column '{image_col}' not found in {csv_file}")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in {csv_file}")

    image_rel_paths = df[image_col].astype(str).tolist()
    labels_arr = df[label_col].astype(int).to_numpy(dtype=np.int64)
    n = len(image_rel_paths)
    if n == 0:
        raise ValueError(f"No samples found in csv: {csv_file}")

    images_npy = out_dir / f"{stem}_images_{h}x{w}_{dtype_tag}.npy"
    labels_npy = out_dir / f"{stem}_labels.npy"
    ids_txt = out_dir / f"{stem}_filenames.txt"

    print(f"[EyeQ] split={stem}, samples={n}")
    print(f"[EyeQ] writing images -> {images_npy}")
    images_mm = np.lib.format.open_memmap(
        images_npy,
        mode="w+",
        dtype=dtype,
        shape=(n, 3, h, w),
    )

    if normalize:
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    else:
        mean = None
        std = None

    bad_count = 0
    for idx, rel_path in enumerate(image_rel_paths):
        full_path = os.path.join(images_root, rel_path) if images_root else rel_path
        try:
            with Image.open(full_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize((w, h), resample=Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                if normalize:
                    arr = (arr - mean) / std
                arr = np.transpose(arr, (2, 0, 1))
        except Exception as exc:
            bad_count += 1
            if bad_count <= 10:
                print(f"[EyeQ] failed to load {full_path}: {exc}")
            arr = np.zeros((3, h, w), dtype=np.float32)

        images_mm[idx] = arr.astype(dtype, copy=False)
        if ((idx + 1) % progress_every == 0) or (idx + 1 == n):
            print(f"[EyeQ] {stem}: {idx + 1}/{n}")

    images_mm.flush()
    np.save(labels_npy, labels_arr)
    _save_text_list(ids_txt, image_rel_paths)
    print(f"[EyeQ] labels -> {labels_npy}")
    print(f"[EyeQ] ids    -> {ids_txt}")
    if bad_count > 0:
        print(f"[EyeQ] warning: {bad_count} images failed and were zero-filled.")


def build_parser():
    parser = argparse.ArgumentParser(description="Offline preprocessing to NPY for DDR / AMD / EyeQ datasets.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    parser_ddr = subparsers.add_parser("ddr", help="Convert DDR split txt + images to npy.")
    parser_ddr.add_argument("--images-root", required=True, help="DDR image root directory.")
    parser_ddr.add_argument(
        "--split-file",
        required=True,
        action="append",
        help="DDR split txt file. Repeat this arg to process multiple splits.",
    )
    parser_ddr.add_argument("--output-dir", required=True, help="Output directory for npy files.")
    parser_ddr.add_argument("--height", type=int, default=224, help="Image height.")
    parser_ddr.add_argument("--width", type=int, default=224, help="Image width.")
    parser_ddr.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Output image dtype.")
    parser_ddr.add_argument("--no-normalize", action="store_true", help="Disable ImageNet normalization.")
    parser_ddr.add_argument("--progress-every", type=int, default=500, help="Progress print interval.")

    parser_amd = subparsers.add_parser("amd", help="Convert AMD OCT nii to npy.")
    parser_amd.add_argument("--dataset-root", required=True, help="AMD dataset root with ID subfolders.")
    parser_amd.add_argument("--label-file", required=True, help="AMD label excel file (train.xlsx).")
    parser_amd.add_argument("--output-dir", required=True, help="Output directory for npy files.")
    parser_amd.add_argument("--split-mode", default="kfold", choices=["all", "kfold"], help="Split strategy.")
    parser_amd.add_argument(
        "--id-order",
        default="sorted",
        choices=["sorted", "os"],
        help="AMD ID ordering before KFold. Use 'os' to match scripts that split directly on os.listdir().",
    )
    parser_amd.add_argument("--fold", type=int, default=0, help="Fold index for kfold split.")
    parser_amd.add_argument("--n-splits", type=int, default=5, help="Number of folds for kfold split.")
    parser_amd.add_argument("--seed", type=int, default=10, help="Random seed for kfold split.")
    parser_amd.add_argument("--depth", type=int, default=96, help="Output depth.")
    parser_amd.add_argument("--height", type=int, default=96, help="Output height.")
    parser_amd.add_argument("--width", type=int, default=96, help="Output width.")
    parser_amd.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Output OCT dtype.")
    parser_amd.add_argument("--progress-every", type=int, default=100, help="Progress print interval.")

    parser_eyeq = subparsers.add_parser("eyeq", help="Convert EyeQ csv + images to npy.")
    parser_eyeq.add_argument("--images-root", default=None, help="EyeQ image root directory.")
    parser_eyeq.add_argument(
        "--csv-file",
        required=True,
        action="append",
        help="EyeQ csv file. Repeat this arg to process multiple splits.",
    )
    parser_eyeq.add_argument("--output-dir", required=True, help="Output directory for npy files.")
    parser_eyeq.add_argument("--image-col", default="image", help="Image path column name in csv.")
    parser_eyeq.add_argument("--label-col", default="quality", help="Label column name in csv.")
    parser_eyeq.add_argument("--height", type=int, default=224, help="Image height.")
    parser_eyeq.add_argument("--width", type=int, default=224, help="Image width.")
    parser_eyeq.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Output image dtype.")
    parser_eyeq.add_argument("--no-normalize", action="store_true", help="Disable ImageNet normalization.")
    parser_eyeq.add_argument("--progress-every", type=int, default=500, help="Progress print interval.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset == "ddr":
        dtype, dtype_tag = _dtype_from_name(args.dtype)
        for split_file in args.split_file:
            preprocess_ddr_split(
                images_root=args.images_root,
                split_file=split_file,
                output_dir=args.output_dir,
                image_size=(args.height, args.width),
                dtype=dtype,
                dtype_tag=dtype_tag,
                normalize=(not args.no_normalize),
                progress_every=args.progress_every,
            )
        return

    if args.dataset == "amd":
        dtype, dtype_tag = _dtype_from_name(args.dtype)
        preprocess_amd_oct(
            dataset_root=args.dataset_root,
            label_file=args.label_file,
            output_dir=args.output_dir,
            split_mode=args.split_mode,
            id_order=args.id_order,
            fold=args.fold,
            n_splits=args.n_splits,
            seed=args.seed,
            out_shape=(args.depth, args.height, args.width),
            dtype=dtype,
            dtype_tag=dtype_tag,
            progress_every=args.progress_every,
        )
        return

    if args.dataset == "eyeq":
        dtype, dtype_tag = _dtype_from_name(args.dtype)
        for csv_file in args.csv_file:
            preprocess_eyeq_csv(
                csv_file=csv_file,
                images_root=args.images_root,
                output_dir=args.output_dir,
                image_col=args.image_col,
                label_col=args.label_col,
                image_size=(args.height, args.width),
                dtype=dtype,
                dtype_tag=dtype_tag,
                normalize=(not args.no_normalize),
                progress_every=args.progress_every,
            )
        return

    raise ValueError(f"Unknown dataset type: {args.dataset}")


if __name__ == "__main__":
    main()
