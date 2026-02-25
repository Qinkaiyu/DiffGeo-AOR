# Imports
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import models_vit
import torch.nn.functional as F
import argparse
import sys
import random
import os.path as osp
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import label_binarize
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import multiprocessing
from diffloss import DiffLoss
from timm.models.layers import trunc_normal_
import math
from einops import rearrange, repeat
import torchvision.models as models
import timm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm
from collections import Counter
import cv2
import nibabel as nib
from sklearn.model_selection import KFold
from resnet3d import ResNet3DEncoder
import numpy as np
from scipy import ndimage
import pandas as pd
from dataloader import ExcelDataset, RegressionDataset,GAMMA_dataset
from util import task_importance_weights,SimpleFusion,RankEmbed,RankReference,GeMPool,TopKAttnPool,FiLMBlock,StepQueryFusion
from build_model import prepare_model_resnet3d,prepare_model_vit,ViTFeatureExtractor,Resnet3dFeatureExtractor

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def label_to_vector(labels, max_length=7):
    '''
    :param labels: [batch_size]，包含标签的张量
    :param max_length: 向量的最大长度
    :return: [batch_size, max_length]，转换后的向量
    '''
    batch_size = labels.size(0)
    vectors = torch.zeros(batch_size, max_length, dtype=torch.int)

    for i in range(batch_size):
        label = labels[i].item()
        vectors[i, :label] = 1

    return vectors
class UnifiedNpyDataset(Dataset):
    def __init__(
        self,
        images_npy,
        labels_npy,
        label_mode="direct",
        expects_3d=True,
        mmap_mode="r",
        normalize_mode="none",
        return_float32=True,
    ):
        self.images_npy = images_npy
        self.labels_npy = labels_npy
        self.label_mode = label_mode
        self.expects_3d = bool(expects_3d)
        self.mmap_mode = mmap_mode
        self.normalize_mode = str(normalize_mode).lower().strip()
        self.return_float32 = bool(return_float32)
        self.images = None
        self.labels = None

        labels_preview = np.load(self.labels_npy, mmap_mode="r")
        self.num_samples = int(labels_preview.shape[0])
        del labels_preview
        print(
            f"[npy-dataset] images={self.images_npy}, labels={self.labels_npy}, "
            f"num_samples={self.num_samples}, expects_3d={self.expects_3d}, "
            f"normalize_mode={self.normalize_mode}, return_float32={self.return_float32}"
        )
        print("[npy-dataset] use direct tensor path (no PIL / no extra image transform)")

    def _lazy_load(self):
        if self.images is None:
            self.images = np.load(self.images_npy, mmap_mode=self.mmap_mode)
        if self.labels is None:
            self.labels = np.load(self.labels_npy, mmap_mode="r")

    def _select_label(self, raw_label):
        if np.isscalar(raw_label):
            return int(raw_label)

        labels = np.asarray(raw_label, dtype=np.int64).reshape(-1)
        labels = labels[labels >= 0]
        if labels.size == 0:
            return 0
        if self.label_mode == "random":
            return int(labels[random.randrange(labels.size)])
        if self.label_mode == "center":
            return int(labels[labels.size // 2])
        if self.label_mode == "majority":
            counts = np.bincount(labels)
            return int(np.argmax(counts))
        return int(labels[0])

    def __getitem__(self, index):
        self._lazy_load()
        image_np = np.asarray(self.images[index])
        # Align 3D npy layout to legacy ToTensor ordering (D, H, W):
        # (C, D, H, W) -> (C, W, D, H)
        if self.expects_3d and image_np.ndim == 4 and image_np.shape[0] in (1, 3):
            image_np = np.transpose(image_np, (0, 3, 1, 2))
        image = torch.from_numpy(image_np)
        image = image.float() if self.return_float32 else image.half()
        label = self._select_label(self.labels[index])
        return image, label

    def __len__(self):
        return self.num_samples


def _canonical_sample_id(x):
    s = str(x).strip()
    if s.isdigit():
        return str(int(s))
    return s


def _read_ids_txt(ids_txt):
    with open(ids_txt, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _read_test_ids_from_csv(csv_path):
    df = pd.read_csv(csv_path, dtype=str)
    if df.shape[1] == 0:
        raise ValueError(f"empty csv: {csv_path}")

    candidate_cols = ["id", "ID", "data", "sample_id", "sampleID", "case_id"]
    id_col = None
    for c in candidate_cols:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]

    ids = []
    for v in df[id_col].tolist():
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        ids.append(s)
    return ids, id_col


class MultiSourceNpyDataset(Dataset):
    """
    Select samples by ID from multiple npy shards (e.g., AMD_train + AMD_val),
    so evaluation can exactly follow an external ID list.
    """

    def __init__(self, sources, selected_ids, label_mode="direct", expects_3d=True, mmap_mode="r", return_float32=True):
        self.sources = list(sources)
        self.label_mode = label_mode
        self.expects_3d = bool(expects_3d)
        self.mmap_mode = mmap_mode
        self.return_float32 = bool(return_float32)

        self.images = [None] * len(self.sources)
        self.labels = [None] * len(self.sources)
        self.sample_locs = []

        id_to_loc = {}
        for src_idx, src in enumerate(self.sources):
            ids = _read_ids_txt(src["ids_txt"])
            labels_preview = np.load(src["labels_npy"], mmap_mode="r")
            n = int(labels_preview.shape[0])
            del labels_preview
            if len(ids) != n:
                raise ValueError(
                    f"ids/labels length mismatch for source[{src_idx}]: ids={len(ids)}, labels={n}, ids_txt={src['ids_txt']}"
                )
            for row_idx, sid in enumerate(ids):
                raw_key = str(sid).strip()
                can_key = _canonical_sample_id(raw_key)
                id_to_loc[raw_key] = (src_idx, row_idx)
                id_to_loc[can_key] = (src_idx, row_idx)

        missing = []
        for sid in selected_ids:
            raw_key = str(sid).strip()
            can_key = _canonical_sample_id(raw_key)
            loc = id_to_loc.get(raw_key, id_to_loc.get(can_key))
            if loc is None:
                missing.append(raw_key)
            else:
                self.sample_locs.append(loc)

        if missing:
            raise ValueError(f"{len(missing)} selected ids not found in npy shards, e.g. {missing[:10]}")

        print(
            f"[multi-npy-dataset] sources={len(self.sources)}, selected={len(self.sample_locs)}, "
            f"expects_3d={self.expects_3d}, return_float32={self.return_float32}"
        )

    def _lazy_load(self, src_idx):
        if self.images[src_idx] is None:
            self.images[src_idx] = np.load(self.sources[src_idx]["images_npy"], mmap_mode=self.mmap_mode)
        if self.labels[src_idx] is None:
            self.labels[src_idx] = np.load(self.sources[src_idx]["labels_npy"], mmap_mode="r")

    def _select_label(self, raw_label):
        if np.isscalar(raw_label):
            return int(raw_label)
        labels = np.asarray(raw_label, dtype=np.int64).reshape(-1)
        labels = labels[labels >= 0]
        if labels.size == 0:
            return 0
        if self.label_mode == "random":
            return int(labels[random.randrange(labels.size)])
        if self.label_mode == "center":
            return int(labels[labels.size // 2])
        if self.label_mode == "majority":
            counts = np.bincount(labels)
            return int(np.argmax(counts))
        return int(labels[0])

    def __getitem__(self, index):
        src_idx, row_idx = self.sample_locs[index]
        self._lazy_load(src_idx)

        image_np = np.asarray(self.images[src_idx][row_idx])
        if self.expects_3d and image_np.ndim == 4 and image_np.shape[0] in (1, 3):
            image_np = np.transpose(image_np, (0, 3, 1, 2))
        image = torch.from_numpy(image_np)
        image = image.float() if self.return_float32 else image.half()

        label = self._select_label(self.labels[src_idx][row_idx])
        return image, label

    def __len__(self):
        return len(self.sample_locs)


def cohen_quadratic_kappa(preds, gt, num_classes):
    """
    计算模型预测和 GT 的 Cohen's Quadratic Kappa
    :param preds: 模型预测结果 (1D list or array)
    :param gt: 真实标签 (1D list or array)
    :param num_classes: 分类总数
    :return: Cohen's Quadratic Kappa 值
    """
    gt = np.array(gt).astype(int)  # 确保 gt 是整数数组
    preds = np.array(preds).astype(int)  # 确保 pred 是整数数组
    # 构造混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes))
    for p, g in zip(preds, gt):
        confusion_matrix[g][p] += 1

    # 创建二次加权矩阵
    weights = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i][j] = (i - j) ** 2 / (num_classes - 1) ** 2

    # 计算观察值 (Observed Agreement)
    observed_agreement = confusion_matrix / np.sum(confusion_matrix)

    # 计算期望值 (Expected Agreement)
    hist_gt = np.sum(confusion_matrix, axis=1)  # GT 的分布
    hist_preds = np.sum(confusion_matrix, axis=0)  # 模型预测的分布
    expected_agreement = np.outer(hist_gt, hist_preds) / np.sum(confusion_matrix)

    # 加权求和
    observed_score = np.sum(weights * observed_agreement)
    expected_score = np.sum(weights * (expected_agreement / np.sum(confusion_matrix)))

    # 计算 Kappa 值
    kappa = 1 - (observed_score / expected_score)
    return kappa

def infer_batch_predictions(feature_extractor, image, number_of_classes=5):
    all_probas = []
    fusion_feature = None
    for t in range(number_of_classes-1):
        if t == 0:

            probas, features = feature_extractor.sample(image, t, temperature=1.0, cfg=1.0)
            fusion_feature = feature_extractor.query_builder(features, probas, t)
            # all_probas.append(probas)
        else:
            probas, fusion_feature = feature_extractor.sample(fusion_feature, t, temperature=1.0, cfg=1.0)
            fusion_feature = feature_extractor.query_builder(fusion_feature, probas, t)
            # all_probas.append(probas)

        all_probas.append(torch.round(probas))
    all_probas = [torch.round(p) for p in all_probas]
    all_probas_tensor = torch.stack(all_probas).squeeze(-1)  # 变成 [64, 5]
    all_probas_tensor = torch.stack(all_probas).transpose(0, 1)  # 变成 [64, 5]
    pred = torch.sum(all_probas_tensor, dim=1)
    return pred

def evaluate_once(feature_extractor, valid_loader, number_of_classes=5):
    feature_extractor.eval()
    correct = 0
    total = 0
    all_pred = []
    all_targets = []

    with torch.no_grad():
        for image, targets in valid_loader:
            image = image.to(device='cuda', non_blocking=True)
            targets = targets.to(device='cuda', non_blocking=True)

            pred = infer_batch_predictions(feature_extractor, image, number_of_classes=number_of_classes)

            all_pred.append(pred.detach().cpu().numpy().reshape(-1))
            all_targets.append(targets.detach().cpu().numpy().reshape(-1))
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    all_pred = np.concatenate(all_pred, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    cm = confusion_matrix(all_targets, all_pred, labels=list(range(number_of_classes)))
    kappa = cohen_quadratic_kappa(all_pred, all_targets, number_of_classes)

    sensityvity = []
    specificity = []
    for i in range(number_of_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tp = cm[i, i]
        sensitivity_i = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensityvity.append(sensitivity_i)
        specificity.append(specificity_i)

    f1 = f1_score(all_targets, all_pred, average='macro')
    mae = mean_absolute_error(all_targets, all_pred)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'mae': mae,
        'kappa': kappa,
        'sensitivity': np.array(sensityvity, dtype=np.float64),
        'specificity': np.array(specificity, dtype=np.float64),
        'cm': cm,
    }

def benchmark_pure_inference(feature_extractor, valid_loader, num_batches=20, warmup_batches=5):
    if not torch.cuda.is_available():
        return None

    cached_images = []
    for image, _ in valid_loader:
        cached_images.append(image.to(device='cuda', non_blocking=True))
        if len(cached_images) >= num_batches:
            break

    if len(cached_images) == 0:
        return None

    feature_extractor.eval()
    with torch.no_grad():
        for image in cached_images[:min(warmup_batches, len(cached_images))]:
            _ = infer_batch_predictions(feature_extractor, image)
        torch.cuda.synchronize()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        total_ms = 0.0
        total_samples = 0

        for image in cached_images:
            starter.record()
            _ = infer_batch_predictions(feature_extractor, image)
            ender.record()
            torch.cuda.synchronize()
            total_ms += starter.elapsed_time(ender)
            total_samples += image.size(0)

    total_seconds = total_ms / 1000.0
    avg_batch_ms = total_ms / len(cached_images)
    avg_sample_ms = total_ms / total_samples if total_samples > 0 else 0.0
    fps = total_samples / total_seconds if total_seconds > 0 else 0.0

    return {
        'num_batches': len(cached_images),
        'num_samples': total_samples,
        'avg_batch_ms': avg_batch_ms,
        'avg_sample_ms': avg_sample_ms,
        'fps': fps,
    }

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    DATASET_NAME = os.environ.get("AORDR_DATASET", "amd").strip().lower()  # eyeq / ddr / amd
    TRAIN_SPLIT = os.environ.get("AORDR_TRAIN_SPLIT", "train").strip().lower()
    TEST_SPLIT = os.environ.get("AORDR_TEST_SPLIT", "test").strip().lower()
    TSNE_EVERY_EPOCHS = int(os.environ.get("AORDR_TSNE_EVERY", "0"))
    TSNE_MAX_BATCHES = int(os.environ.get("AORDR_TSNE_MAX_BATCHES", "-1"))  # -1: use all batches
    TSNE_FILM_STEP = int(os.environ.get("AORDR_TSNE_FILM_STEP", "-1"))  # -1: collect all t>0 steps
    TSNE_LABEL_VECTOR_LENGTH = int(os.environ.get("AORDR_TSNE_LABEL_VEC_LEN", "7"))
    TSNE_OUTPUT_DIR = os.environ.get("AORDR_TSNE_OUTDIR", "./tsne_visualization_fig")
    TSNE_SPLIT_NAME = os.environ.get("AORDR_TSNE_SPLIT_NAME", "val").strip()
    TSNE_SAVE_ONLY = os.environ.get("AORDR_TSNE_SAVE_ONLY", "1").strip().lower() in ("1", "true", "yes", "y")
    TSNE_PLOT_SAMPLE_SCATTER = os.environ.get("AORDR_TSNE_SAMPLE_SCATTER", "0").strip().lower() in ("1", "true", "yes", "y")
    TSNE_SHOW_PROGRESS = os.environ.get("AORDR_TSNE_PROGRESS", "1").strip().lower() in ("1", "true", "yes", "y")

    dataset_specs = {
        "eyeq": {
            "num_classes": 3,
            "backbone": "vit2d",
            "checkpoint": "./pth",
            "expects_3d": False,
            "splits": {
                "train": {
                    "images_npy": r"/mnt/datastore1/qinkaiyu/EyeQ/cache_npy/Label_EyeQ_train_images_224x224_f16.npy",
                    "labels_npy": r"/mnt/datastore1/qinkaiyu/EyeQ/cache_npy/Label_EyeQ_train_labels.npy",
                    "label_mode": "direct",
                },
                "test": {
                    "images_npy": r"/mnt/datastore1/qinkaiyu/EyeQ/cache_npy/Label_EyeQ_test_images_224x224_f16.npy",
                    "labels_npy": r"/mnt/datastore1/qinkaiyu/EyeQ/cache_npy/Label_EyeQ_test_labels.npy",
                    "label_mode": "direct",
                },
            },
        },
        "ddr": {
            "num_classes": 5,
            "backbone": "vit2d",
            "checkpoint": "./pth",
            "expects_3d": False,
            "splits": {
                "train": {
                    "images_npy": r"/path/DDR_train_images_224x224_f16.npy",
                    "labels_npy": r"/path/cache_npy/DDR_train_labels.npy",
                    "label_mode": "random",
                },
                "test": {
                    "images_npy": r"/path/cache_npy/DDR_crossval_images_224x224_f16.npy",
                    "labels_npy": r"/path/cache_npy/DDR_crossval_labels.npy",
                    "label_mode": "center",
                },
            },
        },
        "amd": {
            "num_classes": 4,
            "backbone": "resnet3d",
            "checkpoint": "./pth",
            "expects_3d": True,
            "gamma_test_ids_csv": r"/path/amd.csv",
            "splits": {
                "train": {
                    "images_npy": r"/path/cache_npy/AMD_train_oct_96x96x96_f16.npy",
                    "labels_npy": r"/path/cache_npy/AMD_train_labels.npy",
                    "ids_txt": r"/path/amd/cache_npy/AMD_train_ids.txt",
                    "label_mode": "direct",
                },
                "test": {
                    "images_npy": r"/path/AMD_val_oct_96x96x96_f16.npy",
                    "labels_npy": r"/path/AMD_val_labels.npy",
                    "ids_txt": r"/path/AMD_val_ids.txt",
                    "label_mode": "direct",
                },
            },
        },
    }

    if DATASET_NAME not in dataset_specs:
        raise ValueError(f"Unsupported dataset: {DATASET_NAME}, choose from {list(dataset_specs.keys())}")

    dataset_spec = dataset_specs[DATASET_NAME]
    if TRAIN_SPLIT not in dataset_spec["splits"]:
        raise ValueError(f"{DATASET_NAME} unsupported train split: {TRAIN_SPLIT}")
    if TEST_SPLIT not in dataset_spec["splits"]:
        raise ValueError(f"{DATASET_NAME} unsupported test split: {TEST_SPLIT}")

    train_cfg = dataset_spec["splits"][TRAIN_SPLIT]
    test_cfg = dataset_spec["splits"][TEST_SPLIT]
    expects_3d = bool(dataset_spec["expects_3d"])
    dataset_num_classes = int(dataset_spec["num_classes"])

    for split_name, split_cfg in (("train", train_cfg), ("test", test_cfg)):
        if not os.path.exists(split_cfg["images_npy"]):
            raise FileNotFoundError(f"未找到 {DATASET_NAME} {split_name} images npy: {split_cfg['images_npy']}")
        if not os.path.exists(split_cfg["labels_npy"]):
            raise FileNotFoundError(f"未找到 {DATASET_NAME} {split_name} labels npy: {split_cfg['labels_npy']}")
        if "ids_txt" in split_cfg and not os.path.exists(split_cfg["ids_txt"]):
            raise FileNotFoundError(f"未找到 {DATASET_NAME} {split_name} ids txt: {split_cfg['ids_txt']}")

    if dataset_spec["backbone"] == "resnet3d":
        backbone = prepare_model_resnet3d()
        feature_extractor = Resnet3dFeatureExtractor(backbone)
    else:
        backbone = prepare_model_vit()
        feature_extractor = ViTFeatureExtractor(backbone)

    checkpoint_path = dataset_spec["checkpoint"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到模型权重: {checkpoint_path}")
    # feature_extractor.load_state_dict(torch.load(checkpoint_path))

    print(
        f"[dataset] dataset={DATASET_NAME}, train_split={TRAIN_SPLIT}, test_split={TEST_SPLIT}, "
        f"expects_3d={expects_3d}, num_classes={dataset_num_classes}"
    )

    feature_extractor.to(device = 'cuda')
    for param in feature_extractor.parameters():
        param.requires_grad = True
    # for param in feature_extractor.head.parameters():
    #     param.requires_grad = True

    train_dataset = UnifiedNpyDataset(
        images_npy=train_cfg["images_npy"],
        labels_npy=train_cfg["labels_npy"],
        label_mode=train_cfg["label_mode"],
        expects_3d=expects_3d,
        mmap_mode="r",
        normalize_mode="none",
    )
    num_trainable_params = count_trainable_parameters(feature_extractor)
    print(f"可训练参数的数量: {num_trainable_params}")
    
    if DATASET_NAME == "amd":
        gamma_test_ids_csv = dataset_spec["gamma_test_ids_csv"]
        if not os.path.exists(gamma_test_ids_csv):
            raise FileNotFoundError(f"未找到测试ID列表: {gamma_test_ids_csv}")
        test_ids, id_col = _read_test_ids_from_csv(gamma_test_ids_csv)
        print(f"[test-id-csv] file={gamma_test_ids_csv}, id_col={id_col}, n_ids={len(test_ids)}")
        valid_dataset1 = MultiSourceNpyDataset(
            sources=[
                {
                    "images_npy": train_cfg["images_npy"],
                    "labels_npy": train_cfg["labels_npy"],
                    "ids_txt": train_cfg["ids_txt"],
                },
                {
                    "images_npy": test_cfg["images_npy"],
                    "labels_npy": test_cfg["labels_npy"],
                    "ids_txt": test_cfg["ids_txt"],
                },
            ],
            selected_ids=test_ids,
            label_mode=test_cfg["label_mode"],
            expects_3d=expects_3d,
            mmap_mode="r",
            return_float32=True,
        )
    else:
        valid_dataset1 = UnifiedNpyDataset(
            images_npy=test_cfg["images_npy"],
            labels_npy=test_cfg["labels_npy"],
            label_mode=test_cfg["label_mode"],
            expects_3d=expects_3d,
            mmap_mode="r",
            normalize_mode="none",
        )

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=8)
    valid_loader1 = DataLoader(dataset=valid_dataset1,
                              batch_size=32,
                              shuffle=True,
                              num_workers=8)




    best_accuracy1 = 0
    best_accuracy2 = 0
    eff_batch_size = 64
    lr = 1e-4
    optimizer = torch.optim.AdamW(feature_extractor.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    print('-------------------------------train-------------------------')
    best_f1 = 0
    eval_runs = 1
    for epoch in range(100):
        feature_extractor.train()
        total_cost = 0
        for batch_idx, (image, targets) in enumerate(train_loader):
            image = image.to(device = 'cuda')
            targets = targets.to(device = 'cuda')
            # print('targets',targets)
            total_loss = 0
            vectors_targets = label_to_vector(targets)
            vectors_targets = vectors_targets.T.float().to(device = 'cuda')
            for t in range(4):
                if t == 0:
                    initial_p = torch.zeros_like(vectors_targets[t]).unsqueeze(1)
                    loss,p,fusion_feature = feature_extractor(image, initial_p,vectors_targets[t],t,targets)
                else:
                    loss,p,fusion_feature = feature_extractor(fusion_feature, p,vectors_targets[t],t,targets)
                    # weight= 1
                    # loss = loss * weight
                total_loss += loss
            print('total_loss',total_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_cost += total_loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_cost / len(train_loader)}")
        with open('vit-b_OCT_amd_ab_wo_step.txt', 'a') as f:
            f.write(f'Epoch {epoch}, Average Loss: {total_cost / len(train_loader):.4f}\n')
        print('-------------------------------valid-------------------------')
        run_metrics = []
        for run_idx in range(eval_runs):
            metrics = evaluate_once(feature_extractor, valid_loader1, number_of_classes=dataset_num_classes)
            run_metrics.append(metrics)
            print(
                f"[valid run {run_idx + 1}/{eval_runs}] "
                f"acc={metrics['accuracy']:.4f}, "
                f"f1={metrics['f1']:.4f}, "
                f"mae={metrics['mae']:.4f}, "
                f"kappa={metrics['kappa']:.4f}, "
                f"sensitivity={metrics['sensitivity'].mean():.4f}, "
                f"specificity={metrics['specificity'].mean():.4f}"
            )

        acc_list = np.array([m['accuracy'] for m in run_metrics], dtype=np.float64)
        f1_list = np.array([m['f1'] for m in run_metrics], dtype=np.float64)
        mae_list = np.array([m['mae'] for m in run_metrics], dtype=np.float64)
        kappa_list = np.array([m['kappa'] for m in run_metrics], dtype=np.float64)
        sens_list = np.array([m['sensitivity'].mean() for m in run_metrics], dtype=np.float64)
        spec_list = np.array([m['specificity'].mean() for m in run_metrics], dtype=np.float64)

        acc_mean, acc_std = acc_list.mean(), (acc_list.std(ddof=1) if len(acc_list) > 1 else 0.0)
        f1_mean, f1_std = f1_list.mean(), (f1_list.std(ddof=1) if len(f1_list) > 1 else 0.0)
        mae_mean, mae_std = mae_list.mean(), (mae_list.std(ddof=1) if len(mae_list) > 1 else 0.0)
        kappa_mean, kappa_std = kappa_list.mean(), (kappa_list.std(ddof=1) if len(kappa_list) > 1 else 0.0)
        sens_mean, sens_std = sens_list.mean(), (sens_list.std(ddof=1) if len(sens_list) > 1 else 0.0)
        spec_mean, spec_std = spec_list.mean(), (spec_list.std(ddof=1) if len(spec_list) > 1 else 0.0)

        # print('cm (run1)', run_metrics[0]['cm'])
        # print(
        #     f"[valid summary] "
        #     f"acc={acc_mean:.4f}+/-{acc_std:.4f}, "
        #     f"f1={f1_mean:.4f}+/-{f1_std:.4f}, "
        #     f"mae={mae_mean:.4f}+/-{mae_std:.4f}, "
        #     f"kappa={kappa_mean:.4f}+/-{kappa_std:.4f}, "
        #     f"sensitivity={sens_mean:.4f}+/-{sens_std:.4f}, "
        #     f"specificity={spec_mean:.4f}+/-{spec_std:.4f}"
        # )

        # speed_stats = benchmark_pure_inference(feature_extractor, valid_loader1, num_batches=20, warmup_batches=5)
        # if speed_stats is not None:
        #     print(
        #         f"[pure inference speed] "
        #         f"batches={speed_stats['num_batches']}, "
        #         f"samples={speed_stats['num_samples']}, "
        #         f"avg_batch_ms={speed_stats['avg_batch_ms']:.2f}, "
        #         f"avg_sample_ms={speed_stats['avg_sample_ms']:.3f}, "
        #         f"fps={speed_stats['fps']:.2f}"
        #     )

        # if TSNE_EVERY_EPOCHS > 0 and ((epoch + 1) % TSNE_EVERY_EPOCHS == 0):
        #     from tsne_visualization import save_debug_tensors_for_tsne
        #     raw_npz_path = save_debug_tensors_for_tsne(
        #         feature_extractor=feature_extractor,
        #         data_loader=train_loader,
        #         output_dir=TSNE_OUTPUT_DIR,
        #         epoch_index=epoch + 1,
        #         split_name="train",
        #         device="cuda",
        #         max_batches=TSNE_MAX_BATCHES,
        #         film_step=TSNE_FILM_STEP,
        #         number_of_classes=dataset_num_classes,
        #         label_vector_length=TSNE_LABEL_VECTOR_LENGTH,
        #         show_progress=TSNE_SHOW_PROGRESS,
        #     )
        #     raw_npz_path = save_debug_tensors_for_tsne(
        #         feature_extractor=feature_extractor,
        #         data_loader=valid_loader1,
        #         output_dir=TSNE_OUTPUT_DIR,
        #         epoch_index=epoch + 1,
        #         split_name="val",
        #         device="cuda",
        #         max_batches=TSNE_MAX_BATCHES,
        #         film_step=TSNE_FILM_STEP,
        #         number_of_classes=dataset_num_classes,
        #         label_vector_length=TSNE_LABEL_VECTOR_LENGTH,
        #         show_progress=TSNE_SHOW_PROGRESS,
        #     )
        #     print(f"[tsne-save] epoch={epoch + 1}, saved_files=1")
        #     print(f"[tsne-save] {raw_npz_path}")
            # tsne_paths = run_tsne_visualization(
            #     feature_extractor=feature_extractor,
            #     data_loader=valid_loader1,
            #     output_dir=TSNE_OUTPUT_DIR,
            #     epoch_index=epoch + 1,
            #     split_name=TSNE_SPLIT_NAME,
            #     device="cuda",
            #     max_batches=TSNE_MAX_BATCHES,
            #     film_step=TSNE_FILM_STEP,
            #     number_of_classes=dataset_num_classes,
            #     label_vector_length=TSNE_LABEL_VECTOR_LENGTH,
            #     plot_sample_scatter=TSNE_PLOT_SAMPLE_SCATTER,
            #     show_progress=TSNE_SHOW_PROGRESS,
            # )
            # print(f"[tsne] epoch={epoch + 1}, saved_files={len(tsne_paths)}")
            # for tsne_path in tsne_paths:
            #     print(f"[tsne] {tsne_path}")

        print('-------------------------------accuracy-------------------------')
        with open('vit-b_OCT_amd_test.txt', 'a') as f:
            f.write(
                f"Epoch {epoch}, "
                f"kappa: {kappa_mean:.4f}+/-{kappa_std:.4f}, "
                f"f1: {f1_mean:.4f}+/-{f1_std:.4f}, "
                f"mae: {mae_mean:.4f}+/-{mae_std:.4f}, "
                f"sensitivity: {sens_mean:.4f}+/-{sens_std:.4f}, "
                f"specificity: {spec_mean:.4f}+/-{spec_std:.4f}, "
                f"accuracy: {acc_mean:.4f}+/-{acc_std:.4f}"
            )
            # if speed_stats is not None:
            #     f.write(
            #         f", pure_fps: {speed_stats['fps']:.2f}, "
            #         f"pure_avg_batch_ms: {speed_stats['avg_batch_ms']:.2f}, "
            #         f"pure_avg_sample_ms: {speed_stats['avg_sample_ms']:.3f}"
            #     )
            # f.write('\n')
        if f1_mean > best_f1:
            best_f1 = f1_mean
            torch.save(feature_extractor.state_dict(), './vit-b_OCT_amd_new2.pth')
            print(f"Saved new best model with f1: {best_f1:.4f}")
    print('best_f1',best_f1)

if __name__ == '__main__':
    main()
