import os, sys, json, math, random, gc, time, warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_curve,
                              auc, f1_score, accuracy_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve

from scipy import stats as scipy_stats

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  total_mem={props.total_memory/1e9:.2f}GB")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb, total_gb = free_bytes / 1e9, total_bytes / 1e9
    print(f"GPU free memory at startup: {free_gb:.2f} GB / {total_gb:.2f} GB")
    if free_gb < 0.5 * total_gb:
        raise RuntimeError(
            f""
        )

torch.backends.cudnn.benchmark = True
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


STAGE_TIMES = {}
def tic():
    return time.time()
def toc(t0, stage_name):
    dt = time.time() - t0
    STAGE_TIMES[stage_name] = STAGE_TIMES.get(stage_name, 0.0) + dt
    print(f"    [TIME] {stage_name}: {dt:.1f}s")
    return dt

GLOBAL_START = time.time()

def elapsed_min():
    return (time.time() - GLOBAL_START) / 60.0

def time_left_sec(budget_sec):
    return budget_sec - (time.time() - GLOBAL_START)

def time_up(budget_sec):
    return time_left_sec(budget_sec) <= 0

def stage_time_up(stage_start, stage_budget_sec):
    stage_elapsed = time.time() - stage_start
    return stage_elapsed >= stage_budget_sec or time_up(CFG["TIME_BUDGET_TOTAL_SEC"])

CFG = dict(
    DATA_ROOT   = "/kaggle/input/datasets/kneroma/tacotrashdataset/data",
    ANN_FILE    = "/kaggle/input/datasets/kneroma/tacotrashdataset/data/annotations.json",
    IMG_SIZE    = 512,

    BATCH_SIZE  = 4,
    ACCUM_STEPS = 2,
    NUM_WORKERS = 2,
    PERSISTENT_WORKERS = True,
    PREFETCH_FACTOR = 2,
    CACHE_DECODED_IMAGES = True,

    BOX_DETECTIONS_PER_IMG = 50,

    EPOCHS_DET  = 40,
    FREEZE_EPOCHS = 3,
    PATIENCE    = 7,
    MIN_DELTA   = 1e-3,
    EPOCHS_GNN  = 100,
    LR_HEAD     = 5e-4,
    LR_BACKBONE = 1e-5,
    LR_GNN      = 5e-4,
    VAL_SPLIT   = 0.15,
    TEST_SPLIT  = 0.15,
    N_SUPERCATS = 10,
    OUT_DIR     = "/kaggle/working",
    KNN_K       = 5,
    DUMP_MIN_OBJ= 4,
    DUMP_DIST   = 0.25,
    SCORE_THRESH= 0.3,
    VIS_SAMPLES = 12,
    GC_EVERY    = 200,

    RUN_TESTS         = True,
    RUN_MASK_AP       = True,
    RUN_BOOTSTRAP     = True,
    N_BOOTSTRAP       = 30,
    MIOU_MAX_IMAGES   = 100,
    RUN_BASELINES     = True,
    MAX_TSNE_SAMPLES  = 1500,


    TIME_BUDGET_TOTAL_SEC = 28 * 60,
    TIME_BUDGET_DET_SEC   = 16 * 60,
    TIME_BUDGET_GNN_SEC   = 7 * 60,
    TRAIN_GRAPH_INFER_IMAGES = 300,


    TIME_BUDGET_STANDALONE_GRAPH_SEC       = 3 * 60,
    TIME_BUDGET_STANDALONE_TRANSFORMER_SEC = 3 * 60,

    
    GRAPH_MIN_TEST_GRAPHS      = 60,
    GRAPH_FALLBACK_SCORE_THRESH = 0.15,

 
    GNN_HIDDEN        = 96,
    GNN_HEADS         = 6,
    GNN_LAYERS         = 3,
    GNN_WEIGHT_DECAY  = 3e-5,
    GNN_GRAD_CLIP     = 1.0,
    GNN_WARMUP_EPOCHS = 3,
    PATIENCE_GNN      = 8,
    MIN_DELTA_GNN     = 1e-3,
    GNN_BATCH_SIZE    = 32,

    
    SYNTHETIC_NOISE_FRAC = 0.20,
    SYNTHETIC_NOISE_SEED = 123,
)
os.makedirs(CFG["OUT_DIR"], exist_ok=True)
FIG_DIR = os.path.join(CFG["OUT_DIR"], "figures")
os.makedirs(FIG_DIR, exist_ok=True)
print(CFG)

def savefig(name):
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches="tight")


t0 = tic()
coco = COCO(CFG["ANN_FILE"])
cats = coco.loadCats(coco.getCatIds())
cat_df = pd.DataFrame(cats)
print(f"Fine categories: {len(cat_df)}")
print(cat_df[['id', 'name', 'supercategory']].head(10))

supercats_sorted = (cat_df.groupby("supercategory")["id"].apply(list).to_dict())
ann_all = coco.loadAnns(coco.getAnnIds())
super_counts = {}
catid2super = dict(zip(cat_df.id, cat_df.supercategory))
for a in ann_all:
    s = catid2super[a["category_id"]]
    super_counts[s] = super_counts.get(s, 0) + 1

ranked = sorted(super_counts.items(), key=lambda x: -x[1])
keep = [s for s, _ in ranked[:CFG["N_SUPERCATS"] - 1]]
label_names = ["background"] + keep + ["misc"]
name2idx = {n: i for i, n in enumerate(label_names)}

def super_to_label(supercat_name):
    return name2idx[supercat_name] if supercat_name in name2idx else name2idx["misc"]

catid2label = {cid: super_to_label(catid2super[cid]) for cid in cat_df.id}
NUM_CLASSES = len(label_names)
print("Final class set:", label_names, "-> NUM_CLASSES =", NUM_CLASSES)

def build_reduced_gt_coco(coco_full, catid2label_map, names, n_classes):
    cats_reduced = [{"id": i, "name": names[i], "supercategory": "waste"} for i in range(1, n_classes)]
    anns_reduced = []
    next_id = 1
    for a in coco_full.dataset["annotations"]:
        lbl = catid2label_map[a["category_id"]]
        if lbl == 0:
            continue
        new_ann = dict(a)
        new_ann["category_id"] = lbl
        new_ann["id"] = next_id
        next_id += 1
        anns_reduced.append(new_ann)
    reduced_dict = {"images": coco_full.dataset["images"], "annotations": anns_reduced, "categories": cats_reduced}
    coco_reduced = COCO()
    coco_reduced.dataset = reduced_dict
    coco_reduced.createIndex()
    return coco_reduced

coco_eval_gt = build_reduced_gt_coco(coco, catid2label, label_names, NUM_CLASSES)
print(f"Reduced-label-space GT built for eval: {len(coco_eval_gt.dataset['annotations'])} "
      f"annotations across {NUM_CLASSES - 1} foreground classes.")

class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
for a in ann_all:
    class_counts[catid2label[a["category_id"]]] += 1
print("Per-class annotation counts:", dict(zip(label_names, class_counts.tolist())))
toc(t0, "load_annotations")


img_ids_all = sorted(coco.getImgIds())
train_ids, temp_ids = train_test_split(img_ids_all, test_size=CFG["VAL_SPLIT"] + CFG["TEST_SPLIT"], random_state=SEED)
val_ids, test_ids   = train_test_split(temp_ids, test_size=CFG["TEST_SPLIT"] / (CFG["VAL_SPLIT"] + CFG["TEST_SPLIT"]), random_state=SEED)
print(f"train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")


def run_sanity_tests():
    print("\n=== Running pre-flight sanity tests ===")

    s_train, s_val, s_test = set(train_ids), set(val_ids), set(test_ids)
    assert not (s_train & s_val), "train/val split overlap!"
    assert not (s_train & s_test), "train/test split overlap!"
    assert not (s_val & s_test), "val/test split overlap!"
    assert len(s_train | s_val | s_test) == len(img_ids_all), \
        "train+val+test doesn't cover every image id — split logic is losing images."
    print(f"  [OK] splits disjoint & complete ({len(s_train)}/{len(s_val)}/{len(s_test)})")

    all_cat_ids = set(cat_df.id.tolist())
    assert set(catid2label.keys()) == all_cat_ids, "catid2label is missing some category ids."
    bad_labels = [v for v in catid2label.values() if not (0 <= v < NUM_CLASSES)]
    assert not bad_labels, f"catid2label produced out-of-range label(s): {bad_labels}"
    print(f"  [OK] category->label mapping covers all {len(all_cat_ids)} categories, in range [0,{NUM_CLASSES})")

    assert class_counts.sum() == len(ann_all), \
        f"class_counts sum ({class_counts.sum()}) != total annotations ({len(ann_all)}) — a filtering bug."
    print(f"  [OK] class_counts sum matches total annotation count ({int(class_counts.sum())})")

    sample_id = train_ids[0]
    sample_path = os.path.join(CFG["DATA_ROOT"], coco.loadImgs(sample_id)[0]["file_name"])
    img = cv2.imread(sample_path)
    assert img is not None, (
        f"Could not read a sample training image at {sample_path}. Check CFG['DATA_ROOT'] "
        f"points at the actual TACO image folder before training."
    )
    print(f"  [OK] sample image readable ({sample_path}, shape={img.shape})")

    zero_classes = [label_names[i] for i in range(1, NUM_CLASSES) if class_counts[i] == 0]
    assert not zero_classes, f"Foreground class(es) with zero annotations: {zero_classes}"
    print("  [OK] every foreground class has at least one annotation")

    print("=== All pre-flight sanity tests passed ===\n")

if CFG["RUN_TESTS"]:
    t0 = tic()
    run_sanity_tests()
    toc(t0, "sanity_tests")

class TACODataset(Dataset):
    def __init__(self, coco_obj, img_ids, img_size=512, train=False, cache_decoded=False):
        self.coco = coco_obj
        self.ids = img_ids
        self.img_size = img_size
        self.train = train
        self.cache_decoded = cache_decoded
        self._cache = {} if cache_decoded else None

    def __len__(self):
        return len(self.ids)

    def _decode_raw(self, img_id):
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(CFG["DATA_ROOT"], info["file_name"])
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            h0, w0 = self.img_size, self.img_size
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h0, w0 = img.shape[:2]

        scale = self.img_size / max(h0, w0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        img = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = img

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, masks = [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            x1, y1, x2, y2 = x * scale, y * scale, (x + w) * scale, (y + h) * scale
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(catid2label[a["category_id"]])
            m = self.coco.annToMask(a)
            m = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            m_canvas = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            m_canvas[:new_h, :new_w] = m
            masks.append(m_canvas)

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, self.img_size, self.img_size), dtype=torch.uint8)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8)

        return canvas, boxes_t, labels_t, masks_t, h0, w0, scale

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        if self.cache_decoded and img_id in self._cache:
            canvas_c, boxes_c, labels_c, masks_c, h0, w0, scale = self._cache[img_id]
            canvas = canvas_c.copy()
            boxes = boxes_c.clone()
            labels = labels_c.clone()
            masks = masks_c.clone()
        else:
            canvas, boxes, labels, masks, h0, w0, scale = self._decode_raw(img_id)
            if self.cache_decoded:
                self._cache[img_id] = (canvas.copy(), boxes.clone(), labels.clone(), masks.clone(), h0, w0, scale)

        if self.train and len(boxes) > 0:
            if random.random() < 0.5:
                canvas = canvas[:, ::-1, :].copy()
                flipped_x1 = self.img_size - boxes[:, 2]
                flipped_x2 = self.img_size - boxes[:, 0]
                boxes[:, 0] = flipped_x1
                boxes[:, 2] = flipped_x2
                masks = torch.flip(masks, dims=[2])
            if random.random() < 0.5:
                canvas = canvas.astype(np.float32)
                canvas *= random.uniform(0.85, 1.15)
                mean = canvas.mean(axis=(0, 1), keepdims=True)
                canvas = (canvas - mean) * random.uniform(0.85, 1.15) + mean
                canvas = np.clip(canvas, 0, 255).astype(np.uint8)

        img_t = torch.as_tensor(canvas.astype(np.float32) / 255.0, dtype=torch.float32).permute(2, 0, 1)

        target = {
            "boxes": boxes, "labels": labels, "masks": masks,
            "image_id": torch.tensor([img_id]),
            "orig_size": (h0, w0), "scale": scale,
        }
        return img_t, target

def collate_fn(batch):
    return tuple(zip(*batch))

train_ds = TACODataset(coco, train_ids, CFG["IMG_SIZE"], train=True,  cache_decoded=CFG["CACHE_DECODED_IMAGES"])
val_ds   = TACODataset(coco, val_ids,   CFG["IMG_SIZE"], train=False, cache_decoded=CFG["CACHE_DECODED_IMAGES"])
test_ds  = TACODataset(coco, test_ids,  CFG["IMG_SIZE"], train=False, cache_decoded=CFG["CACHE_DECODED_IMAGES"])

loader_kwargs = dict(num_workers=CFG["NUM_WORKERS"], collate_fn=collate_fn, pin_memory=(DEVICE.type == "cuda"))
if CFG["NUM_WORKERS"] > 0:
    loader_kwargs["persistent_workers"] = CFG["PERSISTENT_WORKERS"]
    loader_kwargs["prefetch_factor"] = CFG["PREFETCH_FACTOR"]

def compute_image_sample_weights(coco_obj, img_ids, catid2label_map, n_classes):
    counts = np.zeros(n_classes)
    per_image_classes = {}
    for img_id in img_ids:
        labels = set()
        for a in coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=img_id)):
            lbl = catid2label_map[a["category_id"]]
            if lbl == 0:
                continue
            labels.add(lbl)
            counts[lbl] += 1
        per_image_classes[img_id] = labels
    counts = np.clip(counts, 1, None)
    inv_freq = 1.0 / counts
    weights = np.array([
        max((inv_freq[l] for l in per_image_classes[img_id]), default=inv_freq.mean())
        for img_id in img_ids
    ], dtype=np.float64)
    return weights

train_sample_weights = compute_image_sample_weights(coco, train_ids, catid2label, NUM_CLASSES)
train_sampler = torch.utils.data.WeightedRandomSampler(
    weights=train_sample_weights, num_samples=len(train_ids), replacement=True)
print(f"Class-balanced train sampler built. Rarest-class images now sampled "
      f"~{(train_sample_weights.max() / train_sample_weights.min()):.1f}x more often than the most common.")

train_loader = DataLoader(train_ds, batch_size=CFG["BATCH_SIZE"], sampler=train_sampler, drop_last=True, **loader_kwargs)
val_loader   = DataLoader(val_ds, batch_size=CFG["BATCH_SIZE"], shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds, batch_size=CFG["BATCH_SIZE"], shuffle=False, **loader_kwargs)

if CFG["RUN_TESTS"]:
    t0 = tic()
    _imgs, _tgts = next(iter(train_loader))
    for im in _imgs:
        assert im.shape == (3, CFG["IMG_SIZE"], CFG["IMG_SIZE"]), f"bad image shape {im.shape}"
        assert torch.isfinite(im).all(), "non-finite values in a training image tensor"
        assert 0.0 <= im.min() and im.max() <= 1.0 + 1e-4, "image tensor out of [0,1] range"
    for t in _tgts:
        assert t["boxes"].shape[0] == t["labels"].shape[0] == t["masks"].shape[0], \
            "boxes/labels/masks count mismatch in a target"
        if t["boxes"].shape[0] > 0:
            assert (t["boxes"][:, 2] > t["boxes"][:, 0]).all() and (t["boxes"][:, 3] > t["boxes"][:, 1]).all(), \
                "degenerate (zero/negative area) box slipped through"
    print(f"  [OK] one real training batch passes shape/dtype/range checks (batch_size={len(_imgs)})")
    del _imgs, _tgts
    toc(t0, "dataset_sanity_batch")


def build_maskrcnn(num_classes, img_size):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=img_size,
        max_size=img_size,
        rpn_post_nms_top_n_test=300,
        box_detections_per_img=CFG["BOX_DETECTIONS_PER_IMG"],
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)
    for p in model.backbone.body.parameters():
        p.requires_grad = False
    return model

def set_backbone_trainable(model, mode):
    body = model.backbone.body
    for p in body.parameters():
        p.requires_grad = False
    if mode == "layer4":
        for p in body.layer4.parameters():
            p.requires_grad = True
    elif mode == "all":
        for p in body.parameters():
            p.requires_grad = True

def build_optimizer(model):
    backbone_params = [p for n, p in model.named_parameters() if "backbone.body" in n and p.requires_grad]
    other_params    = [p for n, p in model.named_parameters() if "backbone.body" not in n and p.requires_grad]
    groups = [{"params": other_params, "lr": CFG["LR_HEAD"]}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": CFG["LR_BACKBONE"]})
    return torch.optim.AdamW(groups, weight_decay=2e-4)

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

det_model = build_maskrcnn(NUM_CLASSES, CFG["IMG_SIZE"]).to(DEVICE)
print("Mask R-CNN ready (backbone frozen for phase 1). Trainable params:",
      sum(p.numel() for p in det_model.parameters() if p.requires_grad))

if CFG["RUN_TESTS"]:
    t0 = tic()
    det_model.eval()
    try:
        with torch.no_grad(), autocast(enabled=(DEVICE.type == "cuda")):
            dummy = [torch.rand(3, CFG["IMG_SIZE"], CFG["IMG_SIZE"], device=DEVICE)]
            out = det_model(dummy)
        assert isinstance(out, list) and "boxes" in out[0] and "masks" in out[0], \
            "Mask R-CNN forward pass didn't return the expected output structure."
        print(f"  [OK] model smoke test passed (dummy forward -> {len(out[0]['boxes'])} boxes)")
        del dummy, out
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        raise RuntimeError(
            "Model smoke test OOM'd even at CFG['IMG_SIZE']x CFG['IMG_SIZE']. Lower "
            "CFG['BATCH_SIZE'] and/or CFG['IMG_SIZE'] and retry."
        )
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    toc(t0, "model_smoke_test")


optimizer = build_optimizer(det_model)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["EPOCHS_DET"])
scaler = GradScaler()

history = {"train_loss": [], "val_loss": []}
best_val = float("inf")
patience_counter = 0
CKPT_PATH = os.path.join(CFG["OUT_DIR"], "maskrcnn_best.pth")
det_ckpt_saved = False

def move_targets(targets):
    return [{k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]

def run_epoch(loader, train=True):
    det_model.train()
    total_loss, n_batches, n_skipped = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for i, (images, targets) in enumerate(loader):
        try:
            images = [img.to(DEVICE, non_blocking=True) for img in images]
            targets = move_targets(targets)
            with autocast(enabled=(DEVICE.type == "cuda")):
                loss_dict = det_model(images, targets)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                print(f"    [WARN] non-finite loss at batch {i}; skipping (not stepping optimizer).")
                optimizer.zero_grad(set_to_none=True)
                continue
            if train:
                scaler.scale(loss / CFG["ACCUM_STEPS"]).backward()
                if (i + 1) % CFG["ACCUM_STEPS"] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            n_batches += 1
            del images, targets, loss_dict, loss
        except torch.cuda.OutOfMemoryError:
            n_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
            continue
        if i % CFG["GC_EVERY"] == 0:
            gc.collect(); torch.cuda.empty_cache()
    if n_skipped:
        print(f"    [OOM-guard] skipped {n_skipped} batch(es) this epoch.")
    if train and n_batches == 0:
        raise RuntimeError(
            "Every training batch failed this epoch (0 succeeded) — lower CFG['BATCH_SIZE'] "
            "and/or CFG['IMG_SIZE'], and restart the kernel before retrying."
        )
    return total_loss / max(n_batches, 1)

t0 = tic()
det_stage_start = time.time()
det_budget = CFG["TIME_BUDGET_DET_SEC"]
for epoch in range(CFG["EPOCHS_DET"]):
    if stage_time_up(det_stage_start, det_budget):
        print(f">>> Detector time budget ({det_budget/60:.1f} min) reached after {epoch} epoch(s); "
              f"stopping detector training and keeping the best checkpoint so far.")
        break

    if epoch == CFG["FREEZE_EPOCHS"]:
        set_backbone_trainable(det_model, "layer4")
        optimizer = build_optimizer(det_model)
        remaining = CFG["EPOCHS_DET"] - CFG["FREEZE_EPOCHS"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
        print(f">>> Unfroze backbone.layer4 at epoch {epoch+1}. "
              f"Trainable params now: {sum(p.numel() for p in det_model.parameters() if p.requires_grad)}")

    te = time.time()
    train_loss = run_epoch(train_loader, train=True)
    with torch.no_grad():
        val_loss = run_epoch(val_loader, train=False)
    scheduler.step()
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0.0
    print(f"[Detector] epoch {epoch+1}/{CFG['EPOCHS_DET']}  train_loss={train_loss:.4f}  "
          f"val_loss={val_loss:.4f}  peak_mem={peak_mem:.2f}GB  ({time.time()-te:.1f}s)  "
          f"[elapsed {elapsed_min():.1f} min]")
    if val_loss < best_val - CFG["MIN_DELTA"]:
        best_val = val_loss
        patience_counter = 0
        torch.save(det_model.state_dict(), CKPT_PATH)
        det_ckpt_saved = True
    else:
        patience_counter += 1
        if patience_counter >= CFG["PATIENCE"] and epoch >= CFG["FREEZE_EPOCHS"] + 3:
            print(f">>> Early stopping at epoch {epoch+1}: no val_loss improvement for {CFG['PATIENCE']} epochs.")
            break
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gc.collect(); torch.cuda.empty_cache()

if det_ckpt_saved:
    det_model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    print("Loaded best detector checkpoint.")
else:
    torch.save(det_model.state_dict(), CKPT_PATH)
    print("No checkpoint improvement was recorded — saved current weights as a fallback.")
toc(t0, "detector_training_total")


plt.figure(figsize=(7, 4))
plt.plot(history["train_loss"], label="train loss")
plt.plot(history["val_loss"], label="val loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Mask R-CNN training curves")
plt.legend(); plt.grid(alpha=0.3)
savefig("detector_curves.png"); plt.show()


@torch.no_grad()
def run_inference(loader, model, score_thresh=0.5, max_images=None,
                   compute_miou=False, miou_max_images=150):
    model.eval()
    bbox_results, segm_results = [], []
    per_image_dets = {}
    ious = []
    n_seen, n_miou_seen = 0, 0
    for images, targets in loader:
        if max_images is not None and n_seen >= max_images:
            break
        images_dev = [img.to(DEVICE, non_blocking=True) for img in images]
        with autocast(enabled=(DEVICE.type == "cuda")):
            outputs = model(images_dev)
        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())
            scale = tgt["scale"]
            keep = out["scores"] >= score_thresh
            boxes = out["boxes"][keep].cpu().numpy() / scale
            labels = out["labels"][keep].cpu().numpy()
            scores = out["scores"][keep].cpu().numpy()
            masks_full = out["masks"][keep].cpu().numpy() if len(out["masks"]) else None
            dets = []
            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes[j]
                w, h = x2 - x1, y2 - y1
                bbox_results.append({
                    "image_id": img_id, "category_id": int(labels[j]),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(scores[j]),
                })
                if masks_full is not None:
                    m = (masks_full[j, 0] > 0.5).astype(np.uint8)
                    orig_h, orig_w = tgt["orig_size"]
                    m_resized = cv2.resize(m, (int(m.shape[1] / scale), int(m.shape[0] / scale)),
                                            interpolation=cv2.INTER_NEAREST)
                    canvas = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    hh, ww = min(orig_h, m_resized.shape[0]), min(orig_w, m_resized.shape[1])
                    canvas[:hh, :ww] = m_resized[:hh, :ww]
                    rle = maskUtils.encode(np.asfortranarray(canvas))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    segm_results.append({
                        "image_id": img_id, "category_id": int(labels[j]),
                        "segmentation": rle, "score": float(scores[j]),
                    })
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                dets.append({"box": [x1, y1, x2, y2], "centroid": (cx, cy),
                             "label": int(labels[j]), "score": float(scores[j])})
            per_image_dets[img_id] = dets
            n_seen += 1

            if compute_miou and n_miou_seen < miou_max_images:
                pred_masks = (masks_full[:, 0] > 0.5) if masks_full is not None and len(masks_full) else np.zeros((0, CFG["IMG_SIZE"], CFG["IMG_SIZE"]), dtype=bool)
                gt_masks = tgt["masks"].numpy()
                if len(pred_masks) > 0 and len(gt_masks) > 0:
                    pred_union = np.any(pred_masks, axis=0)
                    gt_union = np.any(gt_masks.astype(bool), axis=0)
                    inter = np.logical_and(pred_union, gt_union).sum()
                    union = np.logical_or(pred_union, gt_union).sum()
                    if union > 0:
                        ious.append(inter / union)
                n_miou_seen += 1
        del images_dev, outputs
        gc.collect(); torch.cuda.empty_cache()
    return bbox_results, segm_results, per_image_dets, (float(np.mean(ious)) if ious else 0.0)

t0 = tic()
det_results, det_segm_results, test_dets, mask_miou = run_inference(
    test_loader, det_model, score_thresh=CFG["SCORE_THRESH"],
    compute_miou=True, miou_max_images=CFG["MIOU_MAX_IMAGES"])
toc(t0, "inference_pass")

def coco_eval_for_ids(coco_gt, results, iou_type, img_ids, res_path):
    if len(results) == 0:
        return None, [0.0] * 12
    with open(res_path, "w") as f:
        json.dump(results, f)
    coco_dt = coco_gt.loadRes(res_path)
    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev, ev.stats.tolist()

bbox_res_path = os.path.join(CFG["OUT_DIR"], "det_results_bbox.json")
ev_bbox, stats_bbox = coco_eval_for_ids(coco_eval_gt, det_results, "bbox", test_ids, bbox_res_path)
mAP_50_95, mAP_50 = (stats_bbox[0], stats_bbox[1]) if ev_bbox else (0.0, 0.0)
print(f"\n>>> DETECTION (bbox): mAP@[.5:.95] = {mAP_50_95:.3f} | mAP@0.5 = {mAP_50:.3f}")

segm_mAP_50_95, segm_mAP_50 = 0.0, 0.0
if CFG["RUN_MASK_AP"]:
    segm_res_path = os.path.join(CFG["OUT_DIR"], "det_results_segm.json")
    ev_segm, stats_segm = coco_eval_for_ids(coco_eval_gt, det_segm_results, "segm", test_ids, segm_res_path)
    segm_mAP_50_95, segm_mAP_50 = (stats_segm[0], stats_segm[1]) if ev_segm else (0.0, 0.0)
    print(f">>> SEGMENTATION (segm, standard COCO mask AP): mAP@[.5:.95] = {segm_mAP_50_95:.3f} | mAP@0.5 = {segm_mAP_50:.3f}")

print(f">>> Mask mIoU (secondary diagnostic, image-level union, test): {mask_miou:.3f} "
      f"— use SEGM mAP above as the reportable headline segmentation metric.")

def per_class_ap(ev, names, n_classes):
    if ev is None:
        return {}
    precisions = ev.eval["precision"]
    out = {}
    for k in range(n_classes - 1):
        p = precisions[:, :, k, 0, -1]
        p = p[p > -1]
        out[names[k + 1]] = float(p.mean()) if p.size else float("nan")
    return out

per_class = per_class_ap(ev_bbox, label_names, NUM_CLASSES)
if per_class:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(per_class.keys()), list(per_class.values()), color="#55A868")
    ax.set_ylabel("AP@[.5:.95]"); ax.set_title("Per-class detection AP (bbox)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    savefig("per_class_ap.png"); plt.show()

def bootstrap_map_ci(coco_gt, results, iou_type, img_ids, n_boot, res_path):
    if not CFG["RUN_BOOTSTRAP"] or len(results) == 0:
        return None
    with open(res_path, "w") as f:
        json.dump(results, f)
    coco_dt = coco_gt.loadRes(res_path)
    vals = []
    for _ in range(n_boot):
        sample_ids = list(np.random.choice(img_ids, size=len(img_ids), replace=True))
        try:
            ev_b = COCOeval(coco_gt, coco_dt, iouType=iou_type)
            ev_b.params.imgIds = list(set(sample_ids))
            ev_b.evaluate(); ev_b.accumulate(); ev_b.summarize()
            vals.append(ev_b.stats[1])
        except Exception:
            continue
    if not vals:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)

t0 = tic()
map_ci = bootstrap_map_ci(coco_eval_gt, det_results, "bbox", test_ids, CFG["N_BOOTSTRAP"], bbox_res_path)
if map_ci:
    print(f">>> Bootstrap mAP@0.5: mean={map_ci[0]:.3f}  95% CI=[{map_ci[1]:.3f}, {map_ci[2]:.3f}] (n_boot={CFG['N_BOOTSTRAP']})")
toc(t0, "map_bootstrap_ci")

if len(det_results) == 0:
    print("No detections above threshold — consider lowering SCORE_THRESH or training longer.")


train_dets_for_graphs = {}
if not time_up(CFG["TIME_BUDGET_TOTAL_SEC"]):
    t0 = tic()
    n_train_infer = min(CFG["TRAIN_GRAPH_INFER_IMAGES"], len(train_ids))
    train_subset_ids = train_ids[:n_train_infer]
    train_infer_ds = TACODataset(coco, train_subset_ids, CFG["IMG_SIZE"], train=False, cache_decoded=False)
    train_infer_loader = DataLoader(train_infer_ds, batch_size=CFG["BATCH_SIZE"], shuffle=False, **loader_kwargs)
    _, _, train_dets_for_graphs, _ = run_inference(
        train_infer_loader, det_model, score_thresh=CFG["SCORE_THRESH"])
    toc(t0, "train_inference_for_noisy_graphs")
    print(f"Ran detector inference on {len(train_dets_for_graphs)} training images "
          f"for noisy-label graph construction.")
else:
    print("  [skip] time budget already spent — skipping noisy training-graph inference "
          "(GNN stage will fall back to clean GT-based graphs only).")

def box_iou_np(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_b + area_boxes - inter
    return inter / np.clip(union, 1e-6, None)

def match_detections_to_gt(dets, gt_dets, iou_thresh=0.5):
    if len(gt_dets) == 0 or len(dets) == 0:
        return [-1] * len(dets)
    gt_boxes = np.array([d["box"] for d in gt_dets], dtype=np.float32)
    gt_labels = np.array([d["label"] for d in gt_dets], dtype=np.int64)
    matched = []
    for d in dets:
        ious = box_iou_np(np.array(d["box"], dtype=np.float32), gt_boxes)
        best = int(np.argmax(ious))
        matched.append(int(gt_labels[best]) if ious[best] >= iou_thresh else -1)
    return matched

def build_graph_for_image(dets, img_diag, min_obj, dist_thresh, true_labels=None):
    n = len(dets)
    if n == 0:
        return None
    label_onehot = np.eye(NUM_CLASSES, dtype=np.float32)
    feats = np.array([
        np.concatenate([
            [d["centroid"][0] / img_diag, d["centroid"][1] / img_diag,
             (d["box"][2]-d["box"][0]) / img_diag, (d["box"][3]-d["box"][1]) / img_diag,
             d["score"]],
            label_onehot[d["label"]]
        ]) for d in dets], dtype=np.float32)
    centroids = np.array([d["centroid"] for d in dets], dtype=np.float32)
    dmat = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1) / img_diag

    A = np.zeros((n, n), dtype=np.float32)
    k = min(CFG["KNN_K"], n - 1) if n > 1 else 0
    for i in range(n):
        order = np.argsort(dmat[i])
        neigh = order[1:k+1] if k > 0 else []
        for j in neigh:
            A[i, j] = 1.0; A[j, i] = 1.0

    radius_edges = (dmat < dist_thresh).astype(np.float32)
    np.fill_diagonal(radius_edges, 0.0)
    A = np.maximum(A, radius_edges)

    dense_flags = (dmat < dist_thresh).sum(axis=1) - 1
    is_dense_cluster = (dense_flags >= (min_obj - 1)).any()
    graph_label = int(is_dense_cluster and n >= min_obj)
    node_labels = (np.array([d["label"] for d in dets], dtype=np.int64) if true_labels is None
                   else np.array(true_labels, dtype=np.int64))

    max_density = float(dense_flags.max()) if n > 0 else 0.0
    min_dist = float(dmat[~np.eye(n, dtype=bool)].min()) if n > 1 else 1.0
    extra = np.array([max_density, min_dist, float(n)], dtype=np.float32)

    return {"X": feats, "A": A, "D": dmat.astype(np.float32), "n_nodes": n,
            "labels": node_labels, "graph_label": graph_label, "extra": extra}

def get_img_diag(img_id):
    info = coco.loadImgs(img_id)[0]
    return math.sqrt(info["height"] ** 2 + info["width"] ** 2)

def get_gt_dets(img_id):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    dets = []
    for a in coco.loadAnns(ann_ids):
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            continue
        lbl = catid2label[a["category_id"]]
        if lbl == 0:
            continue
        dets.append({"box": [x, y, x + w, y + h], "centroid": (x + w / 2, y + h / 2),
                     "label": lbl, "score": 1.0})
    return dets

gt_dets_by_img = {img_id: get_gt_dets(img_id) for img_id in train_ids + val_ids}
gt_diag_by_img = {img_id: get_img_diag(img_id) for img_id in train_ids + val_ids}

best_cfg, best_gap, best_rate = (CFG["DUMP_MIN_OBJ"], CFG["DUMP_DIST"]), 1.0, 0.0
for min_obj in [2, 3, 4, 5]:
    for dist_thresh in [0.15, 0.20, 0.25, 0.30, 0.35]:
        labels = []
        for img_id, dets in gt_dets_by_img.items():
            if len(dets) == 0:
                continue
            diag = gt_diag_by_img[img_id]
            centroids = np.array([d["centroid"] for d in dets], dtype=np.float32)
            dmat = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1) / diag
            dense_flags = (dmat < dist_thresh).sum(axis=1) - 1
            labels.append(int((dense_flags >= (min_obj - 1)).any() and len(dets) >= min_obj))
        rate = np.mean(labels) if labels else 0.0
        gap = abs(rate - 0.35)
        if gap < best_gap:
            best_gap, best_cfg, best_rate = gap, (min_obj, dist_thresh), rate

CFG["DUMP_MIN_OBJ"], CFG["DUMP_DIST"] = best_cfg
print(f"Calibrated dumping-proxy threshold: DUMP_MIN_OBJ={best_cfg[0]}, DUMP_DIST={best_cfg[1]} "
      f"-> positive rate ~{best_rate:.1%} on train/val GT graphs (target was 35%).")

train_graphs = []
for img_id, dets in gt_dets_by_img.items():
    if len(dets) == 0:
        continue
    g = build_graph_for_image(dets, gt_diag_by_img[img_id], CFG["DUMP_MIN_OBJ"], CFG["DUMP_DIST"])
    if g is not None:
        train_graphs.append(g)

test_graphs_gt = []
for img_id in test_ids:
    dets = get_gt_dets(img_id)
    if len(dets) == 0:
        continue
    g = build_graph_for_image(dets, get_img_diag(img_id), CFG["DUMP_MIN_OBJ"], CFG["DUMP_DIST"])
    if g is not None:
        test_graphs_gt.append(g)

def build_synthetic_noise_test_graphs(img_ids, noise_frac, seed):
    rng = np.random.RandomState(seed)
    possible_labels = list(range(1, NUM_CLASSES))
    out = []
    for img_id in img_ids:
        dets = get_gt_dets(img_id)
        if len(dets) < 2:
            continue
        true_labels = [d["label"] for d in dets]
        noisy_labels = list(true_labels)
        n_corrupt = max(1, int(round(len(dets) * noise_frac)))
        n_corrupt = min(n_corrupt, len(dets))
        corrupt_idx = rng.choice(len(dets), size=n_corrupt, replace=False)
        for idx in corrupt_idx:
            choices = [l for l in possible_labels if l != true_labels[idx]]
            noisy_labels[idx] = int(rng.choice(choices))
        dets_noisy_input = []
        for d, nl in zip(dets, noisy_labels):
            d2 = dict(d)
            d2["label"] = nl
            dets_noisy_input.append(d2)
        g = build_graph_for_image(dets_noisy_input, get_img_diag(img_id),
                                   CFG["DUMP_MIN_OBJ"], CFG["DUMP_DIST"],
                                   true_labels=true_labels)
        if g is not None:
            out.append(g)
    return out

test_graphs_synthetic_noise = build_synthetic_noise_test_graphs(
    test_ids, CFG["SYNTHETIC_NOISE_FRAC"], CFG["SYNTHETIC_NOISE_SEED"])
print(f"Built {len(test_graphs_synthetic_noise)} SYNTHETIC-noise diagnostic graphs "
      f"({CFG['SYNTHETIC_NOISE_FRAC']:.0%} of node input labels deliberately corrupted, GT boxes, "
      f"detector-independent) — isolates context-based label-correction ability.")

def build_test_graphs(test_dets_dict):
    out = []
    for img_id, dets in test_dets_dict.items():
        if len(dets) == 0:
            continue
        true_labels = match_detections_to_gt(dets, get_gt_dets(img_id), iou_thresh=0.5)
        g = build_graph_for_image(dets, get_img_diag(img_id), CFG["DUMP_MIN_OBJ"], CFG["DUMP_DIST"],
                                   true_labels=true_labels)
        if g is not None:
            out.append(g)
    return out

graphs = build_test_graphs(test_dets)

if len(graphs) < CFG["GRAPH_MIN_TEST_GRAPHS"] and CFG["GRAPH_FALLBACK_SCORE_THRESH"] < CFG["SCORE_THRESH"]:
    print(f"  [adaptive] only {len(graphs)} detector-based test graphs at SCORE_THRESH="
          f"{CFG['SCORE_THRESH']} (target >= {CFG['GRAPH_MIN_TEST_GRAPHS']} for a stable "
          f"real-pipeline evaluation). Rebuilding GRAPH-CONSTRUCTION detections at a lower "
          f"threshold ({CFG['GRAPH_FALLBACK_SCORE_THRESH']}) — the mAP numbers reported above "
          f"are unaffected, they already used SCORE_THRESH={CFG['SCORE_THRESH']}.")
    t0 = tic()
    _, _, test_dets, _ = run_inference(test_loader, det_model, score_thresh=CFG["GRAPH_FALLBACK_SCORE_THRESH"])
    toc(t0, "test_inference_lower_thresh_for_graphs")
    graphs = build_test_graphs(test_dets)
    print(f"  [adaptive] rebuilt {len(graphs)} detector-based test graphs at the lower threshold "
          f"(qualitative visualization in Cell 12 will also reflect this threshold).")

noisy_train_graphs = []
for img_id, dets in train_dets_for_graphs.items():
    if len(dets) == 0:
        continue
    true_labels = match_detections_to_gt(dets, gt_dets_by_img.get(img_id, get_gt_dets(img_id)), iou_thresh=0.5)
    g = build_graph_for_image(dets, get_img_diag(img_id), CFG["DUMP_MIN_OBJ"], CFG["DUMP_DIST"],
                               true_labels=true_labels)
    if g is not None:
        noisy_train_graphs.append(g)
if noisy_train_graphs:
    n_nodes_noisy = sum(g["n_nodes"] for g in noisy_train_graphs)
    n_matched = sum(int((np.asarray(g["labels"]) != -1).sum()) for g in noisy_train_graphs)
    n_disagree = sum(int(((np.asarray(g["labels"]) != -1) &
                           (np.asarray(g["labels"]) != np.asarray(g["X"][:, 5:].argmax(axis=1)))).sum())
                      for g in noisy_train_graphs)
    print(f"Built {len(noisy_train_graphs)} NOISY detector-based training graphs "
          f"({n_matched}/{n_nodes_noisy} detections matched to a real GT box at IoU>=0.5, "
          f"of which {n_disagree} disagree with the detector's own predicted label — those "
          f"are the genuine 'use context to correct a mistake' training examples).")
else:
    print("  [skip] no noisy training graphs available (train-set inference produced nothing "
          "— GNN will fall back to training on clean GT-based graphs only).")

print(f"Built {len(train_graphs)} GT-based train/val graphs, "
      f"{len(test_graphs_gt)} GT-based test graphs (trivial ceiling), "
      f"{len(graphs)} detector-based test graphs (real pipeline), "
      f"{len(test_graphs_synthetic_noise)} synthetic-noise diagnostic graphs.")
pos_rate_train = np.mean([g["graph_label"] for g in train_graphs]) if train_graphs else 0
pos_rate_test  = np.mean([g["graph_label"] for g in graphs]) if graphs else 0
print(f"Proxy 'dumping' positive rate — train/val: {pos_rate_train:.1%} | test (detector-based): {pos_rate_test:.1%}")
if abs(pos_rate_test - pos_rate_train) > 0.15:
    print(f"  [caution] train/test proxy-label positive-rate gap is {abs(pos_rate_test-pos_rate_train):.1%} "
          f"— this reflects a real train/test shift driven by detector quality on THIS run, not a "
          f"property of the proxy rule itself. Report both rates alongside any accuracy/F1 number.")
print("(Reminder: this is a DECLARED PROXY label built from object-clustering density, "
      "not verified real-world dumping-site ground truth — TACO has none.)")

if CFG["RUN_TESTS"]:
    assert len(train_graphs) > 0, "No training graphs were built — check DUMP_MIN_OBJ/DUMP_DIST calibration."
    if len(graphs) == 0:
        print("  [WARN] No detector-based test graphs were built even after the adaptive fallback — "
              "every test image had zero kept detections. Downstream 'REAL pipeline' proxy-dumping "
              "metrics will be skipped; bbox/segm mAP above is unaffected.")
    else:
        print(f"  [OK] graph-building sanity check passed ({len(train_graphs)} train / {len(graphs)} test graphs)")


class GraphormerLayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, n_dist_bins=8, max_dist=1.5, use_structural_bias=True):
        super().__init__()
        assert dim % heads == 0, "hidden dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.use_structural_bias = use_structural_bias
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.n_dist_bins = n_dist_bins
        if use_structural_bias:
            self.register_buffer("dist_bin_edges", torch.linspace(0, max_dist, n_dist_bins - 1))
            self.edge_bias = nn.Parameter(torch.zeros(heads, 2))
            self.dist_bias = nn.Parameter(torch.zeros(heads, n_dist_bins))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def _structural_bias(self, A, D):
        edge_idx = A.long().clamp(0, 1)
        edge_b = self.edge_bias[:, edge_idx].permute(1, 0, 2, 3)
        bin_idx = torch.bucketize(D, self.dist_bin_edges).clamp(0, self.n_dist_bins - 1)
        dist_b = self.dist_bias[:, bin_idx].permute(1, 0, 2, 3)
        return edge_b + dist_b

    def forward(self, X, A, D, key_padding_mask):
        B, N, _ = X.shape
        h = self.norm1(X)
        qkv = self.qkv(h).view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_logits = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)

        if self.use_structural_bias:
            attn_logits = attn_logits + self._structural_bias(A, D)
        key_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  
        attn_logits = attn_logits.masked_fill(~key_mask, float("-inf"))
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        out = self.out_proj(out)
        X = X + self.dropout(out)
        X = X + self.dropout(self.ffn(self.norm2(X)))
        return X

class GCNLayer(nn.Module):

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, X, A, D, key_padding_mask):
     
        B, N, _ = X.shape
        mask2d = (key_padding_mask.unsqueeze(1) & key_padding_mask.unsqueeze(2)).float() 
        A_self = (A + torch.eye(N, device=A.device).unsqueeze(0)) * mask2d
        deg = A_self.sum(dim=-1, keepdim=True).clamp(min=1.0)
        A_norm = A_self / deg
        h = self.norm1(X)
        agg = torch.bmm(A_norm, self.lin(h))
        X = X + self.dropout(agg)
        X = X + self.dropout(self.ffn(self.norm2(X)))
        return X

class ContextGraphTransformer(nn.Module):
    """Shared architecture shell used by all three models below: an input
    projection, a stack of mixing layers (Graphormer / plain-Transformer /
    GCN, selected via `layer_type`), a per-node classification head, and an
    attention-pooled graph-level head augmented with the v5 engineered
    scalar features."""
    def __init__(self, in_dim, hidden=64, heads=4, n_layers=2, num_classes=NUM_CLASSES,
                 n_dist_bins=8, dropout=0.1, extra_dim=3, layer_type="graphormer"):
        super().__init__()
        self.layer_type = layer_type
        self.input_proj = nn.Linear(in_dim, hidden)
        if layer_type == "graphormer":
            self.layers = nn.ModuleList([
                GraphormerLayer(hidden, heads=heads, dropout=dropout, n_dist_bins=n_dist_bins,
                                 use_structural_bias=True) for _ in range(n_layers)])
        elif layer_type == "transformer_no_structure":
            self.layers = nn.ModuleList([
                GraphormerLayer(hidden, heads=heads, dropout=dropout, n_dist_bins=n_dist_bins,
                                 use_structural_bias=False) for _ in range(n_layers)])
        elif layer_type == "gcn":
            self.layers = nn.ModuleList([GCNLayer(hidden, dropout=dropout) for _ in range(n_layers)])
        else:
            raise ValueError(f"unknown layer_type {layer_type}")
        self.node_head = nn.Linear(hidden, num_classes)
        self.pool_attn = nn.Linear(hidden, 1)
        self.extra_norm = nn.LayerNorm(extra_dim)
        self.graph_head = nn.Sequential(nn.Linear(hidden + extra_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, X, A, D, mask, extra, return_embedding=False):
        h = self.input_proj(X)
        for layer in self.layers:
            h = layer(h, A, D, mask)
        node_logits = self.node_head(h)
        pool_logits = self.pool_attn(h).squeeze(-1)
        pool_logits = pool_logits.masked_fill(~mask, float("-inf"))
        pool_w = torch.softmax(pool_logits, dim=1).unsqueeze(-1)
        graph_repr = (pool_w * h).sum(dim=1)
        extra_n = self.extra_norm(extra)
        graph_logit = self.graph_head(torch.cat([graph_repr, extra_n], dim=-1))
        if return_embedding:
            return node_logits, graph_logit, h
        return node_logits, graph_logit

def collate_graphs(batch_graphs, device):
    B = len(batch_graphs)
    max_n = max(g["n_nodes"] for g in batch_graphs)
    in_dim = batch_graphs[0]["X"].shape[1]
    Xb = np.zeros((B, max_n, in_dim), dtype=np.float32)
    Ab = np.zeros((B, max_n, max_n), dtype=np.float32)
    Db = np.zeros((B, max_n, max_n), dtype=np.float32)
    labels_b = np.full((B, max_n), -1, dtype=np.int64)
    mask_b = np.zeros((B, max_n), dtype=bool)
    graph_labels = np.zeros((B,), dtype=np.float32)
    extra_b = np.zeros((B, 3), dtype=np.float32)
    for i, g in enumerate(batch_graphs):
        n = g["n_nodes"]
        Xb[i, :n] = g["X"]
        Ab[i, :n, :n] = g["A"]
        Db[i, :n, :n] = g["D"]
        labels_b[i, :n] = g["labels"]
        mask_b[i, :n] = True
        graph_labels[i] = g["graph_label"]
        ex = g.get("extra", np.array([0.0, 1.0, float(n)], dtype=np.float32))
        extra_b[i] = [math.log1p(float(ex[0])), float(min(max(ex[1], 0.0), 1.0)),
                      float(min(ex[2] / 50.0, 2.0))]
    return (torch.tensor(Xb, device=device), torch.tensor(Ab, device=device),
            torch.tensor(Db, device=device), torch.tensor(labels_b, device=device),
            torch.tensor(mask_b, device=device), torch.tensor(graph_labels, device=device),
            torch.tensor(extra_b, device=device))

def graph_to_batch(g, device):
    """Single-graph convenience wrapper for the evaluation loops in Cell 11/11b."""
    return collate_graphs([g], device)

def iterate_batches(graph_list, batch_size, shuffle=True):
    idxs = list(range(len(graph_list)))
    if shuffle:
        random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        chunk = [graph_list[j] for j in idxs[i:i + batch_size]]
        yield chunk

GNN_DEVICE = DEVICE
torch.set_num_threads(max(1, os.cpu_count() or 1))
print(f"GNN stage device: {GNN_DEVICE}  (batch size {CFG['GNN_BATCH_SIZE']})")

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

IN_DIM = 5 + NUM_CLASSES

present_classes = np.where(class_counts > 0)[0]
cw = compute_class_weight(class_weight="balanced", classes=present_classes,
                           y=np.repeat(present_classes, class_counts[present_classes]))
class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
for c, w in zip(present_classes, cw):
    class_weights[c] = w
class_weights = class_weights.to(GNN_DEVICE)

n_pos = sum(g["graph_label"] for g in train_graphs)
n_neg = len(train_graphs) - n_pos
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=GNN_DEVICE)

train_graphs_split, val_graphs_split = train_test_split(train_graphs, test_size=0.2, random_state=SEED) if len(train_graphs) > 5 else (train_graphs, train_graphs)
n_clean_train = len(train_graphs_split)
train_graphs_split = train_graphs_split + noisy_train_graphs
print(f"GNN training pool: {n_clean_train} clean GT graphs + {len(noisy_train_graphs)} noisy "
      f"detector graphs = {len(train_graphs_split)} total.")
print("  [design note] model selection below uses the held-out VALIDATION split (never the "
      "test set) for ALL THREE models — this mirrors the detector's own best-checkpoint logic "
      "and keeps the final test-set numbers honest.")


def train_context_model(layer_type, tag, time_budget_sec, epochs, ckpt_name):
    """Generic training loop shared by the main model and both ablation
    baselines -- identical losses / optimizer recipe / early-stopping logic,
    only `layer_type` (and therefore the mixing mechanism) differs."""
    model = ContextGraphTransformer(
        in_dim=IN_DIM, hidden=CFG["GNN_HIDDEN"], heads=CFG["GNN_HEADS"], n_layers=CFG["GNN_LAYERS"],
        num_classes=NUM_CLASSES, extra_dim=3, layer_type=layer_type,
    ).to(GNN_DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["LR_GNN"], weight_decay=CFG["GNN_WEIGHT_DECAY"])

    def lr_lambda(epoch):
        warmup = CFG["GNN_WARMUP_EPOCHS"]
        total = max(epochs, warmup + 1)
        if epoch < warmup:
            return (epoch + 1) / warmup
        prog = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    hist = {"loss": [], "val_acc": [], "val_auc": []}
    ckpt_path = os.path.join(CFG["OUT_DIR"], ckpt_name)
    best_score, ckpt_saved, patience_counter = -float("inf"), False, 0
    GNN_BATCH = CFG["GNN_BATCH_SIZE"]

    t0_local = tic()
    stage_start = time.time()
    for epoch in range(epochs):
        if stage_time_up(stage_start, time_budget_sec):
            print(f">>> [{tag}] time budget ({time_budget_sec/60:.1f} min) reached after {epoch} epoch(s); stopping.")
            break

        model.train()
        epoch_loss, n_loss_terms = 0.0, 0
        for batch in iterate_batches(train_graphs_split, GNN_BATCH, shuffle=True):
            Xb, Ab, Db, yb_node, mask_b, yb_graph, extra_b = collate_graphs(batch, GNN_DEVICE)
            opt.zero_grad()
            node_logits, graph_logit = model(Xb, Ab, Db, mask_b, extra_b)
            loss_node = F.cross_entropy(
                node_logits.reshape(-1, NUM_CLASSES), yb_node.reshape(-1),
                weight=class_weights, ignore_index=-1)
            loss_graph = F.binary_cross_entropy_with_logits(
                graph_logit.view(-1), yb_graph, pos_weight=pos_weight)
            loss = loss_node + loss_graph
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["GNN_GRAD_CLIP"])
            opt.step()
            epoch_loss += loss.item()
            n_loss_terms += 1
        sched.step()

        model.eval()
        correct, total = 0, 0
        val_probs, val_true = [], []
        with torch.no_grad():
            for batch in iterate_batches(val_graphs_split, GNN_BATCH, shuffle=False):
                Xb, Ab, Db, yb_node, mask_b, yb_graph, extra_b = collate_graphs(batch, GNN_DEVICE)
                _, graph_logit = model(Xb, Ab, Db, mask_b, extra_b)
                probs = torch.sigmoid(graph_logit).view(-1)
                preds = (probs > 0.5).float()
                correct += int((preds == yb_graph).sum().item())
                total += yb_graph.numel()
                val_probs.extend(probs.cpu().tolist())
                val_true.extend(yb_graph.cpu().int().tolist())
        val_acc = correct / max(total, 1)
        if len(set(val_true)) > 1:
            fpr_v, tpr_v, _ = roc_curve(val_true, val_probs)
            val_auc = auc(fpr_v, tpr_v)
        else:
            val_auc = float("nan")
        hist["loss"].append(epoch_loss / max(n_loss_terms, 1))
        hist["val_acc"].append(val_acc)
        hist["val_auc"].append(val_auc)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[{tag}] epoch {epoch+1}/{epochs}  loss={hist['loss'][-1]:.4f}  "
                  f"val_graph_acc={val_acc:.3f}  val_graph_auc={val_auc if not math.isnan(val_auc) else float('nan'):.3f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

        score = val_auc if not math.isnan(val_auc) else val_acc
        if score > best_score + CFG["MIN_DELTA_GNN"]:
            best_score = score
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            ckpt_saved = True
        else:
            patience_counter += 1
            if patience_counter >= CFG["PATIENCE_GNN"] and epoch >= CFG["GNN_WARMUP_EPOCHS"] + 5:
                print(f">>> [{tag}] early stopping at epoch {epoch+1}: no val-score improvement for "
                      f"{CFG['PATIENCE_GNN']} epochs (best={best_score:.3f}).")
                break

    n_epochs_run = len(hist["loss"])
    if ckpt_saved:
        model.load_state_dict(torch.load(ckpt_path, map_location=GNN_DEVICE))
        print(f"[{tag}] loaded BEST checkpoint (val score={best_score:.3f}, from {n_epochs_run} epoch(s) run).")
    else:
        torch.save(model.state_dict(), ckpt_path)
        print(f"[{tag}] no checkpoint improvement recorded after {n_epochs_run} epoch(s) — saved fallback weights.")
    if n_epochs_run < 10:
        print(f"  [caution] [{tag}] only completed {n_epochs_run} epoch(s) before its time budget ran out. "
              f"Treat downstream metrics as UNDERTRAINED until re-run with more wall-clock time.")
    toc(t0_local, f"{tag}_training_total")
    return model, hist, n_epochs_run


gnn, gnn_history, n_gnn_epochs_run = train_context_model(
    layer_type="graphormer", tag="GNN(Graphormer)",
    time_budget_sec=CFG["TIME_BUDGET_GNN_SEC"], epochs=CFG["EPOCHS_GNN"], ckpt_name="gnn_best.pth")

standalone_gcn, gcn_history, n_gcn_epochs_run = train_context_model(
    layer_type="gcn", tag="StandaloneGCN",
    time_budget_sec=CFG["TIME_BUDGET_STANDALONE_GRAPH_SEC"], epochs=CFG["EPOCHS_GNN"], ckpt_name="standalone_gcn_best.pth")

standalone_transformer, transformer_history, n_transformer_epochs_run = train_context_model(
    layer_type="transformer_no_structure", tag="StandaloneTransformer",
    time_budget_sec=CFG["TIME_BUDGET_STANDALONE_TRANSFORMER_SEC"], epochs=CFG["EPOCHS_GNN"], ckpt_name="standalone_transformer_best.pth")


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(gnn_history["loss"]); axes[0].set_title("GNN training loss"); axes[0].set_xlabel("epoch"); axes[0].grid(alpha=0.3)
axes[1].plot(gnn_history["val_acc"], label="val acc", color="green")
axes[1].plot(gnn_history["val_auc"], label="val AUC", color="darkorange")
axes[1].set_title("Proxy dumping-task validation metrics"); axes[1].set_xlabel("epoch"); axes[1].grid(alpha=0.3); axes[1].legend()
plt.tight_layout()
savefig("gnn_curves.png"); plt.show()

all_true, all_pred = [], []
all_graph_true, all_graph_prob = [], []
all_embeddings, all_embed_labels = [], []
gnn.eval()
with torch.no_grad():
    for g in graphs:
        if g["n_nodes"] < 1:
            continue
        Xb, Ab, Db, yb_node, mask_b, yb_graph, extra_b = graph_to_batch(g, GNN_DEVICE)
        node_logits, graph_logit, h = gnn(Xb, Ab, Db, mask_b, extra_b, return_embedding=True)
        pred_node = node_logits[0].argmax(dim=1).cpu().numpy()
        node_true_np = yb_node[0].cpu().numpy()
        valid_mask = node_true_np != -1
        all_true.extend(node_true_np[valid_mask].tolist())
        all_pred.extend(pred_node[valid_mask].tolist())
        all_graph_true.append(int(yb_graph.item()))
        all_graph_prob.append(torch.sigmoid(graph_logit).item())
        if valid_mask.any():
            all_embeddings.append(h[0].cpu().numpy()[valid_mask])
            all_embed_labels.append(node_true_np[valid_mask])
all_embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, 1))
all_embed_labels = np.concatenate(all_embed_labels, axis=0) if all_embed_labels else np.zeros((0,))

node_acc = None
if len(all_true) > 0:
    cm = confusion_matrix(all_true, all_pred, labels=list(range(NUM_CLASSES)))
    support = cm.sum(axis=1)
    zero_support = [label_names[i] for i in range(NUM_CLASSES) if support[i] == 0]
    if zero_support:
        print(f"  [note] classes with ZERO GT-matched instances in the REAL-pipeline test set "
              f"this run: {zero_support}. This is a detector-recall limitation, not a GNN failure.")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(label_names, rotation=90); ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("GNN node-level classification (test) — target = real GT class")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=7)
    plt.colorbar(im)
    plt.tight_layout()
    savefig("node_confusion_matrix.png"); plt.show()
    node_acc = accuracy_score(all_true, all_pred)
    print(f">>> GNN node-level classification accuracy (test): {node_acc:.3f}")

acc = f1 = roc_auc = None
if len(set(all_graph_true)) > 1:
    fpr, tpr, _ = roc_curve(all_graph_true, all_graph_prob)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(all_graph_true, all_graph_prob)
    graph_pred = [1 if p > 0.5 else 0 for p in all_graph_prob]
    acc = accuracy_score(all_graph_true, graph_pred)
    f1 = f1_score(all_graph_true, graph_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); axes[0].plot([0,1],[0,1],'k--',alpha=0.4)
    axes[0].set_title("Proxy dumping detection — ROC"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()
    axes[1].plot(rec, prec); axes[1].set_title("Proxy dumping detection — PR curve")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    plt.tight_layout()
    savefig("dumping_roc_pr.png"); plt.show()
    print(f">>> [REAL PIPELINE] PROXY illegal-dumping task, detector-based test graphs: "
          f"accuracy={acc:.3f}  F1={f1:.3f}  AUC={roc_auc:.3f}  (n_test_graphs={len(all_graph_true)}, "
          f"n_positive={sum(all_graph_true)})")
    if len(all_graph_true) < 60:
        print(f"  [caution] small test-graph pool (n={len(all_graph_true)}) — accuracy/F1 at a fixed "
              f"0.5 threshold can be unstable with few positives; prefer AUC + its bootstrap CI below.")

    if CFG["RUN_BOOTSTRAP"]:
        rng = np.random.RandomState(0)
        y_true_a, y_prob_a = np.array(all_graph_true), np.array(all_graph_prob)
        auc_vals = []
        for _ in range(CFG["N_BOOTSTRAP"]):
            idx = rng.randint(0, len(y_true_a), len(y_true_a))
            if len(set(y_true_a[idx].tolist())) < 2:
                continue
            f_, t_, _ = roc_curve(y_true_a[idx], y_prob_a[idx])
            auc_vals.append(auc(f_, t_))
        if auc_vals:
            print(f">>> Bootstrap AUC 95% CI (GNN): mean={np.mean(auc_vals):.3f}  "
                  f"[{np.percentile(auc_vals,2.5):.3f}, {np.percentile(auc_vals,97.5):.3f}] "
                  f"(n_boot={len(auc_vals)})")

    if len(set(all_graph_true)) > 1 and len(all_graph_true) >= 8:
        try:
            frac_pos, mean_pred = calibration_curve(all_graph_true, all_graph_prob, n_bins=8, strategy="quantile")
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.plot(mean_pred, frac_pos, "o-", label="GNN (real pipeline)")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
            ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
            ax.set_title("Reliability curve — dumping-proxy graph task"); ax.legend()
            plt.tight_layout()
            savefig("calibration_curve.png"); plt.show()
        except ValueError:
            print("  [skip] not enough distinct probability bins for a calibration curve yet.")
else:
    print("Not enough class variety in the DETECTOR-BASED test graph labels for ROC/PR.")

if len(all_embeddings) > 10:
    n_show = min(CFG["MAX_TSNE_SAMPLES"], len(all_embeddings))
    idx = np.random.RandomState(0).choice(len(all_embeddings), n_show, replace=False)
    try:
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, n_show // 10)), random_state=SEED, init="pca")
        proj = tsne.fit_transform(all_embeddings[idx])
        fig, ax = plt.subplots(figsize=(7, 6))
        lab = all_embed_labels[idx].astype(int)
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=lab, cmap="tab10", s=10, alpha=0.8)
        handles = [mpatches.Patch(color=plt.cm.tab10(i / max(NUM_CLASSES - 1, 1)), label=label_names[i])
                   for i in sorted(set(lab.tolist()))]
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        ax.set_title("t-SNE of learned node embeddings (colored by true class)")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        savefig("tsne_node_embeddings.png"); plt.show()
    except Exception as e:
        print(f"  [skip] t-SNE failed ({e}); not critical to the results.")

gt_true, gt_pred, gt_graph_true, gt_graph_prob = [], [], [], []
with torch.no_grad():
    for g in test_graphs_gt:
        if g["n_nodes"] < 1:
            continue
        Xb, Ab, Db, yb_node, mask_b, yb_graph, extra_b = graph_to_batch(g, GNN_DEVICE)
        node_logits, graph_logit = gnn(Xb, Ab, Db, mask_b, extra_b)
        gt_pred.extend(node_logits[0].argmax(dim=1).cpu().numpy().tolist())
        gt_true.extend(yb_node[0].cpu().numpy().tolist())
        gt_graph_true.append(int(yb_graph.item()))
        gt_graph_prob.append(torch.sigmoid(graph_logit).item())

node_acc_upper = accuracy_score(gt_true, gt_pred) if gt_true else None
if node_acc_upper is not None:
    print(f">>> [GT-BOX CEILING] node accuracy on GT-based test graphs (input label == target "
          f"by construction; expected ~1.0 regardless of context-reasoning quality): {node_acc_upper:.3f}")
acc_upper = f1_upper = auc_upper = None
if len(set(gt_graph_true)) > 1:
    gt_graph_pred = [1 if p > 0.5 else 0 for p in gt_graph_prob]
    acc_upper = accuracy_score(gt_graph_true, gt_graph_pred)
    f1_upper = f1_score(gt_graph_true, gt_graph_pred)
    fpr_u, tpr_u, _ = roc_curve(gt_graph_true, gt_graph_prob)
    auc_upper = auc(fpr_u, tpr_u)
    print(f">>> [GT-BOX CEILING] PROXY dumping task on GT-based test graphs: "
          f"accuracy={acc_upper:.3f}  F1={f1_upper:.3f}  AUC={auc_upper:.3f}")


synth_true, synth_pred = [], []
with torch.no_grad():
    for g in test_graphs_synthetic_noise:
        if g["n_nodes"] < 1:
            continue
        Xb, Ab, Db, yb_node, mask_b, yb_graph, extra_b = graph_to_batch(g, GNN_DEVICE)
        node_logits, _ = gnn(Xb, Ab, Db, mask_b, extra_b)
        synth_pred.extend(node_logits[0].argmax(dim=1).cpu().numpy().tolist())
        synth_true.extend(yb_node[0].cpu().numpy().tolist())
synth_acc = accuracy_score(synth_true, synth_pred) if synth_true else None
if synth_acc is not None:
    naive_copy_acc = 1.0 - CFG["SYNTHETIC_NOISE_FRAC"]
    print(f">>> [CONTEXT-CORRECTION DIAGNOSTIC] node accuracy on {CFG['SYNTHETIC_NOISE_FRAC']:.0%} "
          f"synthetically-corrupted GT graphs (detector-independent): {synth_acc:.3f}  "
          f"(naive 'just copy the input label' baseline = {naive_copy_acc:.3f}). "
          f"A score meaningfully ABOVE {naive_copy_acc:.3f} is direct evidence the GNN is using "
          f"neighbor context to override a wrong label, not just echoing its input.")


def evaluate_context_model_on_graphs(model, g_list, device):
    """Runs `model` over every graph in g_list and returns per-graph
    correctness-relevant outputs, mirroring how the main GNN is evaluated
    in Cell 11 (graph_to_batch, batch-of-1)."""
    model.eval()
    graph_true, graph_prob = [], []
    with torch.no_grad():
        for g in g_list:
            if g["n_nodes"] < 1:
                continue
            Xb, Ab, Db, yb_node, mask_b, yb_graph, extra_b = graph_to_batch(g, device)
            _, graph_logit = model(Xb, Ab, Db, mask_b, extra_b)
            graph_true.append(int(yb_graph.item()))
            graph_prob.append(torch.sigmoid(graph_logit).item())
    return np.array(graph_true), np.array(graph_prob)

def summarize_model_on_test(name, y_true, y_prob, n_boot, seed=1):
    entry = {"prob_test": y_prob, "true_test": y_true}
    if len(set(y_true.tolist())) > 1:
        y_pred = (y_prob > 0.5).astype(int)
        entry["pred_test"] = y_pred
        entry["acc"] = accuracy_score(y_true, y_pred)
        entry["f1"] = f1_score(y_true, y_pred)
        fpr_b, tpr_b, _ = roc_curve(y_true, y_prob)
        entry["auc"] = auc(fpr_b, tpr_b)
        rng_b = np.random.RandomState(seed)
        auc_vals_b = []
        for _ in range(n_boot):
            idx = rng_b.randint(0, len(y_true), len(y_true))
            if len(set(y_true[idx].tolist())) < 2:
                continue
            f_, t_, _ = roc_curve(y_true[idx], y_prob[idx])
            auc_vals_b.append(auc(f_, t_))
        if auc_vals_b:
            entry["auc_ci"] = (float(np.percentile(auc_vals_b, 2.5)), float(np.percentile(auc_vals_b, 97.5)))
        ci_str = f" 95% CI=[{entry['auc_ci'][0]:.3f},{entry['auc_ci'][1]:.3f}]" if "auc_ci" in entry else ""
        print(f">>> [BASELINE-{name}] acc={entry['acc']:.3f} f1={entry['f1']:.3f} "
              f"auc={entry['auc']:.3f}{ci_str}")
    else:
        print(f"  [skip] {name}: test labels have only one class.")
    return entry

baseline_results = {}
sig_results = {}
if CFG["RUN_BASELINES"] and len(graphs) > 0:
    t0 = tic()

    print("--- architecture-ablation baselines (same training recipe as the main GNN) ---")
    gcn_true, gcn_prob = evaluate_context_model_on_graphs(standalone_gcn, graphs, GNN_DEVICE)
    transformer_true, transformer_prob = evaluate_context_model_on_graphs(standalone_transformer, graphs, GNN_DEVICE)

    if CFG["RUN_TESTS"] and len(all_graph_true) > 0:
        assert list(gcn_true) == list(all_graph_true), \
            "StandaloneGCN test-graph label order mismatch vs. main GNN — evaluation set drifted."
        assert list(transformer_true) == list(all_graph_true), \
            "StandaloneTransformer test-graph label order mismatch vs. main GNN — evaluation set drifted."

    baseline_results["StandaloneGCN"] = summarize_model_on_test(
        "StandaloneGCN (graph, no attention)", gcn_true, gcn_prob, CFG["N_BOOTSTRAP"], seed=1)
    baseline_results["StandaloneTransformer"] = summarize_model_on_test(
        "StandaloneTransformer (attention, no graph structure)", transformer_true, transformer_prob,
        CFG["N_BOOTSTRAP"], seed=2)

    def paired_bootstrap_auc_diff(y_true, prob_a, prob_b, n_boot, seed=0):
        y_true = np.asarray(y_true); prob_a = np.asarray(prob_a); prob_b = np.asarray(prob_b)
        rng = np.random.RandomState(seed)
        diffs = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(y_true), len(y_true))
            if len(set(y_true[idx].tolist())) < 2:
                continue
            fa, ta, _ = roc_curve(y_true[idx], prob_a[idx]); auc_a = auc(fa, ta)
            fb, tb, _ = roc_curve(y_true[idx], prob_b[idx]); auc_b = auc(fb, tb)
            diffs.append(auc_a - auc_b)
        if not diffs:
            return None
        diffs = np.array(diffs)
        return {
            "mean_diff": float(diffs.mean()),
            "ci_lo": float(np.percentile(diffs, 2.5)),
            "ci_hi": float(np.percentile(diffs, 97.5)),
            "p_gnn_better": float((diffs > 0).mean()),
        }

    if len(all_graph_true) > 0:
        for bname, bres in baseline_results.items():
            if "prob_test" in bres and len(bres["prob_test"]) == len(all_graph_prob):
                res = paired_bootstrap_auc_diff(all_graph_true, all_graph_prob, bres["prob_test"], CFG["N_BOOTSTRAP"])
                if res:
                    sig_results[f"auc_diff_gnn_vs_{bname}"] = res
                    verdict = ("GNN significantly better" if res["ci_lo"] > 0 else
                                "baseline significantly better" if res["ci_hi"] < 0 else
                                "not statistically distinguishable")
                    print(f">>> Paired bootstrap AUC diff (GNN - {bname}): mean={res['mean_diff']:+.3f}  "
                          f"95% CI=[{res['ci_lo']:+.3f}, {res['ci_hi']:+.3f}]  "
                          f"P(GNN>{bname})={res['p_gnn_better']:.2f}  -> {verdict}")


    if len(all_graph_true) > 0:
        gnn_pred = np.array([1 if p > 0.5 else 0 for p in all_graph_prob])
        correct_gnn = (gnn_pred == np.array(all_graph_true)).astype(float)
        for bname, bres in baseline_results.items():
            if "pred_test" in bres and len(bres["pred_test"]) == len(correct_gnn):
                correct_b = (bres["pred_test"] == bres["true_test"]).astype(float)
                try:
                    t_stat, p_val = scipy_stats.ttest_rel(correct_gnn, correct_b)
                    mean_diff = float(np.mean(correct_gnn - correct_b))
                    sig_results[f"ttest_gnn_vs_{bname}"] = {
                        "mean_diff": mean_diff, "t_stat": float(t_stat), "p_value": float(p_val)}
                    verdict = "significant" if (not math.isnan(p_val) and p_val < 0.05) else "not significant"
                    print(f">>> Paired t-test (GNN vs {bname}, per-graph accuracy): "
                          f"mean_diff={mean_diff:+.3f}  t={t_stat:.3f}  p={p_val:.4f} ({verdict})")
                except Exception as e:
                    print(f"  [skip] paired t-test GNN vs {bname} failed: {e}")

    print("\n  [research-validity note]")
    toc(t0, "baselines_and_significance")


def visualize_sample(img_id, dets, graph, ax):
    info = coco.loadImgs(img_id)[0]
    path = os.path.join(CFG["DATA_ROOT"], info["file_name"])
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        color = "red" if graph is not None and graph["graph_label"] == 1 else "lime"
        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x1, max(y1-4, 0), label_names[d["label"]], color=color, fontsize=7,
                 bbox=dict(facecolor="black", alpha=0.4, pad=0.5))
    if graph is not None:
        centroids = [d["centroid"] for d in dets]
        n = len(centroids)
        A = graph["A"]
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] > 0:
                    x1c, y1c = centroids[i]; x2c, y2c = centroids[j]
                    ax.plot([x1c, x2c], [y1c, y2c], color="yellow", alpha=0.5, linewidth=0.8)
    title = "PROXY dumping cluster" if (graph is not None and graph["graph_label"] == 1) else "normal scatter"
    ax.set_title(f"img {img_id} — {title}", fontsize=9)
    ax.axis("off")

n_vis = min(CFG["VIS_SAMPLES"], len(test_dets))
sample_ids = list(test_dets.keys())[:n_vis]
if n_vis > 0:
    ncols = 4
    nrows = math.ceil(n_vis / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, img_id in zip(axes, sample_ids):
        dets = test_dets[img_id]
        g = build_graph_for_image(dets, get_img_diag(img_id), CFG["DUMP_MIN_OBJ"], CFG["DUMP_DIST"])
        visualize_sample(img_id, dets, g, ax)
    for ax in axes[len(sample_ids):]:
        ax.axis("off")
    plt.suptitle("Detection + context graph (green/red boxes, yellow = spatial edges)", fontsize=12)
    plt.tight_layout()
    savefig("qualitative_results.png"); plt.show()
else:
    print("  [skip] no test detections available to visualize.")


def get_auc(d, name):
    return round(float(d[name].get("auc", float("nan"))), 3) if name in d else None

results_table = pd.DataFrame([
    {"Metric": "Detection mAP@[.5:.95] (bbox, test)", "Value": round(float(mAP_50_95), 3)},
    {"Metric": "Detection mAP@0.5 (bbox, test)",       "Value": round(float(mAP_50), 3)},
    {"Metric": "Segmentation mAP@[.5:.95] (segm, test)", "Value": round(float(segm_mAP_50_95), 3)},
    {"Metric": "Segmentation mAP@0.5 (segm, test)",       "Value": round(float(segm_mAP_50), 3)},
    {"Metric": "[secondary] Mask mIoU (image-level union, test)", "Value": round(float(mask_miou), 3)},
    {"Metric": "GNN(Graphormer) epochs completed before time budget", "Value": n_gnn_epochs_run},
    {"Metric": "StandaloneGCN epochs completed before time budget", "Value": n_gcn_epochs_run},
    {"Metric": "StandaloneTransformer epochs completed before time budget", "Value": n_transformer_epochs_run},
    {"Metric": "GNN node classification accuracy — REAL pipeline (test)", "Value": round(float(node_acc), 3) if node_acc is not None else None},
    {"Metric": "GNN node accuracy — GT-BOX CEILING (trivial, expect ~1.0)", "Value": round(float(node_acc_upper), 3) if node_acc_upper is not None else None},
    {"Metric": f"GNN node accuracy — CONTEXT-CORRECTION diagnostic ({CFG['SYNTHETIC_NOISE_FRAC']:.0%} synthetic noise)",
     "Value": round(float(synth_acc), 3) if synth_acc is not None else None},
    {"Metric": "  ...naive 'copy input label' baseline for the diagnostic above",
     "Value": round(1.0 - CFG["SYNTHETIC_NOISE_FRAC"], 3)},
    {"Metric": "PROXY dumping accuracy — REAL pipeline, GNN(Graphormer) (test)", "Value": round(float(acc), 3) if acc is not None else None},
    {"Metric": "PROXY dumping F1 — REAL pipeline, GNN(Graphormer) (test)",       "Value": round(float(f1), 3) if f1 is not None else None},
    {"Metric": "PROXY dumping AUC — REAL pipeline, GNN(Graphormer) (test)",      "Value": round(float(roc_auc), 3) if roc_auc is not None else None},
    {"Metric": "PROXY dumping test-graph count / n_positive", "Value": f"{len(all_graph_true)}/{sum(all_graph_true)}" if all_graph_true else None},
    {"Metric": "Train/test proxy positive-rate gap", "Value": round(abs(pos_rate_test - pos_rate_train), 3)},
    {"Metric": "PROXY dumping accuracy — GT-BOX CEILING (trivial)", "Value": round(float(acc_upper), 3) if acc_upper is not None else None},
    {"Metric": "PROXY dumping AUC — GT-BOX CEILING (trivial)",      "Value": round(float(auc_upper), 3) if auc_upper is not None else None},
    {"Metric": "BASELINE StandaloneGCN AUC (graph structure, no attention)", "Value": get_auc(baseline_results, "StandaloneGCN")},
    {"Metric": "BASELINE StandaloneTransformer AUC (attention, no graph structure)", "Value": get_auc(baseline_results, "StandaloneTransformer")},
    {"Metric": "Paired-bootstrap AUC diff GNN - StandaloneGCN (mean)",
     "Value": round(sig_results["auc_diff_gnn_vs_StandaloneGCN"]["mean_diff"], 3) if "auc_diff_gnn_vs_StandaloneGCN" in sig_results else None},
    {"Metric": "Paired-bootstrap AUC diff GNN - StandaloneTransformer (mean)",
     "Value": round(sig_results["auc_diff_gnn_vs_StandaloneTransformer"]["mean_diff"], 3) if "auc_diff_gnn_vs_StandaloneTransformer" in sig_results else None},
    {"Metric": "Paired t-test mean diff, GNN vs StandaloneGCN (per-graph accuracy)",
     "Value": round(sig_results["ttest_gnn_vs_StandaloneGCN"]["mean_diff"], 3) if "ttest_gnn_vs_StandaloneGCN" in sig_results else None},
    {"Metric": "Paired t-test p-value, GNN vs StandaloneGCN",
     "Value": round(sig_results["ttest_gnn_vs_StandaloneGCN"]["p_value"], 4) if "ttest_gnn_vs_StandaloneGCN" in sig_results else None},
    {"Metric": "Paired t-test mean diff, GNN vs StandaloneTransformer (per-graph accuracy)",
     "Value": round(sig_results["ttest_gnn_vs_StandaloneTransformer"]["mean_diff"], 3) if "ttest_gnn_vs_StandaloneTransformer" in sig_results else None},
    {"Metric": "Paired t-test p-value, GNN vs StandaloneTransformer",
     "Value": round(sig_results["ttest_gnn_vs_StandaloneTransformer"]["p_value"], 4) if "ttest_gnn_vs_StandaloneTransformer" in sig_results else None},
])
print(results_table.to_string(index=False))
results_table.to_csv(os.path.join(CFG["OUT_DIR"], "final_results_table.csv"), index=False)

