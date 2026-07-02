







import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import math
import random
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from einops import rearrange, repeat




def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">> [Seed] 全局随机种子已设置为: {seed}")

SEED = 42
seed_everything(SEED)




try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except ImportError:
        causal_conv1d_fn, causal_conv1d_update = None, None

    HAS_MAMBA_KERNELS = True
    print(">> [Mamba] 底层加速算子加载成功。使用自定义 Coord-Mamba。")
except ImportError:
    HAS_MAMBA_KERNELS = False
    print("!! [Warning] 未找到 Mamba 环境，将退化为 Bi-LSTM 或报错。")




def patch_densenet121_to_14x14_with_mixed_dilation(
    features: nn.Module,
    pattern=(1, 2),
):
    if hasattr(features, "transition3") and hasattr(features.transition3, "pool"):
        features.transition3.pool = nn.Identity()
    else:
        raise RuntimeError("找不到 features.transition3.pool，torchvision 版本可能不同。")

    idx = 0
    for m in features.denseblock4.modules():
        if m.__class__.__name__ == "_DenseLayer" and hasattr(m, "conv2"):
            old = m.conv2
            if isinstance(old, nn.Conv2d) and old.kernel_size == (3, 3):
                d = pattern[idx % len(pattern)]
                idx += 1
                new = nn.Conv2d(
                    in_channels=old.in_channels,
                    out_channels=old.out_channels,
                    kernel_size=3,
                    stride=old.stride,
                    padding=(d, d),
                    dilation=(d, d),
                    groups=old.groups,
                    bias=(old.bias is not None),
                    padding_mode=old.padding_mode,
                )
                new.weight.data.copy_(old.weight.data)
                if old.bias is not None:
                    new.bias.data.copy_(old.bias.data)
                m.conv2 = new

    return features




class CoordAtt2D_for_Mamba(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        mip = max(8, inp // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        B, D, L = x.size()
        H_feat = int(math.sqrt(L))
        W_feat = int(math.sqrt(L))
        if H_feat * W_feat != L:
            return x

        x_2d = x.view(B, D, H_feat, W_feat)
        identity = x_2d

        _, _, h, w = x_2d.size()
        x_h = self.pool_h(x_2d)
        x_w = self.pool_w(x_2d).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out_2d = identity * a_h * a_w
        out = out_2d.view(B, D, L)
        return out




class CoordMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.coord_att = CoordAtt2D_for_Mamba(inp=self.d_inner, reduction=16)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)

        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_fn(
            x, dt, A, B, C, self.D.float(), z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )

        y = self.coord_att(y)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out




data_root = "/path/to/tongue_dataset_split"
save_path = "checkpoints/densenet_mamba_coord_A2_14x14_mixedDil.pth"

vis_save_dir = "visualization_fig9_style_masked_compact"
os.makedirs(vis_save_dir, exist_ok=True)

num_classes = 2
img_size = 320
batch_size = 16
num_epochs = 35
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr_backbone = 1e-5
lr_head = 2e-4
weight_decay = 1e-4
label_smoothing = 0.1


HEATMAP_BLUR_KSIZE = 9
HEATMAP_GAMMA = 2.8
HEATMAP_DISPLAY_THRESH = 0.30
BBOX_THRESH_RATIO = 0.84

print(f"Using device: {device}")
os.makedirs("checkpoints", exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
val_test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(data_root, "val"),   transform=val_test_transform)
test_dataset  = datasets.ImageFolder(os.path.join(data_root, "test"),  transform=val_test_transform)

print("train class_to_idx:", train_dataset.class_to_idx)
print("val   class_to_idx:", val_dataset.class_to_idx)
print("test  class_to_idx:", test_dataset.class_to_idx)

class_to_idx = train_dataset.class_to_idx
class_names = train_dataset.classes
lymphoma_idx = class_to_idx["lymphoma"]
normal_idx = class_to_idx["normal"]

print("lymphoma_idx =", lymphoma_idx)
print("normal_idx   =", normal_idx)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, worker_init_fn=seed_worker, generator=g
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, worker_init_fn=seed_worker, generator=g
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, worker_init_fn=seed_worker, generator=g
)




class DenseMambaTongue(nn.Module):
    def __init__(
        self,
        num_classes=2,
        d_model=1024,
        img_size=224,
        mixed_dilation_pattern=(1, 2),
        use_global_fusion=False,
    ):
        super().__init__()
        self.use_global_fusion = use_global_fusion

        base_model = models.densenet121(weights="DEFAULT")
        self.features = base_model.features

        self.features = patch_densenet121_to_14x14_with_mixed_dilation(
            self.features, pattern=mixed_dilation_pattern
        )

        self.d_model = d_model

        if HAS_MAMBA_KERNELS:
            self.mamba = CoordMamba(
                d_model=self.d_model, d_state=16, d_conv=4, expand=2, use_fast_path=False
            )
        else:
            self.mamba = nn.LSTM(
                input_size=self.d_model,
                hidden_size=self.d_model // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat = self.features(dummy)
            _, _, H, W = feat.shape
            self.num_tokens = H * W
            self.feat_h = H
            self.feat_w = W

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, self.d_model) * 0.02)

        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(self.d_model, num_classes)

        print(f">> [Backbone] DenseNet 输出 token 数 = {self.num_tokens}")
        print(f">> [Backbone] denseblock4 mixed dilation pattern = {mixed_dilation_pattern}")

    def forward(self, x, return_maps=False):
        feat = self.features(x)
        feat = F.relu(feat, inplace=False)

        if self.use_global_fusion:
            g = F.avg_pool2d(feat, kernel_size=2, stride=2)
            g = F.interpolate(g, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            feat = feat + g

        b, c, h, w = feat.shape
        tokens = feat.view(b, c, h * w).permute(0, 2, 1)

        if tokens.size(1) != self.pos_embedding.size(1):
            L0 = self.pos_embedding.size(1)
            s0 = int(math.sqrt(L0))
            pos = self.pos_embedding.transpose(1, 2).view(1, self.d_model, s0, s0)
            pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
            pos = pos.view(1, self.d_model, h * w).transpose(1, 2)
            tokens = tokens + pos
        else:
            tokens = tokens + self.pos_embedding

        if HAS_MAMBA_KERNELS:
            x_fwd = self.mamba(tokens)
            x_bwd = self.mamba(tokens.flip([1])).flip([1])
            x = x_fwd + x_bwd
        else:
            x, _ = self.mamba(tokens)

        x = self.norm(x)
        pooled = x.mean(dim=1)
        pooled = self.dropout(pooled)
        out = self.classifier(pooled)

        if return_maps:
            return out, feat, x
        return out

model = DenseMambaTongue(
    num_classes=num_classes,
    img_size=img_size,
    mixed_dilation_pattern=(1, 2),
    use_global_fusion=False,
).to(device)




backbone_params = list(map(id, model.features.parameters()))
head_params = filter(lambda p: id(p) not in backbone_params, model.parameters())

optimizer = torch.optim.AdamW([
    {"params": model.features.parameters(), "lr": lr_backbone},
    {"params": head_params, "lr": lr_head},
], weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

def compute_auc_for_lymphoma(labels, probs, lymphoma_idx):
    binary_labels = np.array([1 if y == lymphoma_idx else 0 for y in labels])
    if len(np.unique(binary_labels)) < 2:
        return 0.0
    return float(roc_auc_score(binary_labels, probs))

def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, leave=False, desc="Training")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    scheduler.step()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_auc(model, loader, positive_class_idx):
    model.eval()
    probs, gts = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        p = torch.softmax(outputs, dim=1)[:, positive_class_idx]
        probs.extend(p.cpu().numpy())
        gts.extend(labels.numpy())
    return compute_auc_for_lymphoma(gts, probs, positive_class_idx)




def denormalize_image(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    img = img_tensor.detach().cpu().clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).numpy()
    return img

def normalize_map(score_map):
    score_map = score_map.astype(np.float32)
    min_v = score_map.min()
    max_v = score_map.max()
    if max_v - min_v < 1e-8:
        return np.zeros_like(score_map)
    return (score_map - min_v) / (max_v - min_v)

def get_tongue_mask_from_image(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    img = denormalize_image(img_tensor, mean, std)
    img_uint8 = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    kernel2 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_idx).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)

    return mask

def refine_score_map_with_mask(score_map, tongue_mask, smooth=True):
    Hf, Wf = score_map.shape

    mask_small = cv2.resize(
        tongue_mask.astype(np.float32),
        (Wf, Hf),
        interpolation=cv2.INTER_AREA
    )
    mask_small = (mask_small > 0.3).astype(np.float32)

    refined = score_map.copy().astype(np.float32)
    refined = refined - refined.min()
    if refined.max() > 1e-8:
        refined = refined / refined.max()

    refined = refined * mask_small

    if smooth:
        refined = cv2.GaussianBlur(refined, (3, 3), 0)

    refined = refined - refined.min()
    if refined.max() > 1e-8:
        refined = refined / refined.max()

    return refined, mask_small

def get_bbox_from_masked_score_map(score_map, img_h, img_w, thresh_ratio=BBOX_THRESH_RATIO):
    Hf, Wf = score_map.shape
    s = score_map.astype(np.float32)

    if s.max() < 1e-8:
        r, c = Hf // 2, Wf // 2
        rows = np.array([r])
        cols = np.array([c])
    else:
        mask = (s >= thresh_ratio).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1 and stats[1:, cv2.CC_STAT_AREA].max() > 0:
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            region = (labels == largest_idx)
            rows, cols = np.where(region)
        else:
            flat_idx = np.argsort(s.reshape(-1))[-4:]
            rows, cols = np.unravel_index(flat_idx, (Hf, Wf))

    r1, r2 = rows.min(), rows.max()
    c1, c2 = cols.min(), cols.max()

    patch_h = img_h / Hf
    patch_w = img_w / Wf

    x1 = int(round(c1 * patch_w))
    y1 = int(round(r1 * patch_h))
    x2 = int(round((c2 + 1) * patch_w))
    y2 = int(round((r2 + 1) * patch_h))

    return x1, y1, x2, y2

def build_compact_display_heatmap(score_map, tongue_mask, out_h, out_w,
                                  blur_ksize=HEATMAP_BLUR_KSIZE,
                                  gamma=HEATMAP_GAMMA,
                                  display_thresh=HEATMAP_DISPLAY_THRESH):
    heatmap = cv2.resize(
        score_map.astype(np.float32),
        (out_w, out_h),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = cv2.GaussianBlur(heatmap, (blur_ksize, blur_ksize), 0)

    tongue_mask_float = tongue_mask.astype(np.float32)
    heatmap = heatmap * tongue_mask_float

    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 1e-8:
        heatmap = heatmap / heatmap.max()


    heatmap = np.power(heatmap, gamma)

    heatmap[heatmap < display_thresh] = 0.0

    binary = (heatmap > 0).astype(np.uint8)


    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        keep = (labels == largest_idx).astype(np.float32)
        heatmap = heatmap * keep
    else:
        heatmap = heatmap * binary.astype(np.float32)

    if heatmap.max() > 1e-8:
        heatmap = heatmap / heatmap.max()

    return heatmap

@torch.no_grad()
def get_token_contribution_map(model, img_tensor, target_class=None):
    model.eval()
    inp = img_tensor.unsqueeze(0).to(device)
    logits, feat, tokens = model(inp, return_maps=True)

    probs = torch.softmax(logits, dim=1)
    pred_class = logits.argmax(dim=1).item()
    pred_prob = probs[0, pred_class].item()

    if target_class is None:
        target_class = pred_class

    cls_w = model.classifier.weight[target_class]
    token_scores = torch.matmul(tokens[0], cls_w)

    H = W = int(math.sqrt(token_scores.numel()))
    score_map = token_scores.view(H, W).detach().cpu().numpy()

    tongue_mask = get_tongue_mask_from_image(img_tensor)
    refined_score_map, _ = refine_score_map_with_mask(score_map, tongue_mask, smooth=True)

    img_h, img_w = img_tensor.shape[1], img_tensor.shape[2]
    bbox = get_bbox_from_masked_score_map(
        refined_score_map,
        img_h=img_h,
        img_w=img_w,
        thresh_ratio=BBOX_THRESH_RATIO
    )

    return pred_class, pred_prob, target_class, refined_score_map, bbox, tongue_mask

def save_fig9_style_visualization(
    img_tensor,
    score_map,
    bbox,
    tongue_mask,
    save_file,
    gt_label=None,
    pred_label=None,
    pred_prob=None,
    target_label=None,
):
    img = denormalize_image(img_tensor)
    x1, y1, x2, y2 = bbox

    score_map_norm = normalize_map(score_map)


    heatmap = build_compact_display_heatmap(
        score_map=score_map_norm,
        tongue_mask=tongue_mask,
        out_h=img.shape[0],
        out_w=img.shape[1],
        blur_ksize=HEATMAP_BLUR_KSIZE,
        gamma=HEATMAP_GAMMA,
        display_thresh=HEATMAP_DISPLAY_THRESH
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(img)
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2.5, edgecolor="red", facecolor="none"
    )
    axes[1].add_patch(rect)
    axes[1].set_title("Discriminative Region")
    axes[1].axis("off")

    axes[2].imshow(img)
    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=0.38,
        extent=(0, img.shape[1], img.shape[0], 0),
        interpolation="bilinear"
    )
    axes[2].set_title("Masked Token Contribution Heatmap")
    axes[2].axis("off")

    title_parts = []
    if gt_label is not None:
        title_parts.append(f"GT: {gt_label}")
    if pred_label is not None:
        if pred_prob is not None:
            title_parts.append(f"Pred: {pred_label} ({pred_prob:.4f})")
        else:
            title_parts.append(f"Pred: {pred_label}")
    if target_label is not None:
        title_parts.append(f"Response for: {target_label}")

    if len(title_parts) > 0:
        fig.suptitle(" | ".join(title_parts), fontsize=13)

    plt.tight_layout()
    plt.savefig(save_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

def visualize_dataset_samples(
    model,
    dataset,
    class_names,
    save_dir,
    num_per_class=5,
    target_class_mode="pred",
):
    os.makedirs(save_dir, exist_ok=True)
    class_counter = {i: 0 for i in range(len(class_names))}

    for idx in range(len(dataset)):
        img_tensor, label = dataset[idx]

        if class_counter[label] >= num_per_class:
            continue

        if target_class_mode == "pred":
            target_class = None
        elif target_class_mode == "gt":
            target_class = label
        elif isinstance(target_class_mode, int):
            target_class = target_class_mode
        else:
            raise ValueError("target_class_mode must be 'pred', 'gt', or int")

        pred_class, pred_prob, used_target_class, score_map, bbox, tongue_mask = get_token_contribution_map(
            model, img_tensor, target_class=target_class
        )

        save_file = os.path.join(
            save_dir,
            f"class_{class_names[label]}_{class_counter[label]:03d}_pred_{class_names[pred_class]}.png"
        )

        save_fig9_style_visualization(
            img_tensor=img_tensor,
            score_map=score_map,
            bbox=bbox,
            tongue_mask=tongue_mask,
            save_file=save_file,
            gt_label=class_names[label],
            pred_label=class_names[pred_class],
            pred_prob=pred_prob,
            target_label=class_names[used_target_class],
        )

        class_counter[label] += 1

        if all(v >= num_per_class for v in class_counter.values()):
            break

    print(f">> 可视化已保存到: {save_dir}")
    print(f">> 每类保存数量: {class_counter}")




best_auc = 0.0
print(f"\nStart Training Coord-Mamba + (A2)14x14+mixedDil [Seed={SEED}]...")

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
    val_auc = evaluate_auc(model, val_loader, lymphoma_idx)

    current_lr_cnn = optimizer.param_groups[0]["lr"]
    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {train_loss:.4f} | Val AUC(lymphoma): {val_auc:.4f} | LR(CNN): {current_lr_cnn:.1e}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), save_path)
        print(f"==> Best Model Saved (AUC: {best_auc:.4f})")

print(f"\nBest Val AUC (lymphoma): {best_auc:.4f}")




print("\nTesting model...")
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()

y_true, y_pred, y_probs = [], [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing"):
        imgs = imgs.to(device)
        outputs = model(imgs)

        probs = torch.softmax(outputs, dim=1)[:, lymphoma_idx]
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

test_auc = compute_auc_for_lymphoma(y_true, y_probs, lymphoma_idx)
print(f"\nFinal Test AUC (lymphoma as positive class): {test_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))




print("\nGenerating masked Fig.9-style visualizations...")
visualize_dataset_samples(
    model=model,
    dataset=test_dataset,
    class_names=class_names,
    save_dir=vis_save_dir,
    num_per_class=10,
    target_class_mode="pred"
)
