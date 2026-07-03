import os
import cv2
import torch
import gc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from segment_anything import build_sam, SamPredictor
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer) +
        f"({str(logit.max().item())[:4]})"
        for logit in logits_filt
    ]
    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    color = np.random.random(3) if random_color else np.array([30/255, 144/255, 255/255])
    color = np.concatenate([color, np.array([0.6])])
    mask_image = mask.reshape(*mask.shape[-2:], 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def save_mask_data(output_dir, mask_list, box_list, label_list):
    max_logit_index = np.argmax([float(label.split('(')[1][:-1]) for label in label_list])
    mask_img = torch.zeros((*mask_list.shape[-2:], 4), dtype=torch.uint8)
    mask_img[..., :3] = 255
    mask_img[..., 3] = 255
    selected_mask = mask_list[max_logit_index]
    mask_img[selected_mask.cpu().numpy()[0]] = torch.tensor([0, 0, 0, 255], dtype=torch.uint8)
    mask_img_pil = Image.fromarray(mask_img.numpy(), mode="RGBA")
    mask_img_pil.save(os.path.join(output_dir, 'mask.png'))
    json_data = [{'value': 0, 'label': 'background'}]
    json_data += [
        {
            'value': i + 1,
            'label': label.split('(')[0],
            'logit': float(label.split('(')[1][:-1]),
            'box': box.numpy().tolist()
        }
        for i, (label, box) in enumerate(zip(label_list, box_list))
    ]
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

def replace_white_with_black(image_path):
    image = Image.open(image_path).convert("RGBA")
    data = np.array(image)
    red, green, blue, alpha = data.T
    white_areas = (red == 255) & (green == 255) & (blue == 255)
    data[..., :-1][white_areas.T] = [0, 0, 0]
    black_background_image = Image.fromarray(data)
    return black_background_image

def crop_black_borders(image):
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)
    non_black_mask = np.any(image_np != [0, 0, 0], axis=-1)
    non_black_indices = np.where(non_black_mask)
    if non_black_indices[0].size > 0 and non_black_indices[1].size > 0:
        min_y, max_y = np.min(non_black_indices[0]), np.max(non_black_indices[0])
        min_x, max_x = np.min(non_black_indices[1]), np.max(non_black_indices[1])
        cropped_image = image_rgb.crop((min_x, min_y, max_x + 1, max_y + 1))
        return cropped_image
    else:
        print("警告: 图像中未找到非黑色像素")
        return image_rgb

def process_single_image(
    image_path,
    output_dir,
    config_file,
    grounded_checkpoint,
    sam_checkpoint,
    box_threshold,
    text_threshold,
    text_prompt,
    device="cuda"
):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    temp_dir = os.path.join(output_dir, "temp_results")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    print("=" * 60)
    print("开始加载模型...")
    print("=" * 60)
    model = load_model(config_file, grounded_checkpoint, device=device)
    sam_model = build_sam(checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam_model)
    print("模型加载完成！\n")
    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]
    output_subdir = os.path.join(temp_dir, base_name)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    print(f"正在处理: {filename}")
    try:
        print(f"  步骤1: 执行图像分割...")
        image_pil, image = load_image(image_path)
        image_pil.save(os.path.join(output_subdir, "raw_image.jpg"))
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )
        if len(boxes_filt) == 0:
            print(f"  警告: 未检测到目标 '{text_prompt}'")
            return False
        image_np = np.array(image_pil)
        predictor.set_image(image_np)
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_filt.cpu(), image_np.shape[:2]
        ).to(device)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        plt.figure(figsize=(10, 10), dpi=100)
        plt.imshow(image_np)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)
        plt.axis('off')
        plt.savefig(
            os.path.join(output_subdir, "grounded_sam_output.png"),
            bbox_inches="tight",
            dpi=100,
            pad_inches=0.0
        )
        plt.close()
        save_mask_data(output_subdir, masks, boxes_filt, pred_phrases)
        img1 = cv2.imread(os.path.join(output_subdir, "raw_image.jpg"))
        img2 = cv2.imread(os.path.join(output_subdir, "mask.png"))
        white_bg_image = cv2.add(img1, img2)
        white_bg_path = os.path.join(output_subdir, "white_background.png")
        cv2.imwrite(white_bg_path, white_bg_image)
        print(f"  步骤1完成: 分割成功")
        print(f"  步骤2: 白底转黑底...")
        black_bg_image = replace_white_with_black(white_bg_path)
        black_bg_path = os.path.join(output_subdir, "black_background.png")
        black_bg_image.save(black_bg_path)
        print(f"  步骤2完成: 背景转换成功")
        print(f"  步骤3: 裁剪黑边...")
        cropped_image = crop_black_borders(black_bg_image)
        final_output_path = os.path.join(output_dir, f"{base_name}_final.png")
        cropped_image.save(final_output_path)
        print(f"  步骤3完成: 裁剪完成")
        print(f"最终结果已保存: {final_output_path}\n")
        del image_pil, image, image_np, boxes_filt, pred_phrases, masks
        del white_bg_image, black_bg_image, cropped_image
        gc.collect()
        torch.cuda.empty_cache()
        print("=" * 60)
        print(f"处理完成！结果保存在: {output_dir}")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"处理失败: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    process_single_image(
        image_path="/path/to/example_tongue.jpg",
        output_dir="/path/to/segmentation_output",
        config_file="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounded_checkpoint="groundingdino_swint_ogc.pth",
        sam_checkpoint="sam_vit_h_4b8939.pth",
        box_threshold=0.3,
        text_threshold=0.25,
        text_prompt="tongue",
        device="cuda"
    )
