import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt





def out_from_in(layer_param, layer_in):
    n_in, j_in, r_in, start_in = layer_in
    k, s, p, d = layer_param


    k_eff = (k - 1) * d + 1


    n_out = math.floor((n_in + 2 * p - k_eff) / s) + 1


    actual_p = (n_out - 1) * s - n_in + k_eff
    p_left = math.floor(actual_p / 2)


    j_out = j_in * s


    r_out = r_in + (k_eff - 1) * j_in


    start_out = start_in + ((k_eff - 1) / 2 - p_left) * j_in

    return n_out, j_out, r_out, start_out






def build_modified_densenet121_rf_layers(mixed_pattern=(1, 2)):
    layers = []
    names = []


    layers.append([7, 2, 3, 1])
    names.append("conv0")

    layers.append([3, 2, 1, 1])
    names.append("pool0")


    for i in range(6):
        layers.append([3, 1, 1, 1])
        names.append(f"db1_layer{i+1}")


    layers.append([2, 2, 0, 1])
    names.append("trans1_pool")


    for i in range(12):
        layers.append([3, 1, 1, 1])
        names.append(f"db2_layer{i+1}")


    layers.append([2, 2, 0, 1])
    names.append("trans2_pool")


    for i in range(24):
        layers.append([3, 1, 1, 1])
        names.append(f"db3_layer{i+1}")


    names.append("trans3_identity")
    layers.append([1, 1, 0, 1])


    pattern_len = len(mixed_pattern)
    for i in range(16):
        d = mixed_pattern[i % pattern_len]
        p = d
        layers.append([3, 1, p, d])
        names.append(f"db4_layer{i+1}_d{d}")

    return layers, names





def compute_rf(imsize=224, mixed_pattern=(1, 2)):
    layers, names = build_modified_densenet121_rf_layers(mixed_pattern=mixed_pattern)

    current = [imsize, 1, 1, 0.5]
    results = [{
        "name": "input",
        "n": current[0],
        "j": current[1],
        "r": current[2],
        "start": current[3],
    }]

    for name, layer in zip(names, layers):
        if name == "trans3_identity":
            results.append({
                "name": name,
                "n": current[0],
                "j": current[1],
                "r": current[2],
                "start": current[3],
            })
        else:
            current = out_from_in(layer, current)
            results.append({
                "name": name,
                "n": current[0],
                "j": current[1],
                "r": current[2],
                "start": current[3],
            })

    return results





def print_stage_summary(results):
    keep = [
        "input",
        "conv0",
        "pool0",
        "db1_layer6",
        "trans1_pool",
        "db2_layer12",
        "trans2_pool",
        "db3_layer24",
        "trans3_identity",
        "db4_layer16_d2",
    ]

    print("=" * 76)
    print(f"{'Stage':20s} {'Feature Size':>14s} {'Jump':>10s} {'RF Size':>12s} {'Start':>10s}")
    print("=" * 76)

    for x in results:
        if x["name"] in keep:
            print(f"{x['name']:20s} {x['n']:14d} {x['j']:10d} {x['r']:12d} {x['start']:10.2f}")

    print("=" * 76)





def get_rf_box(results, layer_name, idx_x, idx_y):
    target = None
    for x in results:
        if x["name"] == layer_name:
            target = x
            break

    if target is None:
        raise ValueError(f"未找到层名: {layer_name}")

    n = target["n"]
    j = target["j"]
    r = target["r"]
    start = target["start"]

    if not (0 <= idx_x < n and 0 <= idx_y < n):
        raise ValueError(f"索引超出范围，该层特征图大小为 {n}x{n}")

    center_x = start + idx_x * j
    center_y = start + idx_y * j

    left = center_x - (r - 1) / 2
    right = center_x + (r - 1) / 2
    top = center_y - (r - 1) / 2
    bottom = center_y + (r - 1) / 2

    return center_x, center_y, r, (left, top, right, bottom)





def draw_rf_on_resized_input(
    image_path,
    layer_name,
    idx_x,
    idx_y,
    results,
    input_size=224,
    save_path=None,
    color="red",
    line_width=3,
    show_center=True,
):

    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_size, input_size))

    w, h = img.size

    cx, cy, rf_size, (left, top, right, bottom) = get_rf_box(results, layer_name, idx_x, idx_y)


    left_clip = max(0, left)
    top_clip = max(0, top)
    right_clip = min(w - 1, right)
    bottom_clip = min(h - 1, bottom)

    draw = ImageDraw.Draw(img)
    draw.rectangle([left_clip, top_clip, right_clip, bottom_clip], outline=color, width=line_width)

    if show_center:
        radius = 3
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill="yellow",
            outline="yellow"
        )

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"{layer_name} @ ({idx_x}, {idx_y})\n"
        f"RF={rf_size}, center=({cx:.1f}, {cy:.1f})"
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()





def draw_multi_rf_on_resized_input(
    image_path,
    query_list,
    results,
    input_size=224,
    save_path=None,
):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_size, input_size))

    w, h = img.size
    draw = ImageDraw.Draw(img)

    for item in query_list:
        layer = item["layer"]
        idx_x = item["x"]
        idx_y = item["y"]
        color = item.get("color", "red")
        line_width = item.get("line_width", 2)

        cx, cy, rf_size, (left, top, right, bottom) = get_rf_box(results, layer, idx_x, idx_y)

        left_clip = max(0, left)
        top_clip = max(0, top)
        right_clip = min(w - 1, right)
        bottom_clip = min(h - 1, bottom)

        draw.rectangle([left_clip, top_clip, right_clip, bottom_clip], outline=color, width=line_width)

        radius = 2
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=color,
            outline=color
        )

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Theoretical Receptive Fields on Resized Input (224x224)")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()





if __name__ == "__main__":
    image_path = "/path/to/example_tongue.png"
    imsize = 224
    mixed_pattern = (1, 2)

    results = compute_rf(imsize=imsize, mixed_pattern=mixed_pattern)

    print_stage_summary(results)





    draw_rf_on_resized_input(
        image_path=image_path,
        layer_name="db4_layer16_d2",
        idx_x=7,
        idx_y=7,
        results=results,
        input_size=224,
        save_path="rf_db4_center_on_resized_input.png",
        color="red",
    )







    queries = [
        {"layer": "db1_layer6",     "x": 28, "y": 28, "color": "blue"},
        {"layer": "db3_layer24",    "x": 7,  "y": 7,  "color": "green"},
        {"layer": "db4_layer16_d2", "x": 7,  "y": 7,  "color": "red"},
    ]

    draw_multi_rf_on_resized_input(
        image_path=image_path,
        query_list=queries,
        results=results,
        input_size=224,
        save_path="rf_multi_on_resized_input.png",
    )
