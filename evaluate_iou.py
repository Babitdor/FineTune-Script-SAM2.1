import numpy as np
import cv2
import torch
import os
import argparse
from skimage.measure import label, regionprops
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from tabulate import tabulate
from tqdm import tqdm
from peft import PeftModel
from sam2.build_sam import build_sam2
from scripts.pixelselection import select_pixels
from sam2.sam2_image_predictor import SAM2ImagePredictor


def compute_class_metrics(pred_mask, true_mask, class_id):
    """Compute metrics for a specific class"""
    pred_class = pred_mask == class_id
    true_class = true_mask == class_id

    if true_class.sum() == 0:
        return None, None, None, None

    iou = jaccard_score(true_class.flatten(), pred_class.flatten(), zero_division=0)
    precision = precision_score(
        true_class.flatten(), pred_class.flatten(), zero_division=0
    )
    recall = recall_score(true_class.flatten(), pred_class.flatten(), zero_division=0)
    f1 = f1_score(true_class.flatten(), pred_class.flatten(), zero_division=0)

    return iou, precision, recall, f1


def get_center_point(mask, class_id):
    """Get centroid of largest component for a class"""
    binary_mask = (mask == class_id).astype(np.uint8)
    labeled_img = label(binary_mask)
    if labeled_img.max() == 0:  # type: ignore
        return None
    props = regionprops(labeled_img)
    # Get largest component
    largest_prop = max(props, key=lambda x: x.area)
    center = largest_prop.centroid
    return (int(center[1]), int(center[0]))


def resize(img_path, mask_path, size=1024):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        raise ValueError(f"Error loading image or mask at {img_path} or {mask_path}")

    r = np.min([size / img.shape[1], size / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(
        mask,
        (int(mask.shape[1] * r), int(mask.shape[0] * r)),
        interpolation=cv2.INTER_NEAREST,
    )
    return img, mask


def process_image(
    image_path, mask_path, predictor, class_names, num_points=5, strategy="gaussian"
):
    """Process single image and return metrics"""
    # Read and resize image and mask
    img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
    ann_map = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

    # Resize image and mask
    r = min(1024 / img.shape[1], 1024 / img.shape[0])  # Scaling factor
    new_size = (int(img.shape[1] * r)), (int(img.shape[0] * r))
    img = cv2.resize(img, new_size)
    mat_map = cv2.resize(ann_map, new_size, interpolation=cv2.INTER_NEAREST)

    # Initialize results dictionary
    results = {"image": os.path.basename(image_path)}

    # Get unique class IDs in the mask (excluding background 0)
    present_class_ids = np.unique(mat_map)
    present_class_ids = present_class_ids[present_class_ids != 0]

    # If no objects in mask, return empty results
    if len(present_class_ids) == 0:
        for class_name in class_names[1:]:  # Skip background
            results[class_name] = {
                "iou": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "f1": "N/A",
            }
        return results

    # Set the image for SAM predictor
    predictor.set_image(img)

    # Initialize prediction mask
    pred_mask = np.zeros_like(mat_map)

    # Process each class in class_names (skip background at 0)
    for class_id, class_name in enumerate(class_names):
        if class_id == 0:  # Skip background
            continue

        # Check if this class exists in the mask
        if class_id not in present_class_ids:
            results[class_name] = {
                "iou": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "f1": "N/A",
            }
            continue

        # Select points for this class
        try:
            # Get multiple points (default 5)
            points = select_pixels(
                mat_map,
                class_id=class_id,
                num_points=num_points,
                selection_method=strategy,
            )

            # If no points found (shouldn't happen since we checked class exists)
            if len(points) == 0:
                results[class_name] = {
                    "iou": "N/A",
                    "precision": "N/A",
                    "recall": "N/A",
                    "f1": "N/A",
                }
                continue

            # Convert points to numpy array and create labels
            input_points = np.array(points)
            input_labels = np.ones(len(points))  # All points are positive

            # Get SAM prediction with multiple points
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_points,
                input_labels,
                box=None,
                mask_logits=None,
                normalize_coords=True,
            )
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None
            )

            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [
                feat_level[-1].unsqueeze(0)
                for feat_level in predictor._features["high_res_feats"]
            ]

            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )

            prd_masks = predictor._transforms.postprocess_masks(
                low_res_masks, predictor._orig_hw[-1]
            )
            prd_mask = (prd_masks > 0.5).float().cpu().numpy().squeeze()

            # Update prediction mask for this class
            pred_mask[prd_mask.astype(bool)] = class_id

            # Calculate metrics for this class
            iou, precision, recall, f1 = compute_class_metrics(
                pred_mask, mat_map, class_id
            )
            results[class_name] = {
                "iou": f"{iou:.4f}" if iou is not None else "N/A",
                "precision": f"{precision:.4f}" if precision is not None else "N/A",
                "recall": f"{recall:.4f}" if recall is not None else "N/A",
                "f1": f"{f1:.4f}" if f1 is not None else "N/A",
            }

        except Exception as e:
            print(f"Error processing class {class_name}: {str(e)}")
            results[class_name] = {
                "iou": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "f1": "N/A",
            }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAM model on SENSATION dataset"
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to directory containing 'images' and 'masks' folders (e.g., path/to/SENSATION/test/)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model checkpoint (e.g., models/fine_tuned_model.pt)",
    )
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="Path to model config file",
    )
    parser.add_argument("--num-pts", default=3, help="Select Number of Point Prompts")
    parser.add_argument(
        "--strategy", default="gaussian", help="Point selection strategy"
    )
    parser.add_argument("--use-lora", action="store_true", help="Using Lora Adapters")
    parser.add_argument(
        "--lora_adapters",
        help="Path to lora adapters",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = [
        "background",
        "road",
        "sidewalk",
        "Bikelane",
        "person",
        "car",
        "bicycle",
        "traffic sign (front)",
        "traffic light",
        "Obstacle",
    ]

    # Verify data directory structure
    image_dir = os.path.join(args.data_root, "images")
    mask_dir = os.path.join(args.data_root, "masks")
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        raise ValueError(f"Directory must contain 'images' and 'masks' folders")

    # Load model
    print(f"Loading model from {args.model}...")
    sam_model = build_sam2(
        config_file=args.config,
        ckpt_path=args.model,
        device=device,
    )

    if args.use_lora:  # You should have this flag in your args
        # Load the LoRA adapters
        sam_model = PeftModel.from_pretrained(  # type: ignore
            model=sam_model, model_id=args.lora_adapters, device=device
        )
        print("✅ LoRA adapters loaded successfully")
        print(f"Model after LoRA loading: {type(sam_model)}")

    predictor = SAM2ImagePredictor(sam_model)  # type: ignore

    # Get image paths
    image_files = sorted(
        f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))
    )
    print(f"Found {len(image_files)} images in {image_dir}")

    # Process all images
    class_stats = {
        name: {"iou": [], "precision": [], "recall": [], "f1": []}
        for name in class_names[1:]
    }

    for img_file in tqdm(image_files, desc="Evaluating images"):
        base_name = os.path.splitext(img_file)[0]
        mask_file = f"{base_name}.png"
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_file}")
            continue

        result = process_image(
            os.path.join(image_dir, img_file),
            mask_path,
            predictor,
            class_names,
            args.num_pts,
            args.strategy,
        )

        if result:
            for class_name in class_names[1:]:
                metrics = result[class_name]
                if metrics["iou"] != "N/A":
                    class_stats[class_name]["iou"].append(float(metrics["iou"]))
                    class_stats[class_name]["precision"].append(
                        float(metrics["precision"])
                    )
                    class_stats[class_name]["recall"].append(float(metrics["recall"]))
                    class_stats[class_name]["f1"].append(float(metrics["f1"]))

    # Calculate averages
    table_data = []
    for class_id, class_name in enumerate(class_names[1:], start=1):
        if class_stats[class_name]["iou"]:
            avg_iou = np.mean(class_stats[class_name]["iou"])
            avg_precision = np.mean(class_stats[class_name]["precision"])
            avg_recall = np.mean(class_stats[class_name]["recall"])
            avg_f1 = np.mean(class_stats[class_name]["f1"])
        else:
            avg_iou = avg_precision = avg_recall = avg_f1 = 0.0

        table_data.append(
            [
                class_id,
                class_name,
                f"{avg_iou:.4f}",
                f"{avg_precision:.4f}",
                f"{avg_recall:.4f}",
                f"{avg_f1:.4f}",
            ]
        )

    # Calculate overall means
    all_ious = [
        val for class_name in class_names[1:] for val in class_stats[class_name]["iou"]
    ]
    all_precisions = [
        val
        for class_name in class_names[1:]
        for val in class_stats[class_name]["precision"]
    ]
    all_recalls = [
        val
        for class_name in class_names[1:]
        for val in class_stats[class_name]["recall"]
    ]
    all_f1s = [
        val for class_name in class_names[1:] for val in class_stats[class_name]["f1"]
    ]

    mean_iou = np.mean(all_ious) if all_ious else 0.0
    mean_precision = np.mean(all_precisions) if all_precisions else 0.0
    mean_recall = np.mean(all_recalls) if all_recalls else 0.0
    mean_f1 = np.mean(all_f1s) if all_f1s else 0.0

    table_data.append(
        [
            "Mean",
            "-",
            f"{mean_iou:.4f}",
            f"{mean_precision:.4f}",
            f"{mean_recall:.4f}",
            f"{mean_f1:.4f}",
        ]
    )

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS".center(80))
    print("=" * 80)
    print(
        tabulate(
            table_data,
            headers=[
                "Class ID",
                "Class Name",
                "IoU",
                "Precision",
                "Recall",
                "F1 Score",
            ],
            tablefmt="pretty",
            stralign="right",
        )
    )

    # Save results to file
    output_file = os.path.join(args.data_root, "evaluation_results.txt")
    with open(output_file, "w") as f:
        f.write("Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Data root: {args.data_root}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Images evaluated: {len(image_files)}\n\n")
        f.write(
            tabulate(
                table_data,
                headers=[
                    "Class ID",
                    "Class Name",
                    "IoU",
                    "Precision",
                    "Recall",
                    "F1 Score",
                ],
                tablefmt="pretty",
                stralign="right",
            )
        )
        f.write("\n")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
