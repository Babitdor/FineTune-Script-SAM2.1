import argparse
import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scripts.pixelselection import select_pixels
from scripts.segmentation import Segmentator
from PIL import Image, ImageDraw
import pandas as pd
from sklearn.metrics import jaccard_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiClassMaskGenerator:
    def __init__(
        self,
        lora_adapter_path,
        sam2_model_path: str,
        sam2_config_path: str,
        onnx_model_path: str,
        class_colors_csv: str,
        pixel_points: int = 5,
        pixel_step: int = 1,
        iteration: int = 3,
        pixel_select: str = "gaussian",
        good_iou: float = 0.80,
        min_iou: float = 0.30,
        use_lora=False,
    ):

        self.use_lora = use_lora
        self.pixel_points = pixel_points
        self.iteration = iteration
        self.pixel_select = pixel_select
        self.pixel_step = pixel_step
        self.good_iou = good_iou
        self.min_iou = min_iou
        self.lora_adapter_path = lora_adapter_path
        self.sam2 = self._load_sam2(sam2_model_path, sam2_config_path)
        self.segmentator = Segmentator(
            input_width=800,
            input_height=640,
            model_path=onnx_model_path,
            csv_color_path=class_colors_csv,
        )

    def _load_sam2(self, model_path: str, config_path: str):
        sam2_model = build_sam2(config_path, model_path, device)  # type: ignore

        if self.use_lora:  # You should have this flag in your args
            # Load the LoRA adapters
            sam2_model = PeftModel.from_pretrained(  # type: ignore
                model=sam2_model, model_id=self.lora_adapter_path, device=device
            )
            print("✅ LoRA adapters loaded successfully")

        predictor = SAM2ImagePredictor(sam2_model)

        return predictor

    def generate_mask(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Main pipeline for multi-class mask generation with adaptive refinement

        Args:
            image: Input image as numpy array

        Returns:
            Tuple containing:
            - initial_mask: The initial segmentation mask
            - final_mask: The refined mask after SAM2 processing
            - mean_iou: Average IoU score across all classes
        """
        # Get initial class segmentation
        initial_mask = self._get_initial_segmentation(image)
        final_mask = np.zeros_like(initial_mask)
        iou_scores = []
        class_stats = {}  # Track best results per class

        for class_id in np.unique(initial_mask):
            if class_id == 0:  # Skip background
                continue

            best_iou = 0.0
            best_mask = None
            current_iteration = 0
            pixel_points = self.pixel_points  # Start with default points

            while current_iteration < self.iteration and best_iou < self.good_iou:
                # Refine the current class with current parameters
                class_mask = self._refine_class(
                    image=image,
                    initial_mask=initial_mask,
                    class_id=class_id,
                    pixel_points=pixel_points,
                )

                # Calculate IoU for this refinement
                current_iou = self._calculate_iou(class_mask, initial_mask, class_id)

                print(
                    f"Class {class_id} - Iter {current_iteration}: IoU={current_iou:.2f}, Points={pixel_points}"
                )

                # Update best result if improved
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_mask = class_mask

                # Adaptive parameter adjustment
                if current_iou < self.min_iou and current_iteration > 1:
                    # Poor performance - try more points
                    pixel_points += self.pixel_step
                    print(f"Low IoU - increasing points to {pixel_points}")

                current_iteration += 1

            # Store best result for this class
            if best_mask is not None:
                final_mask = np.maximum(final_mask, best_mask)
                iou_scores.append(best_iou)
                class_stats[class_id] = {
                    "iou": best_iou,
                    "iterations": current_iteration,
                    "points_used": pixel_points,
                }

        # Calculate mean IoU
        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

        # Print summary
        print("\nRefinement Summary:")
        for class_id, stats in class_stats.items():
            print(
                f"Class {class_id}: IoU={stats['iou']:.2f} "
                f"(iters={stats['iterations']}, points={stats['points_used']})"
            )
        print(f"Mean IoU: {mean_iou:.2f}")

        return initial_mask, final_mask, mean_iou

    def _get_initial_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Get initial multi-class segmentation using SENSATION model"""
        if image.shape[2] == 3 and image.dtype == "uint8":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        batch_output = self.segmentator.inference([image])
        initial_mask = batch_output[0]

        if initial_mask.shape != image.shape[:2]:
            initial_mask = cv2.resize(
                initial_mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        return initial_mask.astype(np.uint8)

    def _refine_class(
        self, image: np.ndarray, initial_mask: np.ndarray, class_id: int, pixel_points
    ) -> np.ndarray:
        """Refine one class using SAM2 with automatic prompts"""
        # Get prompts from initial mask
        try:
            points = select_pixels(
                segmentation_mask=initial_mask,
                class_id=class_id,
                num_points=pixel_points,
                selection_method=self.pixel_select,
            )
        except ValueError as e:
            # Fallback: use all available pixels for this class
            xs, ys = np.where(initial_mask == class_id)
            points = np.stack([xs, ys], axis=1) if len(xs) > 0 else np.empty((0, 2))
            print(
                f"Warning: {e} Using {len(xs)} available pixels for class {class_id}."
            )
        # SAM2 prediction
        self.sam2.set_image(image)
        masks, scores, logits = self.sam2.predict(
            point_coords=points,
            point_labels=np.ones(len(points)),
            multimask_output=True,
        )

        # Sort in descending order based on scores
        sorted_indices = np.argsort(-scores)

        # Reorder arrays based on sorted scores
        masks = masks[sorted_indices]
        scores = scores[sorted_indices]
        logits = logits[sorted_indices]

        # Select the best (highest score) mask and corresponding logits
        mask = masks[0]
        mask_input = logits[0][
            None, :, :
        ]  # Mask Embeddings, just in case, if useful for other tasks if required.

        return mask.astype(np.uint8) * class_id

    def _calculate_iou(
        self, predicted_mask: np.ndarray, truth_mask: np.ndarray, class_id: int
    ) -> float:
        """
        Calculate the Intersection over Union (IoU) for a given class ID.
        Resizes the truth mask if its size does not match the predicted mask size.

        Args:
            predicted_mask (np.ndarray): The predicted mask as a NumPy array.
            truth_mask (np.ndarray): The ground truth mask as a NumPy array.
            class_id (int): The class ID for which IoU is to be calculated.

        Returns:
            float: The IoU for the specified class ID.
        """
        # Remove extra dimensions from the predicted mask if present
        if predicted_mask.ndim > 2:
            predicted_mask = predicted_mask.squeeze()

        # Check if the masks have the same size
        if predicted_mask.shape != truth_mask.shape:
            truth_mask_pil = Image.fromarray(truth_mask)
            truth_mask_resized = truth_mask_pil.resize(
                (predicted_mask.shape[1], predicted_mask.shape[0]), Image.NEAREST  # type: ignore
            )
            truth_mask = np.array(truth_mask_resized)

        # Ensure masks are the same shape
        if predicted_mask.shape != truth_mask.shape:
            raise ValueError(
                f"Masks are still mismatched in size: {predicted_mask.shape} vs {truth_mask.shape}"
            )

        # Convert the masks into binary masks for the given class ID
        predicted_binary = (predicted_mask == class_id).astype(int).flatten()
        truth_binary = (truth_mask == class_id).astype(int).flatten()

        # Check consistent lengths
        if len(predicted_binary) != len(truth_binary):
            raise ValueError(
                f"Inconsistent lengths: {len(predicted_binary)} vs {len(truth_binary)}"
            )

        # Calculate IoU using scikit-learn's jaccard_score
        iou = jaccard_score(truth_binary, predicted_binary)

        return iou  # type: ignore

    def panoptic_to_rgb(
        self,
        panoptic_seg: torch.Tensor,
        colormap: Optional[
            Dict[Union[int, np.integer], Union[List[int], np.ndarray]]
        ] = None,
    ) -> np.ndarray:
        """
        Convert a panoptic segmentation mask to an RGB image.

        This function converts a 2D panoptic segmentation mask (where each pixel's value represents a segment ID)
        into a color-coded RGB image. If no colormap is provided, each unique segment is assigned a random color.
        The output image is a NumPy array with shape (H, W, 3) and dtype uint8.

        Args:
            panoptic_seg (torch.Tensor):
                A 2D tensor of shape (H, W) representing the panoptic segmentation mask, where each pixel's value is a segment ID.
            colormap (Optional[Dict[int, Union[List[int], np.ndarray]]]):
                An optional dictionary mapping segment IDs to RGB colors. Each color should be a list or array of 3 integers (e.g., [R, G, B]).
                If not provided, random colors will be generated for each unique segment.

        Returns:
            np.ndarray:
                An RGB image (NumPy array) with shape (H, W, 3) and dtype np.uint8.
        """
        # Convert the panoptic segmentation tensor to a NumPy array
        if isinstance(panoptic_seg, torch.Tensor):
            panoptic_np = panoptic_seg.cpu().numpy()
        else:
            panoptic_np = np.asarray(panoptic_seg)
        h, w = panoptic_np.shape
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

        # Find unique segment IDs in the mask
        unique_ids = np.unique(panoptic_np)

        # If no colormap is provided, generate random colors for each segment
        if colormap is None:
            colormap = {}
            for seg_id in unique_ids:
                # Generate a random color (R, G, B) where each channel is in the range [0, 255]
                colormap[seg_id] = np.random.randint(0, 256, size=3, dtype=np.uint8)

        # Assign colors to each pixel in the RGB image based on its segment id
        for seg_id in unique_ids:
            mask = panoptic_np == seg_id
            rgb_img[mask] = colormap[seg_id]

        return rgb_img

    def save_comparison_plot(
        self,
        image: np.ndarray,
        truth_mask: np.ndarray,
        predicted_mask: np.ndarray,
        output_folder: str,
        filename: str,
    ):
        """
        Creates and stores a comparison plot of an image, its ground truth mask, and the predicted mask.

        :param image: The input image as a numpy array (H, W, C) or (H, W).
        :param truth_mask: The ground truth segmentation RGB mask as a numpy array (H, W, 3).
        :param predicted_mask: The predicted segmentation RGB mask as a numpy array (H, W, 3).
        :param output_folder: The folder where the generated image will be saved.
        :param filename: The name of the file to save the image as (including the extension, e.g., 'output.png').
        """
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Create the plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the input image
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[0].axis("off")

        # Plot the ground truth mask
        axes[1].imshow(truth_mask)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        # Plot the predicted mask
        axes[2].imshow(predicted_mask)
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

        # Adjust layout
        plt.tight_layout()

        # Save the figure to the specified output folder
        save_path = os.path.join(output_folder, filename)
        plt.savefig(save_path, dpi=300)

        # Close the plot to free resources
        plt.close()


def process_images(
    input_path: str,
    output_path: str,
    generator: MultiClassMaskGenerator,
    visualize: bool = False,
):
    """Process all images in a folder"""
    os.makedirs(output_path, exist_ok=True)

    for img_file in tqdm(list(Path(input_path).glob("*.*")), desc="Processing images"):
        if img_file.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        image = cv2.imread(str(img_file))
        if image is None:
            print(f"Warning: Could not read image {img_file}, skipping.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        initial_mask, mask, iou_score = generator.generate_mask(image)

        print(f"IOU SCORE for {img_file} :-: {iou_score}")

        if visualize:
            truth_rgb = generator.panoptic_to_rgb(initial_mask)  # type: ignore
            pred_rgb = generator.panoptic_to_rgb(mask)  # type: ignore

            generator.save_comparison_plot(
                image,
                truth_rgb,
                pred_rgb,
                output_path,
                f"{img_file.stem}_comparison.png",
            )

        output_file = Path(output_path) / f"{img_file.stem}_mask.png"
        cv2.imwrite(str(output_file), mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input images")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--sam2_model", type=str, required=True, help="Path to SAM2 model"
    )
    parser.add_argument(
        "--sam2_config", type=str, required=True, help="Path to SAM2 config"
    )
    parser.add_argument(
        "--onnx_model", type=str, required=True, help="Path to ONNX model"
    )

    parser.add_argument(
        "--class_csv", type=str, required=True, help="Path to class colors CSV"
    )
    parser.add_argument(
        "--use-lora", type=bool, default=False, help="Using Lora Adapters"
    )
    parser.add_argument(
        "--lora_adapters",
        help="Path to lora adapters",
    )
    parser.add_argument(
        "--pixel_points", type=int, default=5, help="Number of pixel points per class"
    )
    parser.add_argument(
        "--iteration", type=int, default=10, help="Number of refinement iterations"
    )
    parser.add_argument(
        "--pixel_select",
        type=str,
        default="gaussian",
        choices=[
            "centroid",
            "fps",
            "shuffle",
            "fibonacci",
            "laplace",
            "gaussian",
            "kmeans",
        ],
        help="Pixel selection method",
    )
    parser.add_argument(
        "--pixel_step", type=int, default=1, help="Pixel step increase for poor masks"
    )
    # Quality Assurance parameters
    parser.add_argument(
        "--good_iou", type=float, default=0.80, help="Good IOU Score Requirement"
    )
    parser.add_argument(
        "--min_iou",
        type=float,
        default=0.30,
        help="Minimum acceptable IOU Score Requirement",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Generate comparison visualizations",
    )

    args = parser.parse_args()

    generator = MultiClassMaskGenerator(
        lora_adapter_path=args.lora_adapters,
        sam2_model_path=args.sam2_model,
        sam2_config_path=args.sam2_config,
        onnx_model_path=args.onnx_model,
        class_colors_csv=args.class_csv,
        pixel_points=args.pixel_points,
        pixel_step=args.pixel_step,
        iteration=args.iteration,
        pixel_select=args.pixel_select,
        good_iou=args.good_iou,
        min_iou=args.min_iou,
        use_lora=args.use_lora,
    )

    process_images(args.input, args.output, generator, args.visualize)
    print(f"Generated masks saved to {args.output}")
