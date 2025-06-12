import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import tempfile
import io
from streamlit_drawable_canvas import st_canvas
from scripts.dataloader import SENSATIONDataset
from sam2 import build_sam, sam2_image_predictor

# Set up Streamlit page
st.set_page_config(layout="wide")
st.title("SAM 2.1 Segmentation Demo")
st.write("Upload an image or select from test dataset to generate segmentation masks")

# Configuration
MODEL_PATH = "models/final_model.pt"
CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_b+.yaml"
TEST_MANIFEST = "dataset/SENSATION_DS_Preprocessed/testing/manifest_testing.csv"

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    mode = st.radio(
        "Input Mode",
        ["Upload Image", "Test Dataset Sample"],
        index=0,
        help="Choose between uploading an image or using test dataset samples",
    )

    st.markdown("---")
    st.subheader("Model Parameters")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Threshold for mask binarization",
    )

    st.markdown("---")
    st.subheader("Visualization")
    mask_alpha = st.slider(
        "Mask Transparency",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        help="Adjust mask overlay transparency",
    )


# Load model (cached)
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # "Loading model on {device}..."

    try:
        sam_model = build_sam.build_sam2(
            checkpoint=None,
            config_file=CONFIG_PATH,
        )
        sam_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        sam_model.to(device)
        sam_model.eval()

        predictor = sam2_image_predictor.SAM2ImagePredictor(sam_model)
        return predictor, device
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()


# Load test dataset (cached)
@st.cache_resource
def load_test_dataset():
    try:
        dataset = SENSATIONDataset(manifest_path=TEST_MANIFEST)
        return dataset
    except Exception as e:
        st.error(f"Failed to load test dataset: {str(e)}")
        return None


# Initialize model and dataset
predictor, device = load_model()
test_dataset = load_test_dataset()


def process_image(image, input_points=None, input_labels=None):
    """Process an image through SAM 2.1 model"""
    try:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure image is in correct format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Set image in predictor
        predictor.set_image(image)

        # Prepare prompts
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            (
                input_points
                if input_points is not None
                else np.array([[image.shape[1] // 2, image.shape[0] // 2]])
            ),
            input_labels if input_labels is not None else np.array([1]),
            box=None,
            mask_logits=None,
            normalize_coords=True,
        )

        # Get embeddings
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels),
            boxes=None,
            masks=None,
        )

        # Check if batched mode
        batched_mode = unnorm_coords.shape[0] > 1

        # Get features
        high_res_features = [
            feat_level[-1].unsqueeze(0)
            for feat_level in predictor._features["high_res_feats"]
        ]

        # Predict masks
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Post-process masks
        prd_masks = predictor._transforms.postprocess_masks(
            low_res_masks, predictor._orig_hw[-1]
        )
        prd_masks = (prd_masks > confidence_threshold).float()

        return prd_masks, prd_scores

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        return None, None


def display_results(image, gt_mask=None, pred_mask=None, score=None):
    """Display input image and results"""
    cols = st.columns(2 if gt_mask is None else 3)

    with cols[0]:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    if gt_mask is not None:
        with cols[1]:
            st.subheader("Ground Truth")
            st.image(gt_mask, use_container_width=True, clamp=True)

    result_col = cols[2] if gt_mask is not None else cols[1]
    with result_col:
        st.subheader("Predicted Mask")

        # Create overlay
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        if pred_mask is not None:
            ax.imshow(
                pred_mask.squeeze().cpu().numpy(), alpha=mask_alpha, cmap="viridis"
            )
        ax.axis("off")
        st.pyplot(fig)

        if score is not None:
            st.write(f"Confidence score: {score.item():.2f}")


# Main application logic
if mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload an image for segmentation",
    )

    if uploaded_file is not None:
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            # Open and resize image if needed
            image = Image.open(temp_path).convert("RGB")
            max_size = 1024

            # Resize if image is larger than max_size in either dimension
            if image.width > max_size or image.height > max_size:
                image.thumbnail(
                    (max_size / 1.5, max_size / 1.5), Image.Resampling.LANCZOS
                )

            # Initialize session state for points if not exists
            if "points" and "labels" not in st.session_state:
                st.session_state.points = []
                st.session_state.labels = []
                st.session_state.original_size = image.size
                st.session_state.display_size = image.size

            # Create two columns
            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Add Point Prompts")

                # Point type selector
                point_type = st.radio(
                    "Point Label",
                    ["Positive (1) - Include", "Negative (0) - Exclude"],
                    horizontal=True,
                    key="point_type_selector",
                )

                # Create canvas for point selection
                if "canvas_key" not in st.session_state:
                    st.session_state.canvas_key = 0

                canvas_result = st_canvas(
                    fill_color=(
                        "rgba(0, 255, 0, 0.3)"
                        if point_type.startswith("Positive")
                        else "rgba(255, 0, 0, 0.3)"
                    ),
                    stroke_width=10,
                    background_image=image,
                    height=image.height,
                    width=image.width,
                    drawing_mode="point",
                    key="canvas_{st.session_state.canvas_key}",
                    update_streamlit=True,
                )

            # Handle new points from canvas
            if (
                canvas_result.json_data is not None
                and "objects" in canvas_result.json_data
            ):
                new_points = []
                new_labels = []
                for obj in canvas_result.json_data["objects"]:
                    if obj["type"] == "circle":  # Point
                        x, y = obj["left"], obj["top"]
                        color = obj.get("fill", "")
                        if "0, 255, 0" in color:  # green
                            label = 1
                        else:
                            label = 0

                        new_points.append([x, y])
                        new_labels.append(label)
                st.session_state.points = new_points
                st.session_state.labels = new_labels

            with col2:
                st.subheader("Segmentation Result")

                if st.button("Generate Mask") and st.session_state.points:
                    with st.spinner("Processing image..."):
                        # Convert points to numpy arrays
                        input_points = np.array(st.session_state.points)
                        input_labels = np.array(st.session_state.labels)
                        # Process image with points
                        pred_mask, score = process_image(
                            image, input_points, input_labels
                        )

                    # Display results
                    if pred_mask is not None:
                        # Create visualization
                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.imshow(image)

                        # Show mask overlay
                        ax.imshow(
                            pred_mask.squeeze().cpu().numpy(),
                            alpha=mask_alpha,
                            cmap="viridis",
                        )

                        # Plot points with color coding
                        for point, label in zip(
                            st.session_state.points, st.session_state.labels
                        ):
                            color = "green" if label == 1 else "red"
                            ax.scatter(
                                point[0],
                                point[1],
                                c=color,
                                s=100,
                                edgecolors="white",
                                linewidths=1,
                                label="Positive" if label == 1 else "Negative",
                            )

                        ax.axis("off")

                        # Add legend
                        handles, labels = ax.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))  # Remove duplicates
                        ax.legend(
                            by_label.values(),
                            by_label.keys(),
                            loc="upper right",
                            bbox_to_anchor=(1.1, 1),
                        )

                        st.pyplot(fig)
                        st.success(
                            f"Segmentation complete! Confidence: {score.item():.2f}"
                        )

                        # Option to download mask
                        mask_img = Image.fromarray(
                            (pred_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
                        )
                        buf = io.BytesIO()
                        mask_img.save(buf, format="PNG")
                        buf.seek(0)
                        st.download_button(
                            "Download Mask",
                            buf,
                            file_name="mask.png",
                            mime="image/png",
                        )
                elif not st.session_state.points:
                    st.warning("Please add points to the image first")

            # Clean up
            os.remove(temp_path)

        except Exception as e:
            st.error(f"Error processing uploaded image: {str(e)}")

else:  # Test Dataset Sample
    if test_dataset is None:
        st.warning("Test dataset not available")
    else:
        sample_idx = st.number_input(
            "Sample Index",
            min_value=0,
            max_value=len(test_dataset) - 1,
            value=0,
            step=1,
            help="Select which test sample to use",
        )

        if st.button("Load Sample"):
            try:
                # Get sample from dataset
                sample = test_dataset[sample_idx]
                image = sample["image"]
                gt_mask = sample["mask"]
                input_points = sample["input_points"]
                input_labels = sample["input_labels"]

                # Convert tensors to numpy if needed
                if isinstance(image, torch.Tensor):
                    image = image.permute(1, 2, 0).cpu().numpy()
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)

                if isinstance(gt_mask, torch.Tensor):
                    gt_mask = gt_mask.squeeze().cpu().numpy()

                # Process image
                with st.spinner("Processing sample..."):
                    pred_mask, score = process_image(image, input_points, input_labels)

                # Display results
                if pred_mask is not None:
                    display_results(image, gt_mask, pred_mask, score)

            except Exception as e:
                st.error(f"Error processing test sample: {str(e)}")
