"""This the SENSATION segmentation class to use it in your project"""

import csv
import cv2
import numpy as np
import onnxruntime
import torch


class Segmentator:
    def __init__(
        self,
        input_width: int = 544,
        input_height: int = 544,
        model_path: str = None,  # type: ignore
        csv_color_path: str = None,  # type: ignore
    ):
        self.input_width = input_width
        self.input_height = input_height
        self.model_path = model_path
        if model_path is not None:
            self.onnx_session = onnxruntime.InferenceSession(model_path)
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_labels_from_csv(csv_color_path)

        # Define rgb for sidewalk in mask
        self.sidewalk_rgb = [255, 0, 0]

    def preprocess_image(self, image_array):
        image_array = image_array.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        return image_array

    def inference(self, images):
        size = (self.input_width, self.input_height)
        processed_images = []

        # Resize and preprocess each image in the batch
        for image in images:
            image = cv2.resize(image, size)
            image = image.astype(np.float32)
            image = self.preprocess_image(image)
            processed_images.append(image)

        # Convert the list of processed images to a numpy array and then to a tensor
        batch_images = np.array(processed_images)
        x_tensor = torch.from_numpy(batch_images).permute(0, 3, 1, 2).to(self.DEVICE)

        # Run the ONNX model for segmentation
        ort_inputs = {
            self.onnx_session.get_inputs()[0]
            .name: x_tensor.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        }
        ort_outputs = self.onnx_session.run(None, ort_inputs)

        # Process the output for each image in the batch
        predicted_outputs = [np.argmax(output, axis=0) for output in ort_outputs[0]]  # type: ignore

        return predicted_outputs

    def mask_to_rgb(self, mask):
        """Map grayscale values in a mask to RGB values using the provided color map.

        :param mask: 2D numpy array representing a grayscale image segmentation mask
        :return: 3D numpy array representing an RGB image
        """
        # Validate mask input
        if mask.ndim != 2:
            raise ValueError("Input mask must be a 2D array.")

        # Create an RGB image initialized with zeros
        rgb_image = np.zeros((*mask.shape, 3), dtype=np.uint8)

        # Validate and prepare the color map
        unique_gray_values = set(np.unique(mask))
        max_gray = mask.max()
        color_map_array = np.zeros((max_gray + 1, 3), dtype=np.uint8)
        for gray_value, color in self.color_map.items():
            if gray_value in unique_gray_values:
                color_map_array[gray_value] = color

                # Apply the color map to the mask
                rgb_image = color_map_array[mask]

        return rgb_image

    def load_labels_from_csv(self, csv_color_path: str):
        """
        Loads the class label ID and RGB values from a CSV file into a color map.

        :param csv_color_path: Path to the CSV file containing class labels and colors.
        """
        color_map = {}

        try:
            with open(csv_color_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)

                # Check for required columns in the CSV
                required_columns = {"class_label", "red", "green", "blue"}
                if not required_columns.issubset(reader.fieldnames):  # type: ignore
                    raise ValueError(
                        f"CSV file must contain columns: {', '.join(required_columns)}"
                    )

                for row in reader:
                    # Parse and validate class_label and RGB values
                    try:
                        class_label = int(row["class_label"])
                        red = int(row["red"])
                        green = int(row["green"])
                        blue = int(row["blue"])

                        if not (
                            0 <= red <= 255 and 0 <= green <= 255 and 0 <= blue <= 255
                        ):
                            raise ValueError(
                                f"RGB values must be in the range 0-255. Found: {red}, {green}, {blue}"
                            )

                        color_map[class_label] = [red, green, blue]
                    except ValueError as e:
                        print(f"Skipping row due to error: {e}")

                self.color_map = color_map

        except FileNotFoundError:
            print(f"Error: File not found at {csv_color_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_sidewalk_rgb(self):
        return self.sidewalk_rgb


if __name__ == "__main__":
    image_path = "test/1000032755.jpg"
    save_path = "test/1000032755.png"
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    model_path = "model_weights/DeepLabV3Plus_resnet50.onnx"
    segmentator = Segmentator(model_path=model_path)
    mask = segmentator.inference(image)
    mask_rgb = segmentator.mask_to_rgb(mask)
    mask_rgb = cv2.resize(mask_rgb, (width, height))
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, mask_bgr)
