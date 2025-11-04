import streamlit as st
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
import numpy as np
import cv2
import random
import os
import requests  # Added for downloading the model
# from io import BytesIO # No longer needed if writing to file

# --- Configuration ---

# 1. DEFINE MODEL URL:
#    This is the direct download link for your model.
MODEL_URL = "https://drive.google.com/uc?export=download&id=1K37vE8P3Oe0Gu3xp7-Ykj2knNh9_Tr_E"
#    The script will download the model to this local file path:
LOCAL_MODEL_PATH = "maskrcnn_model.pth"

# 2. DEFINE DEVICE:
#    Automatically uses GPU (cuda) if available, otherwise CPU.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. DEFINE CLASS NAMES:
#    IMPORTANT: This list MUST match the classes your custom model was trained on,
#    in the correct order. The default list for COCO (91 classes) is below.
#    If your model is different, you MUST update this list.
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# --- End Configuration ---


@st.cache_resource  # Cache the model to avoid reloading on each interaction
def load_model(model_url, local_path):
    """
    Downloads the custom model from a URL if not present locally,
    then loads it from the local file path.
    """
    # Check if the model file exists locally
    if not os.path.exists(local_path):
        st.info(f"Downloading model from URL... This may take a moment.")
        try:
            # Use requests to download the file
            response = requests.get(model_url, stream=True)
            response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)

            # Write the file to the local path
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            st.success(f"Model downloaded and saved to '{local_path}'.")

        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}")
            st.info("Please check the MODEL_URL or your internet connection.")
            st.stop()

    # --- Load the model from the local file ---

    # Instantiate the model architecture.
    # We use pretrained=False because we are about to load our own weights.
    # This assumes your custom model has the SAME architecture as the default
    # maskrcnn_resnet50_fpn (i.e., trained on 91 COCO classes).
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    try:
        # Load the custom weights (state dictionary)
        model.load_state_dict(torch.load(local_path, map_location=DEVICE))
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.warning("This can happen if your custom model's architecture (e.g., number of classes) "
                   "doesn't match the default 'maskrcnn_resnet50_fpn' used here. "
                   "If so, you must update the model creation code inside 'load_model()'.")
        st.stop()

    model.to(DEVICE)  # Move model to the selected device
    model.eval()      # Set model to evaluation mode
    return model

# Load the model
model = load_model(MODEL_URL, LOCAL_MODEL_PATH)


def random_color_masks(image):
    """
    Applies a random color to a binary mask.
    """
    colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    color = colors[random.randrange(0, len(colors))] # Use len(colors)
    r[image == 1], g[image == 1], b[image == 1] = color[0], color[1], color[2]
    colored_mask = np.stack([r, g, b], axis=2)
    return colored_mask


def get_prediction(img, threshold=0.5):
    """
    Gets predictions from the model for a single image.
    """
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(DEVICE)  # Move tensor to the correct device
    
    with torch.no_grad(): # Disable gradient calculation
        pred = model([img_tensor])

    # Move predictions back to CPU for numpy operations
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    
    # Check if any predictions exist
    if not pred_score:
        return [], [], []

    try:
        # Find the index of the last prediction with score > threshold
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    except IndexError:  # No predictions above threshold
        return [], [], []

    # Filter masks, boxes, and classes based on the threshold index
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class_indices = list(pred[0]['labels'].cpu().numpy())
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].detach().cpu().numpy())]

    # Handle the case of a single prediction (masks might not have batch dim)
    if len(masks.shape) == 2:
        masks = masks[np.newaxis, ...]

    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class_indices = pred_class_indices[:pred_t + 1]
    
    # Map class indices to names
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in pred_class_indices]

    return masks, pred_boxes, pred_class


def instance_segmentation_image(img, threshold=0.5, rect_th=3, text_size=1, text_th=2):
    """
    Applies instance segmentation masks, boxes, and labels to an image.
    """
    masks, boxes, pred_cls = get_prediction(img, threshold=threshold)
    
    if not masks.any(): # Check if masks is empty
        st.write("No objects detected above the threshold.")
        return np.array(img.convert("RGB")) # Return original image

    img_np = np.array(img.convert("RGB"))  # Convert PIL image to numpy array
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    for i in range(len(masks)):
        # Apply mask
        rgb_mask = random_color_masks(masks[i])
        img_np = cv2.addWeighted(img_np, 1, rgb_mask, 0.5, 0)
        
        # Draw bounding box
        pt1 = boxes[i][0]
        pt2 = boxes[i][1]
        cv2.rectangle(img_np, pt1, pt2, color=(0, 255, 0), thickness=rect_th)
        
        # Put label
        label = pred_cls[i]
        cv2.putText(img_np, label, pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)

    return cv2.cvtColor(img_np, cv2.COLOR_BGR_RGB)  # Convert back to RGB for Streamlit


# --- Streamlit App UI ---

st.title("Instance Segmentation with Custom Mask R-CNN")

# Add a warning about the class names
st.warning("Please ensure the `COCO_INSTANCE_CATEGORY_NAMES` list at the top of the script "
           "matches the classes your custom model was trained on.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    
    # Confidence threshold slider
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    
    with st.spinner("Detecting objects..."):
        segmented_img = instance_segmentation_image(image, threshold=threshold)
        st.image(segmented_img, caption="Segmented Image.", use_column_width=True)

