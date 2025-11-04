%%writefile app.py
import streamlit as st
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
import numpy as np
import cv2
import random

@st.cache_resource # Cache the model to avoid reloading on each interaction
def load_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    # Check for GPU and move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()
    return model

model = load_model()

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

def random_color_masks(image):
    colors = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180], [250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    color = colors[random.randrange(0, 10)]
    r[image==1], g[image==1], b[image==1] = color[0], color[1], color[2]
    colored_mask = np.stack([r,g,b], axis=2)
    return colored_mask

def get_prediction(img, threshold=0.5):
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    pred = model([img_tensor])
    # Move predictions back to CPU for numpy operations
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    try:
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    except IndexError: # No predictions above threshold
        return [], [], []

    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].detach().cpu().numpy())]

    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]

    return masks, pred_boxes, pred_class

def instance_segmentation_image(img, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    masks, boxes, pred_cls = get_prediction(img, threshold=threshold)
    img_np = np.array(img.convert("RGB")) # Convert PIL image to numpy array
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV

    for i in range(len(masks)):
        rgb_mask = random_color_masks(masks[i])
        img_np = cv2.addWeighted(img_np, 1, rgb_mask, 0.5, 0)
        pt1 = boxes[i][0]
        pt2 = boxes[i][1]
        cv2.rectangle(img_np, pt1, pt2, color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img_np, pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)

    return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Convert back to RGB for Streamlit display

st.title("Instance Segmentation with Mask R-CNN")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Detecting objects...")

    # You can add sliders or number inputs for threshold, rect_th, etc. here
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

    segmented_img = instance_segmentation_image(image, threshold=threshold)

    st.image(segmented_img, caption="Segmented Image.", use_column_width=True)
