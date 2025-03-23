import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision import models
from ultralytics import YOLO

def load_fine_tuned_midas():
    """
    Loads the MiDaS_small model for monocular depth estimation and sets it to evaluation mode.
    
    Returns:
        tuple: A tuple containing the MiDaS model (torch.nn.Module) and a torchvision transform pipeline.
    """
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((384, 384)),
        T.ToTensor()
    ])
    return model, transform

def load_yolo_model():
    """
    Loads the YOLOv8 object detection model.

    Returns:
        YOLO: An instance of the YOLOv8 model.
    """
    return YOLO("yolov8s.pt")

def load_segmentation_model():
    """
    Loads the DeepLabV3 ResNet101 model pre-trained on COCO for semantic segmentation.

    Returns:
        tuple: A tuple containing the segmentation model and the preprocessing transform.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, transform

def detect_objects(model, image_path):
    """
    Runs object detection on the given image using the specified YOLO model.

    Args:
        model (YOLO): YOLOv8 model instance.
        image_path (str): Path to the input image.

    Returns:
        list: YOLO detection results.
    """
    return model(image_path)

def segment_objects(model, transform, img):
    """
    Applies semantic segmentation to generate a binary mask from the image.

    Args:
        model (torch.nn.Module): Pre-trained segmentation model.
        transform (callable): Transformation pipeline for preprocessing.
        img (np.ndarray): Input RGB image.

    Returns:
        np.ndarray: Resized segmentation mask as a 2D array.
    """
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)["out"]
    mask = output.argmax(1).byte().cpu().numpy()[0]
    return cv2.resize(mask, (img.shape[1], img.shape[0]))

def draw_detections(image, results):
    """
    Draws bounding boxes and class labels on the image based on detection results.

    Args:
        image (np.ndarray): Input RGB image.
        results (list): List of detection result objects from YOLO.

    Returns:
        np.ndarray: Image with bounding boxes and labels drawn.
    """
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            label = f"{result.names[class_id]}: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return image

def predict_depth(model, transform, img, mask=None):
    """
    Predicts the depth map of an image using a MiDaS model, with optional masking.

    Args:
        model (torch.nn.Module): MiDaS depth estimation model.
        transform (callable): Transformation pipeline for the input image.
        img (np.ndarray): Input RGB image.
        mask (np.ndarray, optional): Segmentation mask to filter the depth map.

    Returns:
        tuple:
            depth (np.ndarray): Raw depth map.
            depth_norm (np.ndarray): Normalized depth map (0 to 1).
            stats (dict): Dictionary with center, average, and top-left region depth values.
    """
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        depth = model(tensor).squeeze().cpu().numpy()
    depth = cv2.resize(depth, (img.shape[1], img.shape[0]))
    if mask is not None:
        depth *= (mask > 0)

    dmin, dmax = depth.min(), depth.max()
    depth_norm = (depth - dmin) / (dmax - dmin)

    h, w = depth.shape
    center = float(depth[h // 2, w // 2])
    avg = float(np.mean(depth))
    region = float(np.mean(depth[:h // 2, :w // 2]))

    stats = {
        "center": center,
        "average": avg,
        "region_top_left": region
    }

    return depth, depth_norm, stats

def warp_image_right(img, depth):
    """
    Generates a right-eye stereo view by warping the input image using optical flow derived from depth.

    Args:
        img (np.ndarray): Original RGB image.
        depth (np.ndarray): Normalized depth map.

    Returns:
        np.ndarray: Warped right-eye view image.
    """
    h, w = img.shape[:2]
    disp = 1.0 / (depth + 1e-6)
    disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(disp, disp, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mx, my = np.meshgrid(np.arange(w), np.arange(h))
    mapx = (mx + flow[..., 0]).astype(np.float32)
    mapy = (my + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)

def run_inference(image_path, output_dir, model_type="midas"):
    """
    Executes the full 3D inference pipeline:
    - Object detection
    - Semantic segmentation
    - Depth estimation
    - Right-eye stereo image synthesis

    Saves all visual outputs and returns their file paths and depth statistics.

    Args:
        image_path (str): Path to the input RGB image.
        output_dir (str): Directory to store the generated output images.
        model_type (str, optional): Model type flag (currently unused).

    Returns:
        dict: Dictionary containing paths to output images and depth statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    midas, midas_transform = load_fine_tuned_midas()
    yolo = load_yolo_model()
    seg_model, seg_transform = load_segmentation_model()

    detections = detect_objects(yolo, image_path)
    detection_img = draw_detections(rgb.copy(), detections)
    seg_mask = segment_objects(seg_model, seg_transform, rgb)
    depth, norm_depth, stats = predict_depth(midas, midas_transform, rgb, seg_mask)
    right_view = warp_image_right(rgb, norm_depth)

    base = os.path.splitext(os.path.basename(image_path))[0]
    paths = {
        "detection": os.path.join(output_dir, f"{base}_detection.jpg"),
        "segmentation": os.path.join(output_dir, f"{base}_segmentation.jpg"),
        "depth": os.path.join(output_dir, f"{base}_depth.jpg"),
        "right": os.path.join(output_dir, f"{base}_right.jpg"),
        "depth_stats": stats
    }

    cv2.imwrite(paths["detection"], cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR))
    plt.imsave(paths["segmentation"], seg_mask, cmap="gray")
    plt.imsave(paths["depth"], norm_depth, cmap="plasma")
    cv2.imwrite(paths["right"], cv2.cvtColor(right_view, cv2.COLOR_RGB2BGR))

    return paths
