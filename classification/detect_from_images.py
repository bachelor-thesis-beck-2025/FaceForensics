import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm

from network.models import model_selection
from dataset.transform import xception_default_data_transforms
import glob
import csv

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def preprocess_image(image, cuda=True, device=None):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR)
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if device is not None:
        preprocessed_image = preprocessed_image.to(device)
    elif cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True, device=None):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda, device)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output


def process_image_file(image_path, model, face_detector, cuda=True, visualize=False, vis_dir=None, device=None):
    image = cv2.imread(image_path)
    if image is None:
        return None

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    if len(faces) == 0:
        return {'path': image_path, 'has_face': 0, 'label': None, 'probs': None}

    # Use largest face
    face = max(faces, key=lambda f: (f.right()-f.left()) * (f.bottom()-f.top()))
    x, y, size = get_boundingbox(face, width, height)
    cropped_face = image[y:y+size, x:x+size]

    with torch.no_grad():
        prediction, output = predict_with_model(cropped_face, model, cuda=cuda, device=device)
        probs = output.detach().cpu().numpy()[0].tolist()

    label = 'fake' if prediction == 1 else 'real'

    if visualize and vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        x0, y0 = face.left(), face.top()
        w, h = face.right() - x0, face.bottom() - y0
        color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
        cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), color, 2)
        cv2.putText(image, f"{probs}->{label}", (x0, y0 + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, 2)
        out_path = os.path.join(vis_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, image)

    return {'path': image_path, 'has_face': 1, 'label': label, 'probs': probs}


def test_on_images(images_path, model_path, output_csv=None, vis_dir=None, cuda=True, root_folder=False):
    # Device
    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    # Build model
    model, *_ = model_selection(modelname='xception', num_out_classes=2)
    model = model.to(device)

    if model_path is not None:
        loaded_ok = False
        # First, safe attempt: load weights only
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
            if isinstance(state, dict):
                model.load_state_dict(state)
                loaded_ok = True
        except Exception:
            loaded_ok = False

        if not loaded_ok:
            # Fallback: allowlist TransferModel and load pickled model/state
            try:
                from torch.serialization import add_safe_globals
                from network.models import TransferModel
                add_safe_globals([TransferModel])
                loaded = torch.load(model_path, map_location=device, weights_only=False)
                if isinstance(loaded, torch.nn.Module):
                    model = loaded.to(device)
                elif isinstance(loaded, dict):
                    model.load_state_dict(loaded)
                else:
                    raise RuntimeError('Unsupported checkpoint type')
                loaded_ok = True
            except Exception as e:
                raise e

        print(f'Model loaded from {model_path}')
    else:
        print('No model path provided. Using randomly initialized model.')


    model.eval()

    face_detector = dlib.get_frontal_face_detector()

    # Collect image files
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    files = []
    if os.path.isdir(images_path):
        for ext in exts:
            if root_folder:
                # Recursive search in all subdirectories
                files.extend(glob.glob(os.path.join(images_path, '**', ext), recursive=True))
            else:
                # Only search in the immediate directory
                files.extend(glob.glob(os.path.join(images_path, ext)))
    else:
        files = [images_path]

    results = []
    for img in tqdm(files):
        res = process_image_file(img, model, face_detector, cuda=cuda, visualize=vis_dir is not None, vis_dir=vis_dir, device=device)
        if res:
            results.append(res)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'has_face', 'label', 'probs'])
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--images_path', '-i', type=str, required=True, help='Path to image file or folder of images')
    p.add_argument('--model_path', '-m', type=str, default=None, help='Path to Xception .p model file')
    p.add_argument('--output_csv', type=str, default=None, help='Optional CSV to write results')
    p.add_argument('--vis_dir', type=str, default=None, help='Optional folder to save annotated images')
    p.add_argument('--cuda', action='store_true', help='Enable CUDA')
    p.add_argument('--root_folder', action='store_true', help='Search recursively in all subdirectories for images')
    args = p.parse_args()

    test_on_images(args.images_path, args.model_path, args.output_csv, args.vis_dir, args.cuda, args.root_folder)
