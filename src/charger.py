"""
Mask R-CNN
Train on the toy charger dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 charger.py train --dataset=/path/to/charger/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 charger.py train --dataset=/path/to/charger/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 charger.py train --dataset=/path/to/charger/dataset --weights=imagenet

    # Apply color splash to an image
    python3 charger.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 charger.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import xml.etree.ElementTree as ET
import tensorflow as tf
import keras

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

############################################################
#  Configurations
############################################################


class chargerConfig(Config):

    NAME = "charger"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + charger
    STEPS_PER_EPOCH = 1800
    DETECTION_MIN_CONFIDENCE = 0.9
    LEARNING_RATE = 0.0001
    NUM_POINTS = 7


############################################################
#  Dataset
############################################################

class ChargerDataset(utils.Dataset):

    def load_charger(self, dataset_dir, subset):
        """Load a subset of the charger dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("charger", 1, "charger")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        if subset == "val":
            dataset_dir = os.path.join(dataset_dir, 'val')
        annotations = os.listdir(os.path.join(dataset_dir, 'annotations'))

        # Add images
        for a in annotations:
            image_path = os.path.join(dataset_dir, 'images', a[:-4]+'.png')
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "charger",
                image_id=a,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                mask=os.path.join(dataset_dir, 'labels', a[:-4]+'_label.png'),
                annotation=os.path.join(dataset_dir, 'annotations', a))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a charger dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "charger":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = cv2.imread(info['mask'])
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def load_kp(self, image_id, num_points):

        info = self.image_info[image_id]
        # print(image_id)
        ann_fname = info['annotation']
        tree = ET.parse(ann_fname)
        root = tree.getroot()
        keypoints = []
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        # offset_x = int(root.find('offset_x').text)
        # offset_y = int(root.find('offset_y').text)
        for object in root.findall('object'):
            kps = object.find('keypoints')
            bbox = object.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            bw = (xmax - xmin) * w
            bh = (ymax - ymin) * h

            for i in range(num_points):
                kp = kps.find('keypoint' + str(i))
                # print("float(kp.find('x').text)", float(kp.find('x').text))
                # print("w", w)
                # print("bw", bw)
                # print("bh", bh)
                # print("", )
                keypoints.append(((float(kp.find('x').text) - xmin) * w / bw,
                                  (float(kp.find('y').text) - ymin) * h / bh))
                # keypoints.append((float(kp.find('x').text)/w, float(kp.find('y').text)/h))
            # print("keypoints", keypoints)
        return keypoints

    def load_yaw(self, image_id):

        info = self.image_info[image_id]
        ann_fname = info['annotation']
        tree = ET.parse(ann_fname)
        root = tree.getroot()
        theta = 0
        for obj in root.findall('object'):
            thetaxml = obj.find('theta')
            if thetaxml is not None:
                theta = float(thetaxml.text)
        return (theta + 1) / 2

    def load_bbox(self, image_id):

        info = self.image_info[image_id]
        ann_fname = info['annotation']
        tree = ET.parse(ann_fname)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.findall('object'):
            bndboxxml = obj.find('bndbox')
            if bndboxxml is not None:
                xmin = int(float(bndboxxml.find('xmin').text) * w)
                ymin = int(float(bndboxxml.find('ymin').text) * h)
                xmax = int(float(bndboxxml.find('xmax').text) * w)
                ymax = int(float(bndboxxml.find('ymax').text) * h)
        return (xmin, ymin, xmax, ymax)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "charger":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ChargerDataset()
    dataset_train.load_charger(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ChargerDataset()
    dataset_val.load_charger(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=300,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = np.zeros_like(image) #skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, gray, image).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path
    save = False
    if save:
        frozen_graph = freeze_session(keras.backend.get_session(),
                                      output_names=[out.op.name for out in model.keras_model.outputs])
        tf.train.write_graph(frozen_graph, "/root/share/tf/Keras", "frozen_inference_graph.pb", as_text=True)

    # Image or video?
    if image_path:
        images = os.listdir(os.path.join(image_path, "images_bright"))
        for im in images:
            # Run model detection and generate the color splash effect
            tree = ET.parse(os.path.join("/root/share/tf/dataset/Inea/6-point/val/", 'annotations', im[:-4] + ".txt"))
            root = tree.getroot()
            theta = 0
            for obj in root.findall('object'):
                theta = float(obj.find('theta').text)
            gt = theta * 180 / 3.14159
            print("GT yaw", gt)
            # image = skimage.io.imread(os.path.join(image_path, "images_bright", im))
            image = cv2.imread(os.path.join(image_path, "images_bright", im))
            r = model.detect([image], verbose=0)[0]
            splash = color_splash(image, r['masks'])
            kps = r['kp'][0][0]
            # print(kps)
            if len(r['rois'])==0:
                continue
            roi = r['rois'][0]
            print("yooooooł", (r['yaw'][0][0] * 2 - 1) * 180 / 3.14159)
            bw = roi[3]-roi[1]
            bh = roi[2]-roi[0]
            for i in range(config.NUM_POINTS):
                # cv2.circle (splash, (int(kps[i*2]*960), int(kps[i*2+1]*720)), 5, (0,0,255), -1)
                cv2.circle(splash, (int(kps[i * 2] * bw) + roi[1], int(kps[i * 2 + 1] * bh + roi[0])), 5, (0, 0, 255),
                           -1)
            cv2.imshow('lol', cv2.resize(splash, (1280, 960)))
            # attention = r['attention']
            # attention = (attention+abs(np.min(attention)))/(abs(np.min(attention))+abs(np.max(attention)))
            # attention = np.squeeze(attention.astype(np.uint8))
            # print(np.max(attention))
            # print(np.min(attention))
            # print(attention.shape)
            # attention = np.transpose(np.squeeze(attention), [2, 0, 1])
            # for i in range(5):
            #     cv2.imshow('att', attention[i])
            #     k = cv2.waitKey(0)

            k = cv2.waitKey(0)
            if k==ord('q'):
                exit(0)
            # Save output
            # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            # skimage.io.imsave(file_name, splash)
    elif video_path:
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        # vwriter = cv2.VideoWriter(file_name,
        #                           cv2.VideoWriter_fourcc(*'MJPG'),
        #                           fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # print(r)
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                cv2.imshow('lol', cv2.resize(splash, (1280, 960)))
                cv2.waitKey(0)
                # Add image to video writer
                # vwriter.write(splash)
                count += 1


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect chargers.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/root/share/tf/dataset/Inea/7-point",
                        help='Directory of the charger dataset')
    parser.add_argument('--weights', required=True,
                        metavar="./mask_rcnn_coco.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = chargerConfig()
    else:
        class InferenceConfig(chargerConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
        # weights_path = model.get_imagenet_inc_res_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)


    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
