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
import math
import numpy as np
import skimage.draw
import cv2
import xml.etree.ElementTree as ET
import tensorflow as tf
import keras
from scipy.spatial.transform import Rotation as R
from PoseEstimator import PoseEstimator

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class chargerConfig(Config):

    NAME = "charger"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + charger
    STEPS_PER_EPOCH = 1200
    DETECTION_MIN_CONFIDENCE = 0.9
    LEARNING_RATE = 0.00001
    NUM_POINTS = 4

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
            if not a.startswith("1_15"):
                continue
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
                keypoints.append([[(float(kp.find('x').text) - xmin) * w / bw,
                                   (float(kp.find('y').text) - ymin) * h / bh],
                                  [(float(kp.find('x').text) * w),
                                   (float(kp.find('y').text) * h)]])
        return keypoints

    def load_im_meta(self, image_id):
        info = self.image_info[image_id]
        ann_fname = info['annotation']
        tree = ET.parse(ann_fname)
        root = tree.getroot()
        off_x = int(root.find('offset_x').text)
        off_y = int(root.find('offset_y').text)
        object = root.find('object')
        cm = object.find('camera_matrix')
        fx = float(cm.find('fx').text)
        fy = float(cm.find('fy').text)
        cx = float(cm.find('cx').text)
        cy = float(cm.find('cy').text)
        return np.array([fx / 6000.0, fy / 6000.0, cx / 6000.0, cy / 6000.0, off_x / 6000.0, off_y / 6000.0]).astype(
            np.float32)

    def load_3d_points(self):
        return np.array(
            [(-0.32, 0.255, 0.65), (-0.075, 0.0, 0.65), (0.075, 0.0, 0.65), (0.32, 0.255, 0.65)]).astype(np.float32)

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


def train(model, epochs):
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

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = np.ones_like(image) * 255  # skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, gray, image).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect(model, image_path=None):
    dataset_val = ChargerDataset()
    dataset_val.load_charger(image_path, "train")
    dataset_val.prepare()
    save = False
    if save:
        frozen_graph = freeze_session(keras.backend.get_session(),
                                      output_names=[out.op.name for out in model.keras_model.outputs])
        tf.train.write_graph(frozen_graph, "/root/share/tf/Keras", "frozen_inference_graph.pb", as_text=True)

    # images = os.listdir(os.path.join(image_path, "images_bright"))
    errors_e2e = np.array([0.0, 0.0, 0.0])
    errors_pnp = np.array([0.0, 0.0, 0.0, 0.0])
    cnt = 0
    cnt0 = 0
    cnt1 = 0
    cnt2 = 0
    for id in dataset_val.image_ids:
        im = dataset_val.image_info[id]["id"]
        # Run model detection and generate the color splash effect
        print(im)
        try:
            tree = ET.parse(os.path.join("/root/share/tf/dataset/Inea/6-point/", 'annotations', im))
            root = tree.getroot()
            obj = root.find('object')
            pose = obj.findall('pose')[1]
            pos = pose.find('position')
            gt_tvec = np.array([float(pos.find('x').text), float(pos.find('y').text), float(pos.find('z').text)])
            ori = pose.find('orientation')
            gt_quat = [float(ori.find('x').text), float(ori.find('y').text), float(ori.find('z').text),
                       float(ori.find('w').text)]
            gt_rot = R.from_quat(gt_quat)
            print("gt_rot", gt_rot.as_euler('xyz', degrees=True))
            print("gt_tvec", gt_tvec)
        except:
            pass
        image = dataset_val.load_image(id)  # cv2.imread(os.path.join(image_path, "images", im[:-4]+".png"))
        image = cv2.cvtColor(np.array(image, dtype=np.float32), cv2.COLOR_BGR2RGB)

        pose_im_meta = dataset_val.load_im_meta(id)
        r = model.detect([image], pose_im_meta, verbose=0)[0]
        splash = color_splash(image, r['masks'])
        kps = r['kp'][0][0]
        pred_pose = r['pose'][0, 0]
        if len(r['rois']) == 0:
            print("no detections")
            continue

        camera_matrix = np.array([[4996.73451, 0, 2732.95188],
                                  [0, 4992.93867, 1890.88113],
                                  [0, 0, 1]])

        dist = pred_pose[0]  # * 100
        center_u = (r['rois'][0, 3] - r['rois'][0, 1]) / 2 + r['rois'][0, 1] + pose_im_meta[4] * 6000
        center_v = r['rois'][0, 0] + pose_im_meta[5] * 6000
        # print("center_u, center_v", center_u, center_v)
        x = (center_u - camera_matrix[0, 2]) / camera_matrix[0, 0] * dist
        y = (center_v - camera_matrix[1, 2]) / camera_matrix[1, 1] * dist
        pred_tvec = np.array([x, y, dist])
        print("pred_tvec", pred_tvec)
        pred_rvec = pred_pose[1:] * 4
        # pred_rot = R.from_rotvec(pred_rvec)
        # print("pred_rot", pred_rot.as_euler('xyz', degrees=True))

        # pred_tvec = pred_pose[:3] * 100
        # pred_tvec[:2] = pred_tvec[:2] - 50
        # # pred_rvec = np.array([0,0,0], dtype=np.float32)

        if gt_tvec[2] < 20:
            errors_e2e[0] += abs(gt_tvec[2] - pred_tvec[2])
            cnt0 += 1
        if gt_tvec[2] < 40 and gt_tvec[2] > 20:
            errors_e2e[1] += abs(gt_tvec[2] - pred_tvec[2])
            cnt1 += 1
        if gt_tvec[2] > 40:
            errors_e2e[2] += abs(gt_tvec[2] - pred_tvec[2])
            cnt2 += 1
        # errors_e2e += abs(gt_tvec - pred_tvec)#, abs(pred_rot.as_euler('xyz', degrees=True)[2] - gt_rot.as_euler('xyz', degrees=True)[2]))  # TODO: clean this up
        # errors += np.append(gt_tvec - pred_tvec, pred_rot.as_euler('xyz', degrees=True)[2] - gt_rot.as_euler('xyz', degrees=True)[2])  # TODO: clean this up
        cnt += 1
        print("errors_e2e:", errors_e2e[0] / cnt0, errors_e2e[1] / cnt1, errors_e2e[2] / cnt2)
        # # print("pose_im_meta", pose_im_meta)


        # camera_matrix = np.array([[1929.14559, 0, 1924.38974],
        #                           [0, 1924.07499, 1100.54838],
        #                           [0, 0, 1]])
        distortion = np.array([-0.11286, 0.11138, 0.00195, -0.00166, 0.00000]).astype(np.float64)
        object_points = np.array(
            [(-0.32, 0.255, 0.65), (-0.075, 0.0, 0.65), (0.075, 0.0, 0.65), (0.32, 0.255, 0.65)]).astype(np.float64)
        # out_points = np.squeeze(
        #     cv2.projectPoints(object_points, pred_rvec, pred_tvec, camera_matrix, distortion)[0])

        out_points = object_points + pred_tvec
        # object_points = pred_rot.apply(object_points)
        # print("out_points", out_points)
        u = camera_matrix[0, 0] * out_points[:, 0] / out_points[:, 2] + camera_matrix[0, 2] - pose_im_meta[4] * 6000
        v = camera_matrix[1, 1] * out_points[:, 1] / out_points[:, 2] + camera_matrix[1, 2] - pose_im_meta[5] * 6000
        # print(out_points.shape)
        for idx, pt in enumerate(zip(u, v)):  # enumerate(out_points):
            # print(pt)
            # print(pose_im_meta[4:])
            # pt = int(pt[0]-pose_im_meta[4]*6000), int(pt[1]-pose_im_meta[5]*6000)
            pt = int(pt[0]), int(pt[1])
            # print("proj points", pt)
            try:
                cv2.circle(splash, (pt), 5, (0, 255, 0), -1)
            except OverflowError:
                print("Reprojection failed. Int overflow")
        # print('rois', r['rois'])
        cv2.rectangle(splash, (r['rois'][0, 1], r['rois'][0, 0]), (r['rois'][0, 3], r['rois'][0, 2]), (222, 222, 222),
                      5)
        out_path = os.path.join("/root/share/tf/dataset/sup-res", im[:-4] + ".png")
        w = r['rois'][0, 2] - r['rois'][0, 0]
        h = r['rois'][0, 3] - r['rois'][0, 1]

        pe = PoseEstimator(camera_matrix)
        kps = np.reshape(kps, (4, 2))
        out_kp = np.zeros_like(kps)
        out_kp[:, 1] = kps[:, 0] * h + r['rois'][0, 1] + pose_im_meta[5] * 6000
        out_kp[:, 0] = kps[:, 1] * w + r['rois'][0, 0] + pose_im_meta[4] * 6000
        # kps = np.transpose(kps, )
        # print("kps", kps)
        pnp_tvec, pnp_rvec = pe.calc_PnP_pose(out_kp)
        pnp_rvec = R.from_rotvec(pnp_rvec)
        pnp_tvec = np.squeeze(pnp_tvec)
        print("pnp_tvec", pnp_tvec)
        errors_pnp += np.append(abs(gt_tvec - pnp_tvec), abs(
            pnp_rvec.as_euler('xyz', degrees=True)[2] - gt_rot.as_euler('xyz', degrees=True)[2]))  # TODO: clean this up
        print("errorsPnP:", errors_pnp / cnt)

        sup_res_img = image[r['rois'][0, 0] - int(0.1 * w):r['rois'][0, 2] + int(0.1 * w),
                      r['rois'][0, 1] - int(0.1 * h):r['rois'][0, 3] + int(0.1 * h)]
        if sup_res_img.shape[0] * sup_res_img.shape[1] > 100 * 100:
            cv2.imwrite(out_path, sup_res_img)
        cv2.imshow('lol', splash)  # cv2.resize(splash, (1280, 960)))

        k = cv2.waitKey(10)
        if k == ord('q'):
            exit(0)

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
        assert args.image or args.dataset, \
            "Provide --image or --dataset to apply color splash"

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
        train(model, 150)
    elif args.command == "splash":
        detect(model, image_path=args.dataset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
