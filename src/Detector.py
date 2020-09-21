# from utils.app_utils import draw_boxes_and_labels
import tensorflow as tf
import numpy as np
import cv2
import time
from mrcnn.config import Config
from mrcnn import model as modellib
from YOLO.yolo import YOLO
from PIL import Image
from sklearn.cluster import DBSCAN

try:
    from cv2 import cv2
except ImportError:
    pass


class ChargerConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "charger"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + charger

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    # LEARNING_RATE = 0.001
    NUM_POINTS = 5


class Detector:

    def __init__(self, path_to_model, path_to_pole_model, path_to_model_bottom=None):
        np.set_printoptions(suppress=True)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        self.slice_size = (960, 960)
        # self.slice_size = (1280, 1280)
        self.offset = (0, 0)
        self.scale = 1.0
        # self.scale = 0.7
        self.charger_to_slice_ratio = 0.5
        self.detections = []
        self.best_detection = None
        # self.mask_reshaped = None
        self.bbox = None
        self.frame_shape = None
        self.moving_avg_image = None
        self.init_det = True
        self.bottom = False
        self.gt_kp = None
        self.gt_pose = None

        self.yolo = YOLO()

        self.pole_detection_graph = tf.Graph()
        with self.pole_detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(path_to_pole_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.pole_sess = tf.compat.v1.Session(graph=self.pole_detection_graph)

        self.config = ChargerConfig()
        self.config.display()

        model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir=path_to_model)

        weights_path = model.find_last()
        # weights_path = "/root/share/tf/Keras/31_08_heatmap/charger20200901T1045/mask_rcnn_charger_0002.h5"

        # Load weights
        # print("Loading weights ", weights_path)
        model.load_weights(weights_path, by_name=True)
        self.det_model = model

        if path_to_model_bottom is not None:
            self.config.NUM_POINTS = 4
            model_bottom = modellib.MaskRCNN(mode="inference", config=self.config,
                                             model_dir=path_to_model_bottom)

            weights_path_bottom = model_bottom.find_last()

            # Load weights
            print("Loading weights ", weights_path_bottom)
            model_bottom.load_weights(weights_path_bottom, by_name=True)
            self.det_model_bottom = model_bottom

        # self.rpn_box_predictor_features = None
        # self.class_predictor_weights = None

    def init_size(self, shape):
        self.frame_shape = shape
        self.moving_avg_image = np.full((self.config.NUM_POINTS, shape[0], shape[1]), 0.5, dtype=np.float64)

    def get_CNN_output(self, image_np):  # TODO: scale keypoints
        start_time = time.time()
        if self.bottom:
            r = self.det_model_bottom.detect([image_np], verbose=0)[0]
        else:
            r = self.det_model.detect([image_np], verbose=0)[0]
        # print("kp detection time:", time.time() - start_time)

        uncertainty = r['uncertainty'][0][0]
        kps = r['kp'][0][0]
        kps = np.transpose(kps, (2, 0, 1))
        absolute_kp = []
        heatmap_uncertainty = []

        for i, kp in enumerate(kps):
            # center = np.unravel_index(kp.argmax(), kp.shape)

            xmin, ymin, xmax, ymax = self.bbox
            alpha = 0.8
            # print("dtypes", kp.astype(np.float64).shape, self.moving_avg_image[i, ymin:ymax, xmin:xmax].shape)
            # kp = cv2.addWeighted(kp.astype(np.float64), alpha,
            #                      cv2.resize(self.moving_avg_image[i, ymin:ymax, xmin:xmax], self.slice_size), 1-alpha, 0.0)
            raw_kp = kp
            kp = kp / np.max(kp)
            # cv2.imshow("kp", kp)
            # cv2.waitKey(0)
            h, w = kp.shape
            # ret, kp = cv2.threshold(kp, 0.85, 1.0, cv2.THRESH_BINARY)
            ret, kp = cv2.threshold(kp, 0.2, 1.0, cv2.THRESH_BINARY)
            X = np.argwhere(kp == 1)
            if X.shape[0] == 0:
                absolute_kp.append((0, 0))
                heatmap_uncertainty = np.array([[np.inf, np.inf], [np.inf, np.inf]])
                continue
            clustering = DBSCAN(eps=3, min_samples=2)
            clustering.fit(X)
            cluster_scores = []
            unique_labels = np.unique(clustering.labels_)
            for id in np.unique(clustering.labels_):
                cluster = X[np.where(clustering.labels_ == id)]
                cluster_scores.append(np.sum(kp[cluster[:, 0], cluster[:, 1]]))
            cluster = X[np.where(clustering.labels_ == unique_labels[np.argmax(cluster_scores)])]
            mask = np.zeros_like(kp)
            mask[cluster[:, 0], cluster[:, 1]] = raw_kp[cluster[:, 0], cluster[:, 1]]

            # self.update_moving_avg(i, mask)
            heatmap_uncertainty.append(np.cov(cluster.T))
            if np.sum(mask) == 0:
                center = (0, 0)
            else:
                center = np.average(np.sum(mask, axis=1) * np.arange(w)) / np.sum(mask) * w, \
                         np.average(np.sum(mask, axis=0) * np.arange(h)) / np.sum(mask) * h
            absolute_kp.append(
                ((center[1] * self.scale[0] + self.bbox[0]),  # * self.frame_shape[1] / self.slice_size[1],
                 (center[0] * self.scale[1] + self.bbox[1])))  # * self.frame_shape[0] / self.slice_size[0]))


        self.best_detection = dict(keypoints=absolute_kp, uncertainty=uncertainty,
                                   heatmap_uncertainty=np.array(heatmap_uncertainty))

    def update_moving_avg(self, i, mask):
        xmin, ymin, xmax, ymax = self.bbox
        overlay = np.zeros(self.frame_shape, dtype=np.float64)
        mk = cv2.resize(mask, (xmax - xmin, ymax - ymin))
        ret, mk = cv2.threshold(mk, 0.5, 1.0, cv2.THRESH_BINARY)
        mk = cv2.dilate(mk, np.ones((10, 10), np.uint8), iterations=10)
        mk = cv2.blur(mk, (20, 20))
        overlay[ymin:ymax, xmin:xmax] = cv2.resize(mk, (xmax - xmin, ymax - ymin))
        cv2.accumulateWeighted(overlay, self.moving_avg_image[i], 0.5)

    def detect(self, frame, gt_pose, gt_kp):
        self.gt_kp = gt_kp
        self.gt_pose = gt_pose
        self.detections = []
        self.best_detection = None
        self.init_detection(frame)
        # if len(self.detections) == 0:
        if self.best_detection is None:
            print("No detections")
        return

    def get_slice(self, frame, offset=None):
        # return cv2.resize(frame, self.slice_size)
        if offset is not None:
            return frame[offset[0]:offset[0] + self.slice_size[0],
                   offset[1]:offset[1] + self.slice_size[1]]
        return frame[self.offset[0]:self.offset[0] + self.slice_size[0],
                     self.offset[1]:self.offset[1] + self.slice_size[1]]

    def init_detection(self, frame):
        start_time = time.time()
        small_frame = cv2.resize(frame, (self.slice_size[1], self.slice_size[0]))

        ymin, xmin, ymax, xmax = self.yolo.detect_image(Image.fromarray(small_frame))
        # self.scale = ((xmax - xmin) / self.slice_size[0], (ymax - ymin) / self.slice_size[1])
        # self.bbox = [xmin, ymin, xmax, ymax]
        ymin = int(ymin * frame.shape[0] / self.slice_size[0])
        ymax = int(ymax * frame.shape[0] / self.slice_size[0])
        xmin = int(xmin * frame.shape[1] / self.slice_size[1])
        xmax = int(xmax * frame.shape[1] / self.slice_size[1])
        margin = (xmax - xmin) * 0.02, (ymax - ymin) * 0.02

        xmin -= int(6 * margin[0])
        ymin -= int(margin[1])
        xmax += int(margin[0])
        ymax += int(margin[1])
        self.scale = ((xmax - xmin) / self.slice_size[0], (ymax - ymin) / self.slice_size[1])
        self.bbox = [xmin, ymin, xmax, ymax]

        try:
            frame = cv2.resize(frame[ymin:ymax, xmin:xmax], self.slice_size)
            cv2.imshow("yolo", frame)
        except cv2.error as e:
            print(e.msg)
            return

        # xmin, ymin, xmax, ymax = self.detect_pole(small_frame)
        # print("pole detection time:", time.time() - start_time)

        self.get_CNN_output(frame)

    def detect_pole(self, small_frame):
        image_np_expanded = np.expand_dims(small_frame, axis=0)
        image_tensor = self.pole_detection_graph.get_tensor_by_name('image_tensor:0')

        boxes = self.pole_detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.pole_detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.pole_detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.pole_detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = self.pole_sess.run([boxes, scores, classes, num_detections],
                                                                      feed_dict={image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        boxes_dict = []
        for box, score in zip(boxes, scores):
            # print("box", box)
            ymin, xmin, ymax, xmax = box
            boxes_dict.append(dict(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax, score=score))
        # print("boxes", boxes)
        # print("scores", scores)

        detections = []
        for idx in range(len(boxes_dict)):
            abs_xmin = int(boxes_dict[idx]['xmin'] * self.frame_shape[1])
            abs_ymin = int(boxes_dict[idx]['ymin'] * self.frame_shape[0])
            abs_xmax = np.min((int(boxes_dict[idx]['xmax'] * self.frame_shape[1]), self.frame_shape[1]))
            abs_ymax = np.min((int(boxes_dict[idx]['ymax'] * self.frame_shape[0]), self.frame_shape[0]))
            if abs_xmax <= abs_xmin or abs_ymax <= abs_ymin:
                continue
            detection = dict(rel_rect=boxes_dict[idx], score=boxes_dict[idx]['score'],
                             abs_rect=np.array([abs_xmin, abs_ymin, abs_xmax, abs_ymax]))
            detections.append(detection)

        if len(detections) > 0:
            best_detection = sorted(detections, key=lambda k: k['score'], reverse=True)[0]
            return best_detection['abs_rect']
