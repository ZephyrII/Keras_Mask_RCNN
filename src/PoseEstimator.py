import cv2

# try:
#     from cv2 import cv2
# except ImportError:
#     pass
import math
# import apriltag
import numpy as np
import rospy
import matplotlib.pyplot as plt


class PoseEstimator:

    def __init__(self, camera_matrix):
        self.gps_data_file = '/root/share/tf/dataset/warsaw/gps_trajectory_extended_13_44.txt'
        self.camera_matrix = camera_matrix
        scale = 1.0
        self.charger_height = 0.33 * scale
        self.charger_width = 0.77 * scale
        self.gps_pose = None
        self.magic_pose_data = None
        self.magic_kp_data = None
        self.PnP_pose_data = None
        self.plot_data = []
        self.ma_z = []
        self.magic_dist = None
        self.magic_kp_dist = None
        self.PnP_dist = None
        self.GPS_dist = None
        self.posecnn_dist = None
        # with open(self.gps_data_file, newline='') as csvfile:
        #     gps_reader = csv.reader(csvfile)
        #     row = next(gps_reader)
        #     self.start_time = float(row[0])

    def calc_error(self, pose_msg, mask_coords, abs_coords, points, pitch, poseCNN=None):
        self.read_GPS_pose(pose_msg)
        self.GPS_dist = self.calc_dist(float(self.gps_pose[0]), float(self.gps_pose[2]))
        self.calc_magic_pose(mask_coords, abs_coords, pitch, points)
        if self.magic_pose_data is None:
            return
        self.magic_dist = self.calc_dist(self.magic_pose_data[0], self.magic_pose_data[2])
        if poseCNN is not None:
            self.posecnn_dist = np.sqrt(
                poseCNN['pose'][0] * poseCNN['pose'][0] + poseCNN['pose'][2] * poseCNN['pose'][2])
            posecnn_error = abs(self.posecnn_dist - self.GPS_dist)
        if points is not None:
            # self.calc_magic_pose_kp(points)
            # self.magic_kp_dist = self.calc_dist(self.magic_kp_data[0], self.magic_kp_data[2])
            self.calc_PnP_pose(points)
            if self.PnP_pose_data is not None:
                self.PnP_dist = self.calc_dist(float(self.PnP_pose_data[0]), float(self.PnP_pose_data[2]))
                PnP_error = abs(self.PnP_dist - self.GPS_dist)
                magic_error = abs(self.magic_dist - self.GPS_dist)
                # magic_kp_error = abs(self.magic_kp_dist - self.GPS_dist)
                self.plot_data.append([self.magic_dist, self.GPS_dist, self.PnP_dist, magic_error,
                                       PnP_error])
                rospy.logdebug('gps, mask_error [m], PnP_error [m]: %f, %f, %f', self.GPS_dist, magic_error, PnP_error)

                labels = ['DGPS', 'Mask', 'SolvePnP', 'Pose']
                # plt.rcParams.update({'font.size': 18})
                plt.clf()
                plt.subplot(211)
                plt.grid()
                np_plot = np.array(self.plot_data)
                plt.gca().invert_xaxis()
                plt.plot(np_plot[:, 1], np_plot[:, 1], 'g')
                plt.plot(np_plot[:, 1], np_plot[:, 0], 'r')
                plt.plot(np_plot[:, 1], np_plot[:, 2], 'b')
                plt.xlabel('distance [m]')
                plt.ylabel('distance [m]')
                # plt.yticks(np.arange(0, 120, step=10))
                # plt.xticks(np.arange(5, 75, step=5))
                # plt.ylim(5, 110)
                # plt.plot(self.plot_data[:, 3], 'm')
                plt.legend(labels)
                plt.subplot(212)
                plt.gca().invert_xaxis()
                # plt.plot(np_plot[:, 1], np_plot[:, 3], 'r')
                plt.plot(np_plot[:, 1], np_plot[:, 4], 'b')
                # plt.plot(np_plot[:, 3], 'r')
                # plt.plot(np_plot[:, 4], 'b')
                # plt.plot(self.plot_data[:, 1], self.plot_data[:, 6], 'g')
                plt.ylim(0.0, .5)
                plt.xlabel('distance [m]')
                plt.ylabel('error [m]')
                # plt.yticks(np.arange(0, 70, step=10))
                # plt.xticks(np.arange(0, 70, step=10))
                plt.legend(["Mask", "PnP"])
                plt.pause(0.05)
                plt.draw()

    def calc_dist(self, x, z):
        return math.sqrt(x ** 2 + z ** 2)
    #
    # def apriltag_pose(self, img):
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     options = apriltag.DetectorOptions(families='tag36h11',
    #                                        border=1,
    #                                        nthreads=4,
    #                                        quad_decimate=1.0,
    #                                        quad_blur=0.0,
    #                                        refine_edges=True,
    #                                        refine_decode=True,
    #                                        refine_pose=True,
    #                                        debug=True,
    #                                        quad_contours=False)
    #     detector = apriltag.Detector(options)
    #     result = detector.detect(img_gray)
    #     if len(result) > 0:
    #         pose, e0, e1 = detector.detection_pose(result[0], (
    #             self.camera_matrix[0, 0], self.camera_matrix[1, 1], self.camera_matrix[0, 2], self.camera_matrix[1, 2]),
    #                                                1.0)
    #         return pose
    #     else:
    #         return []

    def read_GPS_pose(self, pose_msg):
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        z = pose_msg.pose.position.z
        self.gps_pose = [x, y, z]

    def calc_magic_pose(self, mask_coords, abs_coords, pitch, points):
        if mask_coords.shape[0] == 0:
            return None
        x1, y1, x2, y2 = abs_coords
        width = np.max(mask_coords[:, 1]) - np.min(mask_coords[:, 1])
        height = np.max(mask_coords[:, 0]) - np.min(mask_coords[:, 0])

        # width = abs(points[0][0] - points[3][0])
        # height = abs(points[0][1] - points[1][1])
        # self.charger_height = 0.25
        # self.charger_width = 0.64

        # print('height', height)
        # print('width', width)
        # est_z = (0.77 * self.camera_matrix[0, 0]) / width

        # est_z_old = (self.charger_height * self.camera_matrix[0, 0]) / height
        # est_x_old = est_z_old * (x1 + (width / 2) - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        # self.magic_old_data = [est_x_old, 0, est_z_old]

        ####        EXPERIMENTAL        ###
        gamma = pitch
        beta = math.atan2(abs(y2 - self.camera_matrix[1, 2]), self.camera_matrix[1, 1])
        alpha = beta - math.atan2(abs(y1 - self.camera_matrix[1, 2]), self.camera_matrix[1, 1])

        # est_z = self.camera_matrix[1, 1] * charger_height * math.cos(beta + gamma) / (height * math.cos(beta) *
        #         math.cos(gamma) * math.cos(gamma)*(1 + math.tan(gamma) * math.tan(gamma + beta - alpha)))
        est_z = self.camera_matrix[1, 1] / height * self.charger_height * math.cos(beta + gamma) / math.cos(
            beta) * math.cos(beta - alpha + gamma) / math.cos(beta - alpha)

        ####        EXPERIMENTAL        ###

        est_x = est_z * (x1 + (width / 2) - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        proj_width = width * self.calc_dist(est_z, est_x) / self.camera_matrix[0, 0]
        # print('proj_width', proj_width)
        if proj_width > self.charger_width:
            proj_width = self.charger_width

        theta = np.arccos(proj_width / self.charger_width)

        # print("Theta:", np.degrees(theta))
        # print('est_z, est_z_old, diff', est_z, est_z_old, est_z - est_z_old)
        self.magic_pose_data = [est_x, 0, est_z, theta]
        # print("Magic:", self.magic_pose_data[0], self.magic_pose_data[2])
        # print("GT:", self.gps_pose[0], self.gps_pose[2])
        # print("Magic:", self.calc_dist(self.magic_pose_data[0], self.magic_pose_data[2]))

    def calc_magic_pose_kp(self, points):
        print(points)
        width = abs(points[0][0] - points[3][0])
        height = abs(points[0][1] - points[1][1])
        print('width', width)
        # est_z = (0.64 * self.camera_matrix[1, 1]) / width
        gamma = math.radians(10)
        beta = math.atan2(abs(points[3][0] - self.camera_matrix[1, 2]), self.camera_matrix[1, 1])
        alpha = beta - math.atan2(abs(points[0][0] - self.camera_matrix[1, 2]), self.camera_matrix[1, 1])

        # est_z = charger_height/(math.tan(beta+gamma)-math.tan(gamma+beta-alpha))
        # est_z = self.camera_matrix[1, 1] * charger_height * math.cos(beta + gamma) / (height * math.cos(beta) * math.cos(
        #     gamma) * math.cos(gamma)*(1 + math.tan(gamma) * math.tan(gamma + beta - alpha)))
        est_z = self.camera_matrix[1, 1] / height * self.charger_height * math.cos(beta + gamma) / math.cos(
            beta) * math.cos(beta - alpha + gamma) / math.cos(beta - alpha)
        # est_z = (0.33 * self.camera_matrix[0, 0]) / height
        est_x = est_z * (points[0][0] + (width / 2) - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        # print('est_z', est_z, est_x)
        self.magic_kp_data = [est_x, 0, est_z]

    def get_charger_cluster(self, img_roi):
        K = 2
        Z = img_roi.reshape((-1, 3)) / 255
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # get center cluster id
        center - np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape(img_roi.shape)
        cv2.imshow('res', res2)
        shaped_label = label.reshape(img_roi.shape[:2])
        coords_zero = np.argwhere(shaped_label == 0)
        coords_one = np.argwhere(shaped_label == 1)
        std_zero = np.std(coords_zero[:, 0]) + np.std(coords_zero[:, 1])
        std_one = np.std(coords_one[:, 0]) + np.std(coords_one[:, 1])
        if std_one > std_zero:
            return coords_zero
        else:
            return coords_one

    def extract_keypoints(self, frame, charger_contour, roi_pos, roi_shape, charger_coords):
        # poly = np.squeeze(cv2.approxPolyDP(charger_contour, 3, closed=True))
        # # print(poly)
        # # cv2.drawContours(frame, [poly], 0, (0, 255, 255), 1)
        #
        # # Walk from left
        # current_point_x = np.min(charger_contour[:, 0])
        # current_point_y = np.max(charger_contour[charger_contour[:, 0] == current_point_x, 1])
        # prev_point = (current_point_x, current_point_y)
        # current_point = prev_point
        # while current_point[1] >= prev_point[1]:
        #     current_point_x += 1
        #     prev_point = current_point
        #     current_point_y = np.max(charger_contour[charger_contour[:, 0] == current_point_x, 1])
        #     current_point = (current_point_x, current_point_y)
        # cv2.circle(frame, prev_point, 1, (0, 255, 255), 2)
        #
        # # Walk from right
        # current_point_x = np.max(charger_contour[:, 0])
        # current_point_y = np.max(charger_contour[charger_contour[:, 0] == current_point_x, 1])
        # prev_point = (current_point_x, current_point_y)
        # current_point = prev_point
        # while current_point[1] >= prev_point[1]:
        #     current_point_x -= 1
        #     prev_point = current_point
        #     current_point_y = np.max(charger_contour[charger_contour[:, 0] == current_point_x, 1])
        #     current_point = (current_point_x, current_point_y)
        # cv2.circle(frame, prev_point, 1, (0, 255, 255), 2)
        #
        # # cv2.waitKey(0)
        #
        # for pt in poly:
        #     cv2.circle(frame, tuple(pt), 1, (255, 0, 0), 1)

        # min_max = []
        # bottom_line = np.array(bottom_line)
        # min_max.append(argrelextrema(charger_contour, np.greater))
        # min_max.append(argrelextrema(charger_contour, np.less))
        #
        # print(min_max)
        # for pt in min_max:
        #     pt = np.squeeze(pt)
        #     if pt.shape[1]>0:
        #         print(pt)
        #         cv2.circle(frame, pt, 8, (0, 0, 255), 2)

        imagePoints = []
        try:
            # Get bottom point
            coords_bott_left = np.squeeze(charger_coords[np.argwhere(charger_coords[:, 1] < roi_shape[1] / 3)])
            coords_bott_right = np.squeeze(charger_coords[np.argwhere(charger_coords[:, 1] > roi_shape[1] / 3)])
            coords_right = np.squeeze(charger_coords[np.argwhere(charger_coords[:, 0] > roi_shape[0] / 3)])

            circle_center_bl = (coords_bott_left[np.argmax(coords_bott_left[:, 0])][1],
                                coords_bott_left[np.argmax(coords_bott_left[:, 0])][0])
            imagePoints.append((roi_pos[0] + circle_center_bl[0], roi_pos[1] + circle_center_bl[1]))
            # cv2.circle(frame, circle_center_bl, 2, (0, 0, 255), 3)
            circle_center_br = (coords_bott_right[np.argmax(coords_bott_right[:, 0])][1],
                                coords_bott_right[np.argmax(coords_bott_right[:, 0])][0])
            imagePoints.append((roi_pos[0] + circle_center_br[0], roi_pos[1] + circle_center_br[1]))
            circle_center_r = (roi_pos[0] + coords_right[np.argmax(coords_right[:, 1])][1],
                               roi_pos[1] + coords_right[np.argmax(coords_right[:, 1])][0])
            imagePoints.append(circle_center_r)

            coords_center = np.squeeze(charger_coords[np.argwhere(charger_coords[:, 1] == int(
                min(circle_center_bl[0], circle_center_br[0]) + abs(circle_center_bl[0] - circle_center_br[0]) / 2))])
            circle_center_c = (roi_pos[0] + coords_center[np.argmax(coords_center[:, 0])][1],
                               roi_pos[1] + coords_center[np.argmax(coords_center[:, 0])][0])

            imagePoints.append(circle_center_c)
        except Exception as e:
            print(imagePoints)
            print(e)
            return None
        for pt in imagePoints:
            cv2.circle(frame, tuple(pt), 1, (255, 0, 0), 10)
        cv2.imshow("imgs", cv2.resize(frame, (1280, 960)))
        return imagePoints

    def calc_PnP_pose(self, imagePoints):
        # print("imagePoints", np.array(imagePoints).astype(np.int16))
        if imagePoints is None and len(imagePoints) > 0:
            return None
        if len(imagePoints) == 6:
            PnP_image_points = imagePoints[:4]  # [imagePoints[0], imagePoints[3], imagePoints[2], imagePoints[1]]
            # PnP_image_points =[(imagePoints[0][1], imagePoints[0][0]), (imagePoints[1][1], imagePoints[1][0]),
            #                    (imagePoints[2][1], imagePoints[2][0]), (imagePoints[3][1], imagePoints[3][0])]
            # object_points = np.array([(-0.32, 0, -0.65), (-0.075, -0.255, -0.65), (0.075, -0.255, -0.65), (0.32, 0, -0.65)]).astype(np.float64)
            object_points = np.array(
                [(-0.32, 0, -0.65), (-0.075, -0.255, -0.65), (0.075, -0.255, -0.65), (0.32, 0, -0.65)]).astype(
                np.float64) * 0.99
            # object_points = np.array([(-0.32, 0.31, 0.0), (-0.075, 0.075, 0.0), (0.075, 0.075, 0.0), (0.32, 0.31, 0.0)]).astype(np.float64) #*1.165 #A8 front
            # print(object_points)
            # object_points = np.array([(-0.31, 0, -0.65), (-0.075, -0.255, -0.65), (0.075, -0.255, -0.65), (0.31, 0, -0.65)]).astype(np.float64)
        elif len(imagePoints) == 4:
            PnP_image_points = imagePoints
            # object_points = np.array([(-0.32, 0, -0.65), (-0.075, -0.255, -0.65), (0.075, -0.255, -0.65), (0.32, 0, -0.65)]).astype(np.float64)
            object_points = np.array(
                [(-0.32, 0.255, 0.65), (-0.075, 0.0, 0.65), (0.075, 0.0, 0.65), (0.32, 0.255, 0.65)],
                dtype=np.float32)
        elif len(imagePoints) == 8:
            # PnP_image_points = imagePoints[:4]
            # object_points = np.array([(-0.32, 0, -0.65), (-0.075, -0.255, -0.65), (0.075, -0.255, -0.65), (0.32, 0, -0.65)]).astype(np.float64)
            PnP_image_points = imagePoints
            object_points = np.array(
                [(-0.32, 0.0, -0.65), (-0.075, -0.255, -0.65), (0.075, -0.255, -0.65), (0.32, 0.0, -0.65),
                 (0.32, 0.0, 0.65), (0.075, -0.255, 0.65), (-0.075, -0.255, 0.65), (-0.32, 0.0, 0.65)]).astype(
                np.float64)
        # object_points = np.array([(0, 0, 0), (0.64, 0, 0), (0.395, -0.255, 0), (0.245, -0.255, 0)]).astype(np.float64)
        # object_points = np.array([(0, 0, 0), (0.61, 0, 0), (0.38, -0.255, 0), (0.23, -0.255, 0)]).astype(np.float64)
        PnP_image_points = np.array(PnP_image_points).astype(np.float64)
        if self.PnP_pose_data is not None:
            init_guess = np.array(self.PnP_pose_data)
        else:
            init_guess = np.array([0.0, 0.5, 30.0])
        retval, rvec, tvec = cv2.solvePnP(object_points, PnP_image_points, self.camera_matrix,
                                          distCoeffs=None,  # (-0.10112, 0.07739, -0.00447, -0.0070),
                                          rvec=np.array([0.0, 0.0, 0.0]))
        # retval, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, PnP_image_points, self.camera_matrix,
        #                                                  iterationsCount=10000, distCoeffs=(-0.10112, 0.07739, 0.00447, -0.00070, 0.00000))
        # print('TVEC', tvec)
        # print('RVEC', rvec)
        self.PnP_pose_data = tvec
        return tvec, rvec
