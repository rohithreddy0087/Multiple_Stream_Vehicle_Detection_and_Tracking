import cv2
import math
import numpy as np
from PIL import Image as im
import time

class VehicleTracker:
    def __init__(self,names):
        self.cnt = 0
        self.track_id = 0
        self.memory_buffer = {}
        self.frame_buffer = {}
        self.dist_buffer = {}
        self.color_buffer = {}
        self.names = names

    def max_color(self,frame, box):
        data = im.fromarray(frame)

        (x, y, w, h, _) = box
        im_trim1 = data.crop((x, y, x+w, y+h))
        imgn = np.asarray(im_trim1)

        height, width, _ = np.shape(imgn)
        # print(height, width)

        data2 = np.reshape(imgn, (height * width, 3))
        data2 = np.float32(data2)

        #number_clusters = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(
            data2, 1, None, criteria, 10, flags)
        centers = np.around(centers)
        centers = centers.tolist()
        return centers


    def calc_distance(self, pt2, pt):
        (x1, y1, w1, h1, _) = pt2
        (x2, y2, w2, h2, _) = pt
        cx1, cy1 = (x1 + x1 + w1) // 2, (y1 + y1 + h1) // 2
        cx2, cy2 = (x2 + x2 + w2) // 2, (y2 + y2 + h2) // 2

        dist = math.hypot(cx1 - cx2, cy1 - cy2)
        return dist

    def intersection_over_union(self, box, still_box):

        xA = max(box[0], still_box[0])
        yA = max(box[1], still_box[1])
        xB = min(box[2]+box[0], still_box[2]+still_box[0])
        yB = min(box[3]+box[1], still_box[3]+still_box[1])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxArea = (box[2] + 1) * (box[3] + 1)
        still_boxArea = (still_box[2] + 1) * (still_box[3] + 1)

        iou = float(interArea) / float(boxArea + still_boxArea - interArea)
        return iou

    def color_check(self, cur_color, obj_id):
        for i in range(3):
            if abs(self.color_buffer[obj_id][0][i] - cur_color[0][i]) > 20:
                return False
        return True

    def track(self, frame, bboxes, frame_no):

        if frame_no > 100:
            for no, pt in self.frame_buffer.items():
                if pt == frame_no-10:
                    self.memory_buffer.pop(no)
                    self.color_buffer.pop(no)

        center_pts_cur_frame = []

        for box in bboxes:
            (x, y, x2, y2) = (int(box[i]) for i in range(4))
            w, h = x2-x, y2-y
            class_id = box[5]
            center_pts_cur_frame.append([x, y, w, h, class_id])

        if self.cnt == 0:
            for pt in center_pts_cur_frame:
                self.memory_buffer[self.track_id] = pt
                self.frame_buffer[self.track_id] = frame_no
                self.color_buffer[self.track_id] = self.max_color(frame, pt)
                self.track_id += 1
        else:
            center_pts_cur_frame_copy = center_pts_cur_frame.copy()
            for pt in center_pts_cur_frame_copy:
                dist = 50
                iou = 0
                o = -1
                cur_color = self.max_color(frame, pt)
                for obj_id, pt2 in self.memory_buffer.items():
                    d = self.calc_distance(pt2, pt)
                    i = self.intersection_over_union(pt2, pt)
                    if dist > d and iou < i:
                        dist = d
                        iou = i
                        o = obj_id
                if o >= 0:
                    frame_diff = frame_no - self.frame_buffer[o]
                    arg = 12*(frame_diff)

                    iou_arg = 0.45/(frame_no**2-self.frame_buffer[o])
                    if iou > 0:
                        pte = self.memory_buffer[o]

                    if frame_diff == 1:
                        if dist < 15 or iou > 0.45:
                            self.memory_buffer[o] = pt
                            self.frame_buffer[o] = frame_no
                            if pt in center_pts_cur_frame:
                                center_pts_cur_frame.remove(pt)
                            continue
                    elif arg - 50 < dist < arg and pt[2] in range(pte[2]-5, pte[2]+5) and pt[3] in range(pte[3]-15, pte[3]+15) and self.color_check(cur_color, o) or iou > iou_arg:
                        self.memory_buffer[o] = pt
                        self.frame_buffer[o] = frame_no
                        if pt in center_pts_cur_frame:
                            center_pts_cur_frame.remove(pt)
                        continue

            for pt in center_pts_cur_frame:
                self.memory_buffer[self.track_id] = pt
                self.frame_buffer[self.track_id] = frame_no
                self.color_buffer[self.track_id] = self.max_color(frame, pt)
                self.track_id += 1

        ret_track = {}

        for object_id, pt in self.memory_buffer.items():
            if self.frame_buffer[object_id] == frame_no:
                ret_track[object_id] = [self.names[int(pt[4])],[pt[0],pt[1],pt[0]+pt[2],pt[1]+pt[3]]]

        if len(self.memory_buffer) != 0:
            self.cnt += 1

        return ret_track
