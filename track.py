import numpy as np
from operator import itemgetter

from .conventional_tracker.tracker2 import VehicleTracker
from .sort.sort import Sort
from .deep_sort import nn_matching, preprocessing
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker
from .features import Encoder
from .utils.torch_utils import select_device

class Track:
    """
    Different types of trackers are initiated
    """
    def __init__(self,config):
        self.config = config
        self.names = config.names
        self.device = select_device(self.config.device)
        self.tracker_model = config.tracker_model
        if self.tracker_model == "DEEPSORT":
            self.max_cosine_distance = config.max_cosine_distance
            self.nn_budget = config.nn_budget 
            self.nms_max_overlap = config.nms_max_overlap
            self.encoder = Encoder(self.device, wt_path=config.tracker_weights_file) 
            self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
            self.tracker = Tracker(self.metric,max_iou_distance=self.config.iou_threshold, max_age=self.config.max_age, n_init=self.config.max_hits)
        elif self.tracker_model == "CONVENTIONAL":
            self.tracker = VehicleTracker(self.names)
        elif self.tracker_model == "SORT":
            self.tracker = Sort(max_age=self.config.max_age, min_hits=self.config.max_hits, iou_threshold=self.config.iou_threshold)

    def deepsort_tracker(self,frame,det):
        """Deep sort tracker

        Args:
            frame (numpy array): video frame
            det (numpy array): predictions

        Returns:
            objects (dict): track ids as keys and other data as values
        """
        boxes = det[:, :4]
        scores = det[:,4]
        classes = det[:,5]
        features = self.encoder.getFeatures(frame = frame, out_boxes = boxes)
        boxes[:, 2] = boxes[:,2] - boxes[:,0]
        boxes[:, 3] = boxes[:,3] - boxes[:,1]
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, classes, features)]

        # indices = preprocessing.non_max_suppression(boxes, classes, self.nms_max_overlap, scores)
        # detections = [detections[i] for i in indices]       

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)
        objects = {}
        # update tracks
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            class_name = track.get_class()
            object_id = track.track_id
            bbox = track.to_tlbr()
            objects[object_id] = [self.names[int(class_name)],[bbox[0],bbox[1],bbox[2],bbox[3]]]
        return objects
    
    def sort_tracker(self,dets):
        """Sort Tracker

        Args:
            dets (numpy array): predictions

        Returns:
            objects (dict): track ids as keys and other data as values
        """
        #np.empty((0, 5)) 
        tracks= self.tracker.update(dets[:,:6])
        objects = {}
        for track in tracks:
            _id = int(track[4])
            objects[_id] = [self.names[int(track[5])],track[:4].astype(np.int32).tolist()]
        return objects

    def process(self,det,frame,frame_number):
        """process frames for tracking

        Args:
            det (list): list of predictions
            frame (numpy array): video frame
            frame_number (int): frame number

        Returns:
            _type_: _description_
        """
        objects = {}
        if self.tracker_model == "DEEPSORT":
            if len(det) > 0:
                objects = self.deepsort_tracker(frame,det)
        elif self.tracker_model == "CONVENTIONAL":
            if len(det) > 0:
                objects = self.tracker.track(frame,det,frame_number)
        elif self.tracker_model == "SORT":
            if len(det) == 0:
                det = np.empty((0, 6)) 
            objects = self.sort_tracker(det)
        return objects

def get_track_object(global_var,config):
    """Creates an instance of Track class and stores it in the global_var dictionary

    Args:
        global_var (dict): to store all the classes initated
        config (configparser): config object to read from confil files

    Returns:
        Track: instance of Track class
    """
    if "Track" not in global_var:
        global_var["Track"] = Track(config)
    return global_var["Track"]