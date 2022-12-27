from configparser import ConfigParser
import logging

class ParseConfig:
    """
    Parses configfile and stores them in attributes

    Args:
        configfile(str): path to configfile, ex: 'configfile.ini'

    Attributes:
        names(list): list of vehicle classes
        logger(logging): logger object
        arms(list): list of stream ids
        host(str): socket server host
        port(int): socket server port
        device(int): device to run the Machine Learning model.ie., GPU or CPU
        check_freq(int): frequency to check if all the threads are alive
        debug(bool): debug flag, cv2 window
        model(str): real or emulator flag
        detector_model(str): type of object detection model
        detector_weight_file(str): path to weights of object detection model
        img_size(int): size of input image to the object detection model
        conf_thres(float): confidence threshold for NMS
        iou_thres(float): IoU threshold for NMS
        tracker_model(str): type of object tracking model
        tracker_weights_file(str): path to weights of object tracking model
        max_cosine_distance(float): parameter of deep sort algorithm
        nn_budget(float): parameter of deep sort algorithm
        nms_max_overlap(float): parameter of deep sort algorithm
        iou_threshold(float): IoU threshold for object tracking model
        max_hits(int): maximum hits param for object tracking model 
        max_age(int): maximum age param for object tracking model 
        streams(dict): keys as stream_ids and values as video path
        zones(dict): keys as stream_ids and values as zone dimensions

    """
    def __init__(self, configfile = "configfile.ini"):
        self.names = ['car', 'motorbike', 'cycle', 'pedestrian', 'truck', 'bus', 'three wheeler']
        parser = ConfigParser()
        parser.read(configfile)
        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        self.logger = logging.getLogger('Tracker')
        fileHandler = logging.FileHandler("tracker_debug.log")
        fileHandler.setFormatter(log_formatter)
        self.logger .addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(log_formatter)
        self.logger .addHandler(consoleHandler)
        self.logger.setLevel(logging.DEBUG)

        self.arms = parser.get('DEFAULT','NUMBER_STREAMS',fallback="")
        if self.arms != "":
            self.arms = self.arms.split(",")
        self.host = parser.get('DEFAULT','SOCKET_HOST',fallback="0.0.0.0")
        self.port = int(parser.get('DEFAULT','SOCKET_PORT',fallback=1233))
        self.device = parser.get('DEFAULT','DEVICE',fallback=0)
        self.check_freq = int(parser.get('DEFAULT','CHECK_FREQUENCY',fallback=1))
        self.debug = parser.getboolean('DEFAULT','DEBUG',fallback=True)
        self.model = parser.get('DEFAULT','MODEL',fallback="REAL")
        self.detector_model = parser.get('DETECTOR','MODEL',fallback="yolov5n")
        self.detector_weights_file = parser.get('DETECTOR','WEIGHTS',fallback="/weights/best.pt")
        self.img_size = int(parser.get('DETECTOR','IMAGE_SIZE',fallback=640))
        self.conf_thres = float(parser.get('DETECTOR','CONFIDENCE_THRESHOLD',fallback=0.55))
        self.iou_thres = float(parser.get('DETECTOR','IoU_THRESHOLD',fallback=0.55))
        self.data = parser.get('DETECTOR','DATA',fallback="data.yaml")
        self.tracker_model = parser.get('TRACKER','MODEL',fallback="DEEPSORT")
        if self.tracker_model == "DEEPSORT":
            self.tracker_weights_file = parser.get('DEEPSORT','WEIGHTS',fallback="/weights/model640.pt")
            self.max_cosine_distance = float(parser.get('DEEPSORT','max_cosine_distance',fallback=0.4))
            self.nn_budget = parser.get('DEEPSORT','nn_budget',fallback=None)
            self.nms_max_overlap = float(parser.get('DEEPSORT','nms_max_overlap',fallback=1.0))
            self.iou_threshold = float(parser.get('DEEPSORT','iou_threshold',fallback=0.9))
            self.max_hits = int(parser.get('DEEPSORT','min_hits',fallback=3))
            self.max_age = int(parser.get('DEEPSORT','max_age',fallback=60))
        if self.tracker_model == "SORT":
            self.iou_threshold = float(parser.get('SORT','iou_threshold',fallback=0.3))
            self.max_hits = int(parser.get('SORT','min_hits',fallback=600))
            self.max_age = int(parser.get('SORT','max_age',fallback=3))
        self.streams = {}
        self.zones = {}
        self.get_camera_specfic_details(parser)
        self.logger.debug("Config parser attributes %s",self.__dict__)

    def get_camera_specfic_details(self,parser):
        """
        Gets details for each camera

        Args:
            parser
        """
        for arm in self.arms :
            self.streams[arm] = parser.get(arm,'STREAM')
            zones = int(parser.get(arm,'number_of_zones'))
            self.zones[arm] = {}
            for zone in range(1,zones+1):
                dims = parser.get(arm,'zone_dimensions_'+str(zone))
                self.zones[arm][zone] =  [int(dim) for dim in dims.split(",")]

def get_config(global_var,configfile):
    """Creates an instance of ConfigFileparser and stores it in the global_var dictionary
    Args:
        global_var (dict): to store all the classes initated
        configfile (str): path to config file

    Returns:
        ConfigFileparser: Instance of ConfigFileparser class
    """
    if "Config" not in global_var:
        global_var["Config"] = ParseConfig(configfile)
    return global_var["Config"]