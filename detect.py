import torch

from .utils.general import non_max_suppression
from .utils.torch_utils import select_device
from .models.common import DetectMultiBackend
from .config_parser import get_config
from .data_loader import StreamData
from .storage import global_var

class Detect:
    """Object detection model with NMS is initalised and used on batch of images to perform detections
    Args:
        global_var (dict): to store all the classes initated
        configfile (str): path to config file
    Attributes:
        config (ConfigFileparser): Instance of ConfigFileparser class
        device(int): device to run the Machine Learning model.ie., GPU or CPU
        detector_model(DetectMultiBackend): object detection model
        data(StreamData): Instance of StreamData to preprocess video frames
    """
    def __init__(self,configfile):
        self.config = get_config(global_var,configfile)
        self.device = select_device(self.config.device)
        self.detector_model = DetectMultiBackend(self.config.detector_weights_file, data=self.config.data, device=self.device)
        self.data = StreamData(device=self.device)

    def detect(self,frames):
        """Object detection and NMS

        Args:
            frames (list): list of video frames from all streams

        Returns:
            dets (list): list of detector predictions
        """
        with torch.no_grad():
            images = self.data.pre_process(frames)
            det = self.detector_model(images)
            det = non_max_suppression(det, self.config.conf_thres, self.config.iou_thres)
            dets = []
            for pred in det:
                pred[:, [0, 2]] = pred[:, [0, 2]].clip(0, frames[0].shape[1])  # x1, x2
                pred[:, [1, 3]] = pred[:, [1, 3]].clip(0, frames[0].shape[0])
                pred = pred.cpu().numpy()
                dets.append(pred)
        return dets

    def get_from_queue(self,queue):
        """get frame from queue

        Args:
            queue (Queue): Queue for respective stream

        Returns:
            frame (numpy array): frame from live stream or video
        """
        frame = queue.get()
        return frame

    def put_in_queue(self,dets,queue):
        """puts predictions of frame in queue

        Args:
            dets (numpy array): predicitons
            queue (Queue): Queue for respective stream
        """
        if queue.empty():
            queue.put(dets)

    def process(self,send_queues,recv_queues):
        """Process the input video frames and output predicitions

        Args:
            send_queues (List): List of queues to send predicitions to main thread
            recv_queues (List): List of queues to receive predicitions from main thread
        """
        while True:
            data = []
            for queue in send_queues:
                frame = self.get_from_queue(queue)
                data.append(frame)
            det = self.detect(data)
            ind = 0
            for queue in recv_queues:
                self.put_in_queue(det[ind],queue)
                ind += 1

def run_detect(send,recv,configfile):
    """process input frames

    Args:
        send (List): List of queues to send predicitions to main thread
        recv (List): List of queues to receive predicitions from main thread
        configfile(str): path to configfile, ex: 'configfile.ini'

    Returns:
        Detect Instance: Instance of Detect Class
    """
    if "Detect" not in global_var:
        global_var["Detect"] = Detect(configfile)
    global_var["Detect"].process(send,recv)
    return global_var["Detect"]