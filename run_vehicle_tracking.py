import os
from datetime import datetime
import time
import cv2

from .zone_assign import get_zoneassignment_object
from .config_parser import get_config
from .track import get_track_object
from .storage import global_var


def put_in_batch_queue(frame,queue):
    """
    Puts the frame into queue

    Args:
        frame (np.array): frame from a live stream
        queue (Queue): send queue of respective arm id
    """
    queue.put(frame)

def get_from_batch_queue(queue):
    """
    Dets the detections of the frame
    
    Args:
        queue (Queue): recv queue of respective arm id
    """
    det = queue.get()
    return det

def execute_vehicle_tracking(path,arm_id, config, track, assign_zone, send_queue, recv_queue, socket_queue):
    """
    Detects, Tracks and assigns zone for a given arm_id video or stream

    Args:
        path (str): video path or stream url
        arm_id (str): arm_id
        config (NyanamConfig object): contains all the data from configfiles
        track (Track object): used to track vehicles 
        assign_zone (ZoneAssignment object): assigns zone based on detection
        send_queue (Queue): used to send frames to the detector process
        recv_queue (Queue): receives detections from the detector process
        socket_queue (Queue): vbv data is pushed into this queue, to publish data to all connected clients

    Returns:
        vehicle_count(int): count of all the vehicles in that respective video
        frame_count(int): total number of frames in the stream or video 
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS of input video is ,",fps)
    vehicle_count,frame_count = 0,0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break
        frame = cv2.resize(frame,(640,640))
        put_in_batch_queue(frame,send_queue)
        det = get_from_batch_queue(recv_queue)
        objects = track.process(det,frame,frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame,counts = assign_zone.assign_zone(objects,frame,socket_queue)
        vehicle_count += counts
        frame_count += 1
        if config.debug:
            result = cv2.resize(frame,(480,480))
            cv2.imshow(arm_id, result)
            if cv2.waitKey(1) == ord('q'): break
    cv2.destroyAllWindows()
    return vehicle_count,frame_count

def vehicle_tracking(arm_id, configfile, send_queue,recv_queue,socket_queue):
    """
    Gets the config, zone_assignment and track object.

    Based on model, it executes either emulator or real stream

    """
    config = get_config(global_var,configfile)
    assign_zone = get_zoneassignment_object(global_var,arm_id,config)
    track = get_track_object(global_var,config)
    if config.model == "EMULATOR":
        video_dir_path = config.streams[arm_id] + "/" + arm_id
        print(os.listdir(video_dir_path))
        for video in  os.listdir(video_dir_path):
            t1 = time.time()
            video_path = video_dir_path + "/" + video
            vehicle_count,frame_count = execute_vehicle_tracking(video_path,arm_id, config, track, assign_zone, send_queue, recv_queue, socket_queue)
            t2 = time.time()
            with open("counts.txt","a") as f:
                f.write(str(datetime.now().strftime("%d-%m-%Y %H:%M:%S")) + "," +arm_id + "," + video + "," + str(vehicle_count)+","+str((t2-t1)/frame_count)+","+str((t2-t1))+","+str(frame_count)+"\n")
    execute_vehicle_tracking(config.streams[arm_id],arm_id, config, track, assign_zone, send_queue, recv_queue, socket_queue)