from multiprocessing import Process
from datetime import datetime
import cv2
import numpy as np
from config_parser import get_config

class Polygon():
    """
    Creates a zone for given coordinates, 
    and has methods which gives information regarding whteher a  vehicle is in the zone or not
    """

    def __init__(self,points:list) -> None:
        self.INT_MAX = 10000
        self.points = points

    def onSegment(self,p:tuple, q:tuple, r:tuple) -> bool:
        if ((q[0] <= max(p[0], r[0])) &
            (q[0] >= min(p[0], r[0])) &
            (q[1] <= max(p[1], r[1])) &
            (q[1] >= min(p[1], r[1]))):
            return True
        return False
    
    def orientation(self,p:tuple, q:tuple, r:tuple) -> int:
        val = (((q[1] - p[1]) *
                (r[0] - q[0])) -
            ((q[0] - p[0]) *
                (r[1] - q[1])))
        if val == 0:
            return 0
        if val > 0:
            return 1
        else:
            return 2 

    def doIntersect(self,p1, q1, p2, q2):
    
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)
    
        if (o1 != o2) and (o3 != o4):
            return True
        
        if (o1 == 0) and (self.onSegment(p1, p2, q1)):
            return True
    
        if (o2 == 0) and (self.onSegment(p1, q2, q1)):
            return True

        if (o3 == 0) and (self.onSegment(p2, p1, q2)):
            return True
    
        if (o4 == 0) and (self.onSegment(p2, q1, q2)):
            return True
    
        return False
 
    def contains_points(self, p) -> bool:
        n = len(self.points)
        p = p[0]

        if n < 3:
            return False

        extreme = (self.INT_MAX, p[1])
        count = i = 0
        
        while True:
            next = (i + 1) % n
            
            if (self.doIntersect(self.points[i],
                            self.points[next],
                            p, extreme)):
                                
                if self.orientation(self.points[i], p,
                            self.points[next]) == 0:
                    return self.onSegment(self.points[i], p,
                                    self.points[next])
                                    
                count += 1
                
            i = next
            
            if (i == 0):
                break
            
        return count % 2 == 1

class ZoneAssignment:
    """
    Tracks and Assigns zone to the vehicles when vehicle enters the zone
    Args:
        arm_id(str):
        config (NyanamConfig object): contains all the data from configfiles
    Attributes:
        config (NyanamConfig object): contains all the data from configfiles
        logger (Logger object): logger
        zones (dict): keys as zone ids and values as list of dimensions
        polygons (list): list of Polygon objects
        tracked_objects(list): list of all tracked objects
        zone_coords(dict): keys as zone ids and values as list of dimensions

    """
    def __init__(self,arm_id,config):
        self.config = config
        self.logger = config.logger
        self.arm_id = arm_id
        self.zones = config.zones[arm_id]
        self.polygons = []        
        self.tracked_objects = []
        self.zone_coords = {}
        for _id,zone in self.zones.items():
            p = Polygon([(zone[0],zone[1]), (zone[2],zone[3]), (zone[4],zone[5]), (zone[6],zone[7])])
            self.polygons.append(p)
            self.zone_coords[_id] = np.array([[zone[0],zone[1]], [zone[2],zone[3]], [zone[4],zone[5]], [zone[6],zone[7]]],np.int32)
    
    def check_vehicle_in_zone(self,id,x,y):
        """
        Checks if vehicle is any of the zone.

        Args:
            id (int): 
            x (float): center x of the bbox
            y (float): center y of the bbox

        Returns:
            zone_id: return zone number in which the vehicle is present
        """
        for n,p in enumerate(self.polygons):
            pos = p.contains_points([(x, y)])
            if pos:
                return n+1
        return -1

    def generate_json(self,object,class_name,event,lane,queue):
        """
        Generates vbv if the object is detected

        Args:
            object (int): object id
            class_name (str): object class
            event (str): vehicle entry or vehicle exit
            lane (int): zone number
            queue (Queue): puts the data into socket queue
        """
        vbv = {
                "ObjectID":object,
                "TimeStamp":str(datetime.now().strftime("%d-%m-%Y %H:%M:%S")),
                "VehicleClass":class_name,
                "DetectorSCN":self.arm_id,
                "Lane":lane,
                "Event":event
            }
        self.tracked_objects.append(object)
        queue.put(vbv)

    def assign_zone(self,objects,frame,queue):
        """


        Args:
            objects (list): list of all tracked objects
            frame (np.array): 
            queue (Queue): puts the data into socket queue

        Returns:
            frame (np.array): 
            counts(int): count off all the vehicles in zone
        """
        counts = 0
        if self.config.debug:
            for id,dims in self.zone_coords.items():
                frame = cv2.polylines(frame, [dims] ,True, (255,0,0), 2)
        for object,vals in objects.items():
            bbox = vals[1]
            class_name = vals[0]
            color = (255,0,0)
            position = self.check_vehicle_in_zone(object,int(bbox[0]+bbox[2])/2,int(bbox[1]+bbox[3])/2)
            if position != -1:
                if object not in self.tracked_objects:
                    self.generate_json(object,class_name,"vehicle-entry",position,queue)
                    counts += 1
                    color = (0,255,0)
            # else:
            #     if object in self.tracked_objects:
            #         self.generate_json(object,class_name,"vehicle-exit",self.tracked_objects[object]["Lane"],queue)
            #         del self.tracked_objects[object]
            #         color = (0,0,255)
            if len(self.tracked_objects)>200:
                self.tracked_objects.pop(10)
            if self.config.debug:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(object)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, str(object)+str(class_name),(int(bbox[0]), int(bbox[1]-10)),0, 0.5, (255,255,255),1)
        return frame, counts

def get_zoneassignment_object(global_var,arm_id,config):
    """
    Args:
        global_var (dict): to store all the classes initated
        arm_id (str): stream_id
        config (configparser): config object to read from confil files

    Returns:
        ZoneAssignment: Instance of ZoneAssignment class
    """
    if "ZoneAssignment" not in global_var:
        global_var["ZoneAssignment"] = ZoneAssignment(arm_id,config)
    return global_var["ZoneAssignment"]