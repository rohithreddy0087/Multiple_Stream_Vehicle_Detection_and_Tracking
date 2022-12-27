from threading import Thread
import torch.multiprocessing as mp
import schedule
import time
import sys

from .config_parser import get_config
from .detect import run_detect
from .socket_server import get_socketserver_object
from .run_vehicle_tracking import vehicle_tracking
from .storage import global_var

def start_detector_process(send_queues,recv_queues,config,configfile,processes):
    """run object detection on each stream

    Args:
        send_queues (dict): keys as stream ids and values as queue objects
        recv_queues (dict): keys as stream ids and values as queue objects
        config (ParseConfig): ParseConfig Object
        configfile (str): configfile location
        processes (dict): multi process dict
    """
    config.logger.debug("Starting detector process")
    send,recv = [],[]
    for s,q in zip(send_queues,recv_queues):
        send.append(send_queues[s])
        recv.append(recv_queues[q])
    p = mp.Process(target = run_detect, args=(send,recv,configfile,))
    processes["detector"] = p
    p.start()

class VehicleTracking:
    """
    VehicleTracking class, starts all the processes and threads.

    Args:
        configfile(str): path to configfile, ex: 'configfile.ini'

    Attributes:
        configfile(str): path to configfile
        config( config object): parses config file and stores all the required data
        send_queues(dict): keys are arm_ids and values as Queue objects
        recv_queues(dict): keys are arm_ids and values as Queue objects
        socket_queue(Queue): used to transfer data to socket server
        server(SocketServer object): starts a server
        logger(itspelogger object): logger object
        threads(dict): keys are names and values are thread objects
        processes(dict): keys are names and values are process objects
    """
    def __init__(self,configfile):
        try:
            mp.set_start_method('spawn')
        except Exception as err:
            print(err)
        self.configfile = configfile
        self.config = get_config(global_var,configfile)
        self.send_queues = {}
        self.recv_queues = {}
        for ind,_ in enumerate(self.config.arms):
            self.send_queues[ind] = mp.Queue()
            self.recv_queues[ind] = mp.Queue()
        self.socket_queue = mp.Queue()
        self.server = get_socketserver_object(global_var,self.config,self.socket_queue)
        self.logger = self.config.logger
        self.threads = {}
        self.processes = {}

    def check_and_restart_threads(self):
        """
        For every configured frequency, all the threads are checked, if any thread has stopped running, they are again restarted
        """
        stopped_threads = []
        for thread, thread_obj in self.threads.items():
            if not thread_obj.is_alive():
                self.logger.debug("%s thread has stopped running. Restarting again ...", thread)
                stopped_threads.append(thread)
        if len(stopped_threads) == 0:
            self.logger.debug("All threads are running fine")
        else:
            for thread in stopped_threads:
                del self.threads[thread]
                if thread == "Data_Publish":
                    self.start_data_publish_thread()
                elif thread == "Socket_Server":
                    self.start_socket_server_thread()
    
    def check_and_restart_process(self):
        """
        For every configured frequency, all the processes are checked, if any process has stopped running, they are again restarted
        """
        stopped_processes = []
        self.logger.debug("Process %s",self.processes)
        for arm_id,process in self.processes.items():
            process.join(timeout=0)
            if not process.is_alive():
                self.logger.debug("Detector %s has stopped running. Restarting again ...", arm_id)
                stopped_processes.append(arm_id)
        if len(stopped_processes) == 0:
            self.logger.debug("All processes are running fine")
        else:
            for arm_id in stopped_processes:
                del self.processes[arm_id]
                self.start_vehicle_tracking_process(arm_id,self.configfile)

    def end_process(self):
        """
        Ends all processes
        """
        self.logger.debug("Ending all processes")
        for _,process in self.processes.items():
            process.join()
    
    def start_data_publish_thread(self):
        """
        Starts a thread to publish data to the clients connected
        """
        t1 = Thread(target = self.server.threaded_client,)
        self.threads["Data_Publish"] = t1
        t1.daemon = True
        t1.start()

    def start_socket_server_thread(self):
        """
        Socket server thread is started.
        """
        t2 = Thread(target = self.server.socket_connector,)
        self.threads["Socket_Server"] = t2
        t2.daemon = True
        t2.start() 
    
    def start_vehicle_tracking_process(self, arm_id,configfile):
        """
        vehicle_tracking process for each video stream is started

        Args:
            arm_id (str): stream id
            configfile (str): config file
        """
        self.logger.debug("Starting vehicle_tracking process for %s", arm_id)
        p = mp.Process(target = vehicle_tracking, args =(arm_id,configfile,self.send_queues[self.config.arms.index(arm_id)],self.recv_queues[self.config.arms.index(arm_id)],self.socket_queue,))
        self.processes[arm_id] = p
        p.start()

    def main(self):
        """
        Starts all threads and processes.
        Schedules checks for the above started threads and process
        """
        for arm_id in self.config.arms:
            self.start_vehicle_tracking_process(arm_id, self.configfile)
        start_detector_process(self.send_queues,self.recv_queues,self.config,self.configfile,self.processes)
        self.start_data_publish_thread()
        self.start_socket_server_thread()
        schedule.every(self.config.check_freq).minutes.do(self.check_and_restart_process)
        schedule.every(self.config.check_freq).minutes.do(self.check_and_restart_threads)
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                # self.end_process()
                sys.exit(0)

if __name__ == '__main__':
    s = VehicleTracking('configfile.ini')
    s.main()
