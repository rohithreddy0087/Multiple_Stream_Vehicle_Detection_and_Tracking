import socket
import json

class SocketServer():
    """Creates a socket server and a method to connect to multiple clients

    Args:
        config(ConfigFileParser): object of ConfigFileParser
        queue(Queue): queue to send and recieve data between socket server and main thread
    Attributes:
        queue(Queue): queue to send and recieve data between socket server and main thread
        logger(logging): logger object
        detectors(List): list of streams
        config(ConfigFileParser): object of ConfigFileParser
        connected_clients(List): Maintains the list of all connected clients
    """
    def __init__(self,config,queue):
        self.queue = queue
        self.logger = config.logger
        self.detectors = config.arms
        self.config = config
        self.connected_clients = []

    def socket_server(self):
        """Binds the host and port

        Returns:
            socket: Socket object
        """
        server_socket = socket.socket()
        try:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.config.host, self.config.port))
        except socket.error as err:
            self.logger.error(str(err))
        server_socket.listen(15)
        self.logger.debug("Intialized Socket Server")
        return server_socket

    def end_connection(self,connection):
        """ends a connection
        """
        connection.close()

    def threaded_client(self):
        """Runs an infinte loop, communicates between server and multiple clients
        """
        while True:
            if not self.connected_clients:
                if not self.queue.empty():
                    data = self.queue.get()
            else:
                for connection in self.connected_clients:
                    if not self.queue.empty():
                        data = self.queue.get()
                        data = json.dumps(self.queue.get())
                        try:
                            connection.sendall(str(data).encode())
                        except Exception as err:
                            self.logger.debug("Exception %s", err)
                            self.connected_clients.remove(connection)
                            break

    def socket_connector(self):
        """Creates a socket object and runs a infinte loop to connect to the clients
        """
        server_socket = self.socket_server()
        while True:
            client, address = server_socket.accept()
            self.logger.debug('Connected to: ' + address[0] + ':' + str(address[1]))
            self.connected_clients.append(client)

def get_socketserver_object(global_var,config,queue):
    """Creates an instance of SocketServer class and stores it in the global_var dictionary

    Args:
        global_var (dict): to store all the classes initated
        config (configparser): config object to read from confil files

    Returns:
        SocketServer: instance of SocketServer class
    """
    if "SocketServer" not in global_var:
        global_var["SocketServer"] = SocketServer(config,queue)
    return global_var["SocketServer"]