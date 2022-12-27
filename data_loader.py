import numpy as np
import torch

class StreamData():
    """
    Batches images into a single numpy array of 4 dimensions

    Args:
        device(int): device to run the Machine Learning model.ie., GPU or CPU
    """
    def __init__(self,device):
        self.device = device

    def pre_process(self, img):
        """pre process images to load them into model

        Args:
            img (numpy array): frame from video

        Returns:
            img (tensor): processed frame
        """
        img = np.array(img)
        img = img[:,:, :, ::-1].transpose(0,3,1,2)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0 
        return img