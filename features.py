import numpy as np
import torch
import torchvision
from scipy.stats import multivariate_normal

def get_gaussian_mask():
	"""generates gaussian mask

	Returns:
		mask(numpy array): gaussian mask
	"""
	#128 is image size
	x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
	xy = np.column_stack([x.flat, y.flat])
	mu = np.array([0.5,0.5])
	sigma = np.array([0.22,0.22])
	covariance = np.diag(sigma**2) 
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance) 
	z = z.reshape(x.shape) 

	z = z / z.max()
	z  = z.astype(np.float32)

	mask = torch.from_numpy(z)

	return mask

class Encoder():
	"""
	Features for each detected object are returned using a siamese network.
	"""
	def __init__(self,device,wt_path=None):
		#loading this encoder is slow, should be done only once.
		self.device = device
		self.encoder = torch.load(wt_path,map_location=torch.device(self.device))			
		self.encoder = self.encoder.eval()
		print("Deep sort model loaded from path: ", wt_path)
		self.gaussian_mask = get_gaussian_mask().to(self.device)

	def pre_process(self,frame,detections):	
		"""pre processing frames and detections to load into siamese networks

		Args:
			frame (_type_): _description_
			detections (_type_): _description_

		Returns:
			_type_: _description_
		"""
		transforms = torchvision.transforms.Compose([ \
			torchvision.transforms.ToPILImage(),\
			torchvision.transforms.Resize((128,128)),\
			torchvision.transforms.ToTensor()])
		crops = []
		for d in detections:
			for i,_ in enumerate(d):
				if d[i] <0:
					d[i] = 0
			xmin,ymin,xmax,ymax = d
			ymin = abs(int(ymin))
			ymax = abs(int(ymax))
			xmin = abs(int(xmin))
			xmax = abs(int(xmax))
			try:
				crop = frame[ymin:ymax,xmin:xmax,:]
				crop = transforms(crop)
				crops.append(crop)
			except Exception as err:
				continue
		crops = torch.stack(crops)
		return crops

	
	def getFeatures(self, frame, out_boxes):
		"""unique features for each class detected

		Args:
			frame (numpy array): frame
			out_boxes (numpy array): bouding box predictions

		Returns:
			features(List): list of features needed for deep sort tracking
		"""

		if out_boxes==[]:			
			return None
		detections = np.array(out_boxes)

		processed_crops = self.pre_process(frame,detections).to(self.device)
		processed_crops = self.gaussian_mask * processed_crops

		features = self.encoder.forward_once(processed_crops)
		features = features.detach().cpu().numpy()

		if len(features.shape)==1:
			features = np.expand_dims(features,0)

		return features



