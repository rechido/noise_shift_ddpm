#https://arxiv.org/abs/1801.01973
#A Note on the Inception Score

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

try:
	from tqdm import tqdm
except ImportError:
	# If not tqdm is not available, provide a mock version of it
	def tqdm(x): return x



def inception_score(batch, cuda=True, batch_size=32, resize=False, splits=1):
	"""Computes the inception score of the generated images imgs

	imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
	cuda -- whether or not to run on GPU
	batch_size -- batch size for feeding into Inception v3
	splits -- number of splits
	"""


	def get_pred(x):
		if resize:
			x = up(x)
		x = inception_model(x)
		return F.softmax(x,dim=-1).data.cpu().numpy()

	# gray image
	if batch.shape[1] < 3:  batch = batch.expand(-1,3,-1,-1) 


	N = batch.shape[0]

	if batch_size > N:  batch_size = N

	# Set up dtype
	if cuda:
		dtype = torch.cuda.FloatTensor
	else:
		if torch.cuda.is_available():
			print("WARNING: You have a CUDA device, so you should probably set cuda=True")
		dtype = torch.FloatTensor

	# Load inception model
	#inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
	inception_model = inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).type(dtype)
	inception_model.eval();
	up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)


	# Get predictions
	preds = np.zeros((N, 1000))


	for i in tqdm(range(0, batch.shape[0], batch_size)): #tqdm: progressive bar

		start = i
		end = i + batch_size

		data = batch[start:end]

		if cuda:
			data=data.type(torch.cuda.FloatTensor)
		
		preds[start:end] = get_pred(data)

	'''
	for i, batch in enumerate(dataloader, 0):
		batch = batch.type(dtype)
		batchv = Variable(batch)
		batch_size_i = batch.size()[0]

		preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

	'''

	# Now compute the mean kl-div
	split_scores = []

	for k in range(splits):
		part = preds[k * (N // splits): (k+1) * (N // splits), :]
		py = np.mean(part, axis=0)
		scores = []
		for i in range(part.shape[0]):
			pyx = part[i, :]
			scores.append(entropy(pyx, py))
		split_scores.append(np.exp(np.mean(scores)))

	return np.mean(split_scores), np.std(split_scores) # average and std of Inception Score over the images
'''
if __name__ == '__main__':
	class IgnoreLabelDataset(torch.utils.data.Dataset):
		def __init__(self, orig):
			self.orig = orig

		def __getitem__(self, index):
			return self.orig[index][0]

		def __len__(self):
			return len(self.orig)

	import torchvision.datasets as dset
	import torchvision.transforms as transforms

	cifar = dset.CIFAR10(root='/hdd1/dataset/CIFAR/', download=False,
							 transform=transforms.Compose([
								 transforms.Scale(32),
								 transforms.ToTensor(),
								 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
							 ])
	)

	IgnoreLabelDataset(cifar)

	print ("Calculating Inception Score...")
	print (inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=32, resize=True, splits=10))
'''
