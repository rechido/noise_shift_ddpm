
'''
# How to install
pip3 install prdc
pip3 install install pytorch-fid-wrapper; pip3 install git+https://github.com/evenmn/pytorch-sfid

# Toy example
python3 precision_recall.py
'''

import torch
import numpy 
import numpy as np
import sys 
import os
import matplotlib.pyplot as plt
import glob
import time
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torchvision

#from prdc import compute_prdc
try:                from .prdc import prdc 
except ImportError: from  prdc import prdc 
try:                from .prdc import compute_prdc
except ImportError: from  prdc import compute_prdc
try:                from .inceptionV3 import InceptionV3
except ImportError: from  inceptionV3 import InceptionV3

# ======================================================================
# cuda
# ======================================================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

def data2batch_to_compute_prec_rec(batch, img_size=224, batch_size=1):
	_, nc, height, width = batch.shape
	batch = batch.reshape(-1,batch_size,nc,height,width)
	if height != img_size or width != img_size:  # resize
		resized_batch=[]
		for i in range(len(batch)):
			data    = batch[i].squeeze(0)
			resized = torch.nn.functional.interpolate(input=data,size=(img_size, img_size))
			resized_batch.append( resized.unsqueeze(0).detach().cpu() )
		batch = torch.cat( resized_batch, dim=0 )
	if nc < 3:  batch = batch.expand(-1,-1,3,-1,-1)  # gray to rgb
	return batch


# https://github.com/mseitzer/pytorch-fid
def get_activations(batch, model, batch_size=50, dims=2048):

	model.eval()

	if batch_size > batch.shape[0]:
		print(('Warning: batch size is bigger than the data size. '
			   'Setting batch size to data size'))
		batch_size = batch.shape[0]

	pred_arr = np.empty((batch.shape[0], dims))

	#for i in tqdm(range(0, batch.shape[0], batch_size)): #tqdm: progressive bar
	for i in range(0, batch.shape[0], batch_size): 
		start = i
		end   = i + batch_size
		data  = batch[start:end].to(device)

		pred = model(data)[0]

		# If model output is not scalar, apply global spatial average pooling.
		# This happens if you choose a dimensionality not equal 2048.
		if pred.size(2) != 1 or pred.size(3) != 1:
			pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

		pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

	return pred_arr

# https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/5ad4629b07f3f3a51184c39d3dbe9085a60e264c/improved_precision_recall.py
def convert_image_batch_to_feature(batch_data, is_2d_image=True, feature_extractor=None, batch_size=5):

	if is_2d_image:
		if feature_extractor == None:
			#print('loading vgg16 for improved precision and recall...', end='', flush=True)
			#feature_extractor = models.vgg16(pretrained=True).eval().to(device)
			feature_extractor = models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).eval().to(device)
			img_size = 224 # VGG

			#print(f'image data is converted to (3,{img_size},{img_size}) to compute precision and recall')
			resized_batch = data2batch_to_compute_prec_rec(batch_data, img_size=img_size, batch_size=batch_size)	

			#print('converting resized data to feature...')
			batch_features = []
			for i in range(len(resized_batch)):
				batch    = resized_batch[i].squeeze(0)
				features = feature_extractor.features(batch.to(device)).view(-1, 7 * 7 * 512) # VGG11
				features = feature_extractor.classifier[:4](features)
				batch_features.append( features.detach().cpu() ) 
			batch_features = torch.cat( batch_features, dim=0 ).numpy()  # (batch_size, 4096) VGG
		
		elif feature_extractor.lower() == 'inceptionv3':
			#print(f'loading {feature_extractor} for improved precision and recall...', end='', flush=True)
			dims=2048
			block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
			feature_extractor = InceptionV3([block_idx]).to(device)
			img_size = 299 # InceptinoV3
			
			#print(f'image data is converted to (3,{img_size},{img_size}) to compute precision and recall')
			resized_batch = data2batch_to_compute_prec_rec(batch_data, img_size=img_size, batch_size=batch_size)	

			# data2batch_to_compute_prec_rec returns [num-batch, batch-size, c, w, h] for VGG.
			# We reshape it to (N,c,w,h).
			resized_batch = resized_batch.view(-1,resized_batch.shape[-3],resized_batch.shape[-2],resized_batch.shape[-1])

			#print('converting resized data to feature...')
			batch_features = get_activations(resized_batch, feature_extractor, batch_size=batch_size, dims=dims)
	else:
		# 1d data
		batch_features = batch_data.reshape(batch_data.shape[0],-1).detach().cpu().numpy()

	return batch_features

# batch_data_real:   real data with [N,c,H,W]
# batch_data_fake:   fake data with [N,c,H,W]
# is_2d_image:       [True] for images, and [False] for 1D data like two-moons
# feature_extractor: Network to computes features of images
# k:                 The number of points to compute manifold, The original paper recommends k=[5] while recent papers use k=[3].
def compute_precision_and_recall(batch_data_real, batch_data_fake, is_2d_image=True, feature_extractor=None, k_set=[3], batch_size=5, distance_metric='euclidean'):

	if len(batch_data_real) != len(batch_data_fake):
		print('warning: real data size {} is not equal to fake data size {}'.format(len(batch_data_real), len(batch_data_fake)))
		print('         precision and recall assumes the same number of reals and fakes')

	# ------------------------------------------------------------------------------------------------------
	# Step1: Given real, fake images, with a pretrained network, we compute features of the data
	# https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/5ad4629b07f3f3a51184c39d3dbe9085a60e264c/improved_precision_recall.py
	# ------------------------------------------------------------------------------------------------------
	real_features = convert_image_batch_to_feature(batch_data_real, is_2d_image=is_2d_image, feature_extractor=feature_extractor, batch_size=batch_size)
	fake_features = convert_image_batch_to_feature(batch_data_fake, is_2d_image=is_2d_image, feature_extractor=feature_extractor, batch_size=batch_size)

	# ------------------------------------------------------------------------------------------------------
	# Step2: Given features, we compute precision, recall, and others based on 
	# Reliable Fidelity and Diversity Metrics for Generative Models (ICML 2020)
	# https://github.com/clovaai/generative-evaluation-prdc
	# ------------------------------------------------------------------------------------------------------
	prdc_results=[]
	for nearest_k in k_set:
		dictionary_prdc = compute_prdc(real_features=real_features, fake_features=fake_features, nearest_k=nearest_k, distance_metric=distance_metric)	
		prdc_results.append( dictionary_prdc )

	# prdc_results[0]['precision']
	# prdc_results[0]['recall']
	# prdc_results[0]['density']
	# prdc_results[0]['coverage']

	return prdc_results 
	





# batch_data_real:   real data with [N,c,H,W]
# batch_data_fake:   fake data with [N,c,H,W]
# is_2d_image:       [True] for images, and [False] for 1D data like two-moons
# feature_extractor: Network to computes features of images
def compute_replicate(batch_data_real, batch_data_fake, is_2d_image=True, feature_extractor=None, batch_size=5, distance_metric='euclidean', weight_set=[1,3,7,15]):

	real_features = convert_image_batch_to_feature(batch_data_real, is_2d_image=is_2d_image, feature_extractor=feature_extractor, batch_size=batch_size)
	fake_features = convert_image_batch_to_feature(batch_data_fake, is_2d_image=is_2d_image, feature_extractor=feature_extractor, batch_size=batch_size)

	dictionary_replicate, first_smallest_indices, second_smallest_indices = prdc.compute_replicate(real_features, fake_features, distance_metric=distance_metric, weight_set=weight_set)

	return dictionary_replicate, first_smallest_indices, second_smallest_indices

if __name__ == '__main__':
	
	# ---------------------------------------------------------------------------------------------------------------
	# demonstration: cifar-10 like similar images
	batch_data_real =  torch.randn([10,3,28,28])
	batch_data_fake =  0.95 * batch_data_real + 0.05 * torch.randn([10,3,28,28]) 

	prdc_results = compute_precision_and_recall(batch_data_real, batch_data_fake, k_set=[3], is_2d_image=True, distance_metric='euclidean') 

	print('recall',    prdc_results[0]['recall'])       # [0]: result using 1st(0-th) k in k_set
	print('precision', prdc_results[0]['precision']) 
	print('density',   prdc_results[0]['density']) 
	print('coverage',  prdc_results[0]['coverage']) 
	
	weight_set            = [1,2,3,4,5] # [3] is recommended
	replicate_results, first_smallest_indices, second_smallest_indices = compute_replicate(batch_data_real, batch_data_fake, is_2d_image=True, weight_set=weight_set) 
	for weight in weight_set:
		print(f'replicate({weight})', replicate_results[f'replicate({weight})'])

	'''
	# ---------------------------------------------------------------------------------------------------------------
	# demonstration: synthetic data using different metrics

	# CASE 1: similar distributions
	batch_data_real =  torch.randn([10000,3,1,1])
	batch_data_fake =  batch_data_real #torch.randn([10000,3,1,1])

	for shift in [0,1,2,5]:
		batch_data_fake_shifted = batch_data_fake + shift

		for distance_metric in ['euclidean','cosine']:
			prdc_results = compute_precision_and_recall(batch_data_real, batch_data_fake_shifted, k_set=[3], is_2d_image=False, distance_metric=distance_metric)

			print(f"metric {distance_metric}, data shift {shift}")
			print('recall',    prdc_results[0]['recall'])       # [0]: result using 1st(0-th) k in k_set
			print('precision', prdc_results[0]['precision']) 
			print('density',   prdc_results[0]['density']) 
			print('coverage',  prdc_results[0]['coverage']) 
	'''

