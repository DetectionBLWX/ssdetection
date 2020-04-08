'''
Function:
	some weight initialization methods from mmcv
Author:
	Charles
'''
import numpy as np
import torch.nn as nn


'''constant init'''
def constantInit(module, val, bias=0):
	if hasattr(module, 'weight') and module.weight is not None:
		nn.init.constant_(module.weight, val)
	if hasattr(module, 'bias') and module.bias is not None:
		nn.init.constant_(module.bias, bias)


'''xavier init'''
def xavierInit(module, gain=1, bias=0, distribution='normal'):
	assert distribution in ['uniform', 'normal']
	if distribution == 'uniform':
		nn.init.xavier_uniform_(module.weight, gain=gain)
	else:
		nn.init.xavier_normal_(module.weight, gain=gain)
	if hasattr(module, 'bias') and module.bias is not None:
		nn.init.constant_(module.bias, bias)


'''normal init'''
def normalInit(module, mean=0, std=1, bias=0):
	nn.init.normal_(module.weight, mean, std)
	if hasattr(module, 'bias') and module.bias is not None:
		nn.init.constant_(module.bias, bias)


'''uniform init'''
def uniformInit(module, a=0, b=1, bias=0):
	nn.init.uniform_(module.weight, a, b)
	if hasattr(module, 'bias') and module.bias is not None:
		nn.init.constant_(module.bias, bias)


'''kaiming init'''
def kaimingInit(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
	assert distribution in ['uniform', 'normal']
	if distribution == 'uniform':
		nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
	else:
		nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
	if hasattr(module, 'bias') and module.bias is not None:
		nn.init.constant_(module.bias, bias)


'''`XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch, Acknowledgment to FAIR's internal code'''
def caffe2XavierInit(module, bias=0):
	kaimingInit(module, a=1, mode='fan_in', nonlinearity='leaky_relu', distribution='uniform')


'''initialize conv/fc bias value according to giving probablity'''
def biasInitWithProb(prior_prob):
	bias_init = float(-np.log((1 - prior_prob) / prior_prob))
	return bias_init