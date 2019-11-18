'''
Function:
	vgg
Author:
	Charles
'''
import torch
import torchvision


'''vgg from torchvision==0.2.2'''
def VGGs(vgg_type, pretrained=False):
	if vgg_type == 'vgg16':
		model = torchvision.models.vgg16(pretrained=pretrained)
	elif vgg_type == 'vgg19':
		model = torchvision.models.vgg19(pretrained=pretrained)
	else:
		raise ValueError('Unsupport vgg_type <%s>...' % vgg_type)
	return model