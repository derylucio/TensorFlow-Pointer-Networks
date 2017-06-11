import tensorflow as tf
import os
import vgg   
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope as vs
#import resnet_v1
import numpy as np 

from tensorflow.contrib.slim.python.slim.nets import resnet_v1

fc_dim = 8192

class  CNN_FeatureExtractor(object):
	def __init__(self):
		self.checkpoints_dir = "ckpts"
		self.ckpt_name = 'resnet_v1_50.ckpt' 


	def getCNNFeatures(self, input_tensor, out_dim, fc_initializer):
		graph = tf.Graph()

		with graph.as_default():

			with slim.arg_scope(resnet_v1.resnet_arg_scope()):
				net, end_points = resnet_v1.resnet_v1_50(input_tensor, num_classes=None)
		model_path = os.path.join(self.checkpoints_dir, self.ckpt_name)
		init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, slim.get_model_variables('resnet_v1'))
		flattened = tf.reshape(end_points["resnet_v1_50/block4"], [-1, fc_dim])
		print flattened.get_shape()
		with vs.variable_scope('fc_resnet'):
	        	W = vs.get_variable("W", [fc_dim, out_dim], initializer=fc_initializer)
	        	b = vs.get_variable("b", [out_dim], initializer=fc_initializer)
	        	output = tf.nn.relu(tf.matmul(flattened, W) + b)
	
		return init_fn, output

#TEST: 
# cnn_f_extractor = CNN_FeatureExtractor()
# inputt = tf.constant(np.arange(12288, dtype=np.float32), shape=[1, 64, 64, 3]) 
# inputfn, features = cnn_f_extractor.getCNNFeatures(inputt, 256, tf.contrib.layers.variance_scaling_initializer())
# print features.get_shape()