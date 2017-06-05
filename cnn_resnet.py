import tensorflow as tf
import os
import vgg   
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope as vs
#import resnet_v1
import numpy as np 

from tensorflow.contrib.slim.python.slim.nets import resnet_v1

class  CNN_FeatureExtractor(object):
	def __init__(self):
		self.checkpoints_dir = "ckpts"
		self.ckpt_name = 'resnet_v1_50.ckpt' 


	def getCNNFeatures(self, input_tensor, num_classes, fc_initializer):
		graph = tf.Graph()

		with graph.as_default():

			with slim.arg_scope(resnet_v1.resnet_arg_scope()):
				net, end_points = resnet_v1.resnet_v1_50(input_tensor, num_classes=num_classes)
		model_path = os.path.join(self.checkpoints_dir, self.ckpt_name)
		init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, slim.get_model_variables('resnet_v1'))
		
		#flattened = tf.reshape(net, [-1, fc_dim])
		
		print ('last_layer: ', end_points['predictions'].shape)


		return init_fn, end_points['predictions'] 

# TEST: 
# cnn_f_extractor = CNN_FeatureExtractor()
# inputt = tf.constant(np.arange(12288, dtype=np.float32), shape=[1, 64, 64, 3]) 
# inputfn, features, first_layer = cnn_f_extractor.getCNNFEatures(inputt, 256, 1)
