import tensorflow as tf
import os
import vgg
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope as vs
#import tensorflow.contrib.slim.nets

class  CNN_FeatureExtractor(object):
	def __init__(self):
		self.checkpoints_dir = "ckpts"
		self.ckpt_name = 'vgg_16.ckpt' 

	def getCNNFeatures(self, input_tensor, fc_dim, out_dim, fc_initializer, use_full=False):
		graph = tf.Graph()

		with graph.as_default():

			with slim.arg_scope(vgg.vgg_arg_scope()):
				logits, end_points = vgg.vgg_16(input_tensor, is_training=False)
		model_path = os.path.join(self.checkpoints_dir, self.ckpt_name)
		variables_to_restore = tf.contrib.framework.get_variables_to_restore()
		variables_to_restore = [var for var in variables_to_restore if 'vgg_16' in var.name] # only use vgg things!
		init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
		pool_result = end_points['vgg_16/pool5']
		print ('end_points: ', end_points)
		flattened = tf.reshape(pool_result, [-1, fc_dim])
		with vs.variable_scope('fc_vgg'):
	        	W = vs.get_variable("W", [fc_dim, out_dim], initializer=fc_initializer)
	        	b = vs.get_variable("b", [out_dim], initializer=fc_initializer)
	        	output = tf.nn.relu(tf.matmul(flattened, W) + b)
		return init_fn, output, end_points['vgg_16/conv1/conv1_1']

    
    
