# from datasets import dataset_utils
# import tensorflow as tf

# url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"

# # Specify where you want to download the model to
# checkpoints_dir = "../ckpts"


# if not tf.gfile.Exists(checkpoints_dir):
#     tf.gfile.MakeDirs(checkpoints_dir)

# dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)


import tensorflow as tf
import os
import vgg
import vgg_full
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope as vs
#import tensorflow.contrib.slim.nets

class  CNN_FeatureExtractor(object):
	def __init__(self):
		self.checkpoints_dir = "ckpts"
		self.ckpt_name = 'vgg_16.ckpt' 

	def getCNNFEatures(self, input_tensor, fc_dim, out_dim, fc_initializer, use_full=False, keep_prob=0.5):
		graph = tf.Graph()

		with graph.as_default():
			#vgg = tf.contrib.slim.nets.vgg

			
			# if get_inputfn:
			# 	with slim.arg_scope(vgg.vgg_arg_scope(), reuse=True):
			# 		logits, end_points = vgg.vgg_16(input_tensor, is_training=False)	 # might want to change this once we decide to finetune
			# else:
			with slim.arg_scope(vgg.vgg_arg_scope()):
				logits, end_points = vgg.vgg_16(input_tensor, is_training=False) if not use_full else vgg_full.vgg_16(input_tensor, is_training=False)
		model_path = os.path.join(self.checkpoints_dir, self.ckpt_name)
		variables_to_restore = tf.contrib.framework.get_variables_to_restore()
		variables_to_restore = [var for var in variables_to_restore if 'vgg_16' in var.name] # only use vgg things!
		init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
		pool_result = end_points['vgg_16/pool5'] if not use_full else end_points['vgg_16/fc7'] 
		print("pool shape", pool_result.get_shape(), " usefull", use_full)
		flattened = tf.reshape(pool_result, [-1, fc_dim])
		print("flattened ", flattened.get_shape())
		#flattened = tf.nn.dropout(flattened, keep_prob)
		with vs.variable_scope('fc_vgg'):
	        	W = vs.get_variable("W", [fc_dim, out_dim], initializer=fc_initializer)
	        	b = vs.get_variable("b", [out_dim], initializer=fc_initializer)
	        	output = tf.nn.relu(tf.matmul(flattened, W) + b)
		#if not use_full: output = tf.nn.dropout(output, keep_prob)
		return init_fn, output, flattened

    
    
