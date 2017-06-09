"""Implementation of Pointer networks: http://arxiv.org/pdf/1506.03134v1.pdf.
"""

from __future__ import absolute_import, division, print_function

import random

import numpy as np
import tensorflow as tf

from dataset import DataGenerator
from pointer import pointer_decoder
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from cnn import CNN_FeatureExtractor
import random
import numpy as np
import os
import sys 
sys.path.append("utils/")
from evaluation import NeighborAccuracy, directAccuracy  
RAND_SEED = 1234
CKPT_DIR = "model_ckpts"

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('max_steps', 4, 'Maximum number of pieces in puzzle')
flags.DEFINE_integer('puzzle_width', 2, 'Puzzle Width')
flags.DEFINE_integer('puzzle_height', 2, 'Puzzle Height')
flags.DEFINE_integer('image_dim', 64, 'If use_cnn is set to true, we use this as the dimensions of each piece image')
flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate') # Hyper param
flags.DEFINE_integer('fc_dim', 1024, 'Dimension of final pre-encoder state - if using fully connected') # HYPER-PARAMS
flags.DEFINE_integer('vgg_dim', 2048, 'Dimensionality flattnened vgg pool feature') 
flags.DEFINE_string('optimizer', 'Adam', 'Optimizer to use for training') # HYPER-PARAMS
flags.DEFINE_integer('nb_epochs', 1000, 'the number of epochs to run')
flags.DEFINE_float('lr_decay', 0.95, 'the decay rate of the learning rate') # HYPER-PARAMS
flags.DEFINE_integer('lr_decay_period', 100, 'the number of iterations after which to decay learning rate.') # HYPER-PARAMS
flags.DEFINE_float('reg', 0.1, 'regularization on model parameters') # HYPER-PARAMS
flags.DEFINE_bool('load_from_ckpts', False, 'Whether to load weights from checkpoints')
flags.DEFINE_bool('tune_vgg', False, "Whether to finetune vgg")
flags.DEFINE_bool("use_jigsaws", True, "whether to use jigsaws for training")
flags.DEFINE_string("model_path", "model_ckpts/CNN_max_steps4_rnn_size-800_learning_rate-0.0001_fc_dim-1024_num-glimpses-0_reg-0.001_optimizer-Adam_bidirect-True_cell-type-GRU_num_layers-2_used-attn-one-hot/specials", "the path to the checkpointed model") #HYPER-PARAMS
flags.DEFINE_integer("train_data", 2560, "amount of data to train on")

class ClassifierNetwork(object):
    def __init__(self, max_len, batch_size, learning_rate, learning_rate_decay_factor, fc_dim, image_dim, vgg_dim, num_classes = 256, use_jigsaws=False):
        """Create the network. A simplified network that handles only sorting.
        
        Args:
            max_len: maximum length of the model.
            input_size: size of the inputs data.
            size: number of units in each layer of the model.
            num_layers: number of layers in the model.
            max_gradient_norm: gradients will be clipped to maximally this norm.
            batch_size: the size of the batches used during training;
                the model construction is independent of batch_size, so it can be
                changed after initialization if this is convenient, e.g., for decoding.
            learning_rate: learning rate to start with.
            learning_rate_decay_factor: decay learning rate by this much when needed.
        """
        self.init = tf.random_uniform_initializer(-0.001, 0.005)#tf.contrib.layers.variance_scaling_initializer()
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_pbty")
        # https://github.com/devsisters/pointer-network-tensorflow/blob/master/model.py#L148 

        i_size = [batch_size, image_dim, image_dim, 3] if not use_jigsaws else [batch_size, max_len, image_dim, image_dim, 3]
        self.inputs = tf.placeholder(tf.float32, i_size, name="inputs")
        self.targets = tf.placeholder(tf.float32, [batch_size, num_classes], name="Targets")  # one hot

        # Encoder
        # Neeed to pass both encode inputs and everything through a dense layer.
        cnn_f_extractor = CNN_FeatureExtractor()
        stacked_ins = tf.reshape(self.inputs, [-1, image_dim, image_dim, 3])
        print("SHAPERS ", stacked_ins.get_shape())
        if not use_jigsaws:
            fc_dim = num_classes
        inputfn, features, vgg_feat  = cnn_f_extractor.getCNNFEatures(stacked_ins, vgg_dim, fc_dim, self.init, use_full=(not use_jigsaws), keep_prob=self.keep_prob)
        self.vgg_feat = vgg_feat
        self.inputfn = inputfn
        if use_jigsaws:
            #variables_to_restore = tf.contrib.framework.get_variables_to_restore()
            #variables_to_restore = [var for var in variables_to_restore if 'fc_vgg' in var.name] # only use vgg things!
            #jig_init = tf.contrib.framework.assign_from_checkpoint_fn(FLAGS.model_path, variables_to_restore)
            #self.jig_init = jig_init
            self.inputfn = inputfn
            print("features after conv ", features.get_shape(), " maxlen", max_len, " fc_dim ", fc_dim)
            features = tf.reshape(features, [-1, max_len*fc_dim])
            self.jigsaw_feat = features
            print("jig features_shape ", features.get_shape())
            with vs.variable_scope("class_scope"):
                W = vs.get_variable("W", [max_len*fc_dim,  num_classes], initializer=self.init)
                b = vs.get_variable("b", [num_classes], initializer=self.init)
                features = tf.nn.elu(tf.matmul(features, W) + b)
        print("feature shape ", features.get_shape())
        self.outputs = tf.reshape(features, [-1, num_classes])
        #print("features shape now : ", len(self.outputs), " ", self.outputs[0].get_shape())
           

    def create_feed_dict(self, input_data, target_data, keep_prob):
        feed_dict = {}
        feed_dict[self.inputs] = input_data
        feed_dict[self.targets] = target_data

        feed_dict[self.keep_prob] = keep_prob
        return feed_dict


    def getOptimizer(self, optim):
        if optim == "Adam":
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif optim == "RMSProp":
            return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif optim == "Adadelta":
            return tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        else:
            return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)


    def step(self, optim, nb_epochs, lr_decay_period, reg, use_cnn, model_str, load_frm_ckpts, tune_vgg=False, use_jigsaws=False):

        loss = 0.0
        print(self.outputs.get_shape(), self.targets.get_shape())
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs,  labels=self.targets)
        print("loss_shape", loss.get_shape())
        loss = tf.reduce_mean(loss) 
        reg_loss = 0
        var_list = []
        for tf_var in tf.trainable_variables():
            #print(" Trainable : ", tf_var.name)
            if not ('Bias' in tf_var.name):
                if use_cnn and ( not ('vgg_16' in tf_var.name) or tune_vgg):
                    if 'vgg_16' in tf_var.name : print('Added vgg weights for training')
                    if 'fc_vgg' in tf_var.name : continue
                    var_list.append(tf_var)
                    reg_loss += tf.nn.l2_loss(tf_var)
                else:
                    var_list.append(tf_var)
                    reg_loss += tf.nn.l2_loss(tf_var)
        
        loss += reg * reg_loss 

        optimizer = self.getOptimizer(optim) #tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)

        train_loss_value = 0.0
        test_loss_value = 0.0

        epoch_data = []
        ckpt_file = CKPT_DIR + "/" + model_str + "/" + model_str
        saver = tf.train.Saver()
        if use_jigsaws: 
             var_dict = {}
             for tf_var in tf.trainable_variables():
                if "fc_vgg" in tf_var.name:
                    print(tf_var.name)
                    var_dict[tf_var.name] = tf_var 
             jig_saver = tf.train.Saver(var_dict)
        config = tf.ConfigProto(allow_soft_placement=True)
        test_losses = []
        vgg_feats, jigsaw_feats = [], []
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            self.inputfn(sess)
            if use_jigsaws:
                print("Restoring the special weights, NOW !")
                jig_saver.restore(sess, FLAGS.model_path)
            if(load_frm_ckpts): 
                print("Restoring Ckpts at : ", ckpt_file)
                saver.restore(sess, ckpt_file)
            for i in range(nb_epochs):
                if i > 0 and (i % lr_decay_period == 0):
                    sess.run([self.learning_rate_decay_op])

                input_data, _ , targets_data = dataset.next_batch(
                    FLAGS.batch_size, FLAGS.max_steps)

                # Train
                feed_dict = self.create_feed_dict(input_data, targets_data, 0.5)
                d_x, d_reg, l, train_pred  = sess.run([loss, reg_loss, train_op, self.outputs], feed_dict=feed_dict)
                #vgg_feat  = sess.run([self.vgg_feat], feed_dict=feed_dict)
                #vgg_feats.extend(vgg_feat)
                #jigsaw_feats.extend(jigsaws_feats)
                #continue
                train_loss_value = d_x #0.9 * train_loss_value + 0.1 * d_x
                #print("temporal features :", len(tempf[0]), tempf[0])
                #print("here are the ouptuts : ",train_pred[0])
                #print("here are the targets : ", targets_data[0])
                train_pred = np.argmax(train_pred, axis=1)
                targets = np.argmax(targets_data, axis=1)
                train_acc = np.sum(train_pred == targets)*1.0/len(train_pred)
                if i % 1 == 0:
                    print('Step: %d' % i)
                    print("Train: ", train_loss_value - reg *d_reg, " reg:",  reg *d_reg, " acc : ", train_acc)

                input_data, _, targets_data = dataset.next_batch(
                    FLAGS.batch_size, FLAGS.max_steps, train_mode=False)
                # Test
                feed_dict = self.create_feed_dict(input_data,  targets_data, 1.0)
                test_loss_value, test_pred, d_reg  = sess.run([loss,  self.outputs, reg_loss], feed_dict=feed_dict) #0.9 * test_loss_value + 0.1 * sess.run(test_loss, feed_dict=feed_dict)             
                test_losses.append(test_loss_value)

                test_pred = np.argmax(test_pred, axis=1)
                targets = np.argmax(targets_data, axis=1)
                test_acc = np.sum(test_pred == targets)*1.0/len(test_pred)

                if i % 1 == 0:
                    print("Test: ", test_loss_value - reg *d_reg, " test_acc : ", test_acc)
                
                if i > 0 and min(test_losses) >= test_loss_value: 
                    saver.save(sess, ckpt_file)
                if True: #i > 0 and  i % 5 == 0 :
                    epoch_data.append([train_loss_value - reg *d_reg, test_loss_value - reg *d_reg, train_acc, test_acc])
                    np.save(CKPT_DIR + "/" + model_str + '/epoch_data_' + model_str, epoch_data)
            #np.save("vgg_features", vgg_feats)
            #np.save("jigsaw_features", jigsaw_feats)

def getModelStr():
    model_str = "Unsup-JIGSAW_" if FLAGS.use_jigsaws else "Temp-Unsup-INIT_"
    model_str += "learning_rate-" + str(FLAGS.learning_rate) + "_fc_dim-" + str(FLAGS.fc_dim) 
    model_str += "_reg-" + str(FLAGS.reg) 
    model_str += "_optimizer-" + FLAGS.optimizer + "_train-data-" + str(FLAGS.train_data)
    if FLAGS.tune_vgg: model_str += '_tuneVGG'
    return model_str

if __name__ == "__main__":
    # TODO: replace other with params
    model_str = getModelStr()
    print(model_str)
    if not os.path.isdir(CKPT_DIR):
        print('Creating ckpt directory')
        os.mkdir(CKPT_DIR)
    if not os.path.isdir(CKPT_DIR + "/" + model_str):
        print('Creating ckpt directory for : ', model_str)
        os.mkdir(CKPT_DIR + "/" + model_str)
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    with tf.device('/gpu:0'):
        classifier_network = ClassifierNetwork(FLAGS.max_steps, FLAGS.batch_size, FLAGS.learning_rate, \
                                        FLAGS.lr_decay, FLAGS.fc_dim, FLAGS.image_dim, FLAGS.vgg_dim, num_classes = 256, use_jigsaws=FLAGS.use_jigsaws)
        dataset = DataGenerator(FLAGS.puzzle_width, FLAGS.puzzle_width, 1000, True, FLAGS.image_dim, unsup=True)
        classifier_network.step(FLAGS.optimizer, FLAGS.nb_epochs, FLAGS.lr_decay_period, FLAGS.reg, True, model_str,FLAGS.load_from_ckpts, tune_vgg=FLAGS.tune_vgg, use_jigsaws=FLAGS.use_jigsaws)
