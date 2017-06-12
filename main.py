"""Implementation of Pointer networks: http://arxiv.org/pdf/1506.03134v1.pdf. [L]
"""

from __future__ import absolute_import, division, print_function

import random

import numpy as np
import tensorflow as tf

from dataset import DataGenerator
from pointer import pointer_decoder
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
import cnn 
import cnn_resnet
import random
import numpy as np
import os
import sys 
import scipy.ndimage
sys.path.append("utils/")
from evaluation import NeighborAccuracy, directAccuracy  
import fitness_vectorized as fv
import datagenerator as dg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAND_SEED = 1234
FC_DIM_RESNET = 1000 
CKPT_DIR = "model_ckpts"

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('max_steps', 9, 'Maximum number of pieces in puzzle')
flags.DEFINE_integer('rnn_size', 1000, 'RNN size.  ') # HYPER-PARAMS
flags.DEFINE_integer('puzzle_width', 3, 'Puzzle Width')
flags.DEFINE_integer('puzzle_height', 3, 'Puzzle Height')
flags.DEFINE_integer('image_dim', 64, 'If use_cnn is set to true, we use this as the dimensions of each piece image')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate') # Hyper param
flags.DEFINE_integer('inter_dim', 4096, 'Dimension of intermediate state - if using fully connected' ) # HYPER-PARAMS
flags.DEFINE_integer('fc_dim', 256, 'Dimension of final pre-encoder state - if using fully connected') # HYPER-PARAMS
flags.DEFINE_integer('input_dim', 12288, 'Dimensionality of input images - use if flattened') 
flags.DEFINE_integer('vgg_dim', 2048, 'Dimensionality flattnened vgg pool feature') 
flags.DEFINE_string('optimizer', 'Adam', 'Optimizer to use for training') # HYPER-PARAMS
flags.DEFINE_integer('nb_epochs', 10000, 'the number of epochs to run')
flags.DEFINE_float('lr_decay', 0.95, 'the decay rate of the learning rate') # HYPER-PARAMS
flags.DEFINE_integer('lr_decay_period', 100, 'the number of iterations after which to decay learning rate.') # HYPER-PARAMS
flags.DEFINE_float('reg', 0.001, 'regularization on model parameters') # HYPER-PARAMS
flags.DEFINE_bool('use_cnn', True, 'Whether to use CNN or MLP for input dimensionality reduction') 
flags.DEFINE_bool('resnet_cnn', False, 'Whether to use resnet model of CNN.') 
flags.DEFINE_bool('load_from_ckpts', False, 'Whether to load weights from checkpoints')
flags.DEFINE_bool('tune_vgg', False, "Whether to finetune vgg")
flags.DEFINE_bool('bidirect', True, "Whether to use a bidirectional rnn for encoder")
flags.DEFINE_string('cell_type', 'GRU', 'The type of RNN cell to use for the pointer network') # HYPER-PARAMS
flags.DEFINE_bool('encoder_attn_1hot', True, 'Whether to use linear combination of attention, or argmax of attention to choose input') # HYPER-PARAMS
flags.DEFINE_float('dp', -1, "The rate to apply dropout. Put a negative value if you want to use l2regularization instead") #HYPER-PARAMS
flags.DEFINE_integer('num_glimpses', 0, "The number of times to perform glimpses before final attention") # HYPTER-PARAMS
flags.DEFINE_integer('num_layers', 2, 'Number of layers for the RNN') # SANYA testing
flags.DEFINE_bool('test_mode', False, 'whether or not we are testing as opposed to training')

class PointerNetwork(object):
    def __init__(self, max_len, input_size, size, num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, inter_dim, fc_dim, use_cnn, resnet_cnn, image_dim, vgg_dim, bidirect):
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
        self.init = tf.contrib.layers.variance_scaling_initializer()
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_pbty")
        cell, cell_fw, cell_bw = None, None, None
        # https://github.com/devsisters/pointer-network-tensorflow/blob/master/model.py#L148 
        if num_layers == 1:
            if bidirect:
                cell_fw, cell_bw = self.getCell(size), self.getCell(size)
                decoder_cell = self.getCell(size * 2)
            else:
                cell = self.getCell(size)
                decoder_cell = self.getCell(size)
        elif num_layers > 1:
            if bidirect:
                cell_fw = tf.contrib.rnn.MultiRNNCell([self.getCell(size) for _ in range(num_layers)], state_is_tuple=True)
                cell_bw = tf.contrib.rnn.MultiRNNCell([self.getCell(size) for _ in range(num_layers)], state_is_tuple=True)
                decoder_cell = tf.contrib.rnn.MultiRNNCell([self.getCell(size * 2) for _ in range(num_layers)], state_is_tuple=True)
            else:
                cell = tf.contrib.rnn.MultiRNNCell([self.getCell(size) for _ in range(num_layers)], state_is_tuple=True)
                decoder_cell =  tf.contrib.rnn.MultiRNNCell([self.getCell(size) for _ in range(num_layers)], state_is_tuple=True)

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_targets = []
        self.target_weights = []
        self.fnames = []
        self.imgs = []

        for i in range(max_len):
            i_size = [batch_size, input_size] if not use_cnn else [batch_size, image_dim, image_dim, 3]
            self.encoder_inputs.append(tf.placeholder(
                tf.float32, i_size, name="EncoderInput%d" % i))

        for i in range(max_len + 1):
            i_size = [batch_size, input_size] if not use_cnn else [batch_size, image_dim, image_dim, 3]
            if i > 0 : self.decoder_inputs.append(tf.placeholder(
                tf.float32, i_size, name="DecoderInput%d" % i))
            self.decoder_targets.append(tf.placeholder(
                tf.float32, [batch_size, max_len + 1], name="DecoderTarget%d" % i))  # one hot
            self.target_weights.append(tf.placeholder(
                tf.float32, [batch_size, 1], name="TargetWeight%d" % i))
        
        # Encoder
        trainable_init_state = tf.Variable(tf.zeros(i_size), "trainable_init_state")
        self.decoder_inputs_updated  = [trainable_init_state] + self.decoder_inputs
        # Neeed to pass both encode inputs and everything through a dense layer.
        if use_cnn:
            # Encoder 
            if resnet_cnn:
                cnn_f_extractor = cnn_resnet.CNN_FeatureExtractor()
            else: 
                cnn_f_extractor = cnn.CNN_FeatureExtractor()
            all_inps = []
            num_encoder = len(self.encoder_inputs)
            all_inps.extend(self.encoder_inputs)
            all_inps.extend(self.decoder_inputs_updated)
            stacked_ins = tf.stack(all_inps)
            stacked_ins = tf.reshape(stacked_ins, [-1, image_dim, image_dim, 3])
            if resnet_cnn:
                inputfn, features = cnn_f_extractor.getCNNFeatures(stacked_ins, fc_dim, self.init)
            else: 
                inputfn, features = cnn_f_extractor.getCNNFeatures(stacked_ins, vgg_dim, fc_dim, self.init)

            self.print_out = features[0]
            self.inputfn = inputfn
            features = tf.reshape(features, [-1, batch_size, fc_dim])
            all_out = tf.unstack(features) 
            self.proj_encoder_inputs = all_out[:num_encoder]
            self.proj_decoder_inputs = all_out[num_encoder:]
        else:
            with vs.variable_scope("projector_scope"):
                W1 = vs.get_variable("W1", [input_size, inter_dim])
                b1 = vs.get_variable("b1", [inter_dim])
                W2 = vs.get_variable("W2", [inter_dim, fc_dim])
                b2 = vs.get_variable("b2", [fc_dim])
                self.proj_encoder_inputs = []
                for inp in self.encoder_inputs:
                    out = tf.nn.relu(tf.matmul(inp, W1) + b1)
                    out = tf.nn.relu(tf.matmul(out, W2) + b2)
                    self.proj_encoder_inputs.append(out)

            with vs.variable_scope("projector_scope", reuse=True):
                W1 = vs.get_variable("W1", [input_size, inter_dim])
                b1 = vs.get_variable("b1", [inter_dim])
                W2 = vs.get_variable("W2", [inter_dim, fc_dim])
                b2 = vs.get_variable("b2", [fc_dim])
                self.proj_decoder_inputs = []
                for inp in self.decoder_inputs:
                    out = tf.nn.relu(tf.matmul(inp, W1) + b1)
                    out = tf.nn.relu(tf.matmul(out, W2) + b2)
                    self.proj_decoder_inputs.append(out)

        # Need for attention
        if not bidirect:
            encoder_outputs, final_state = tf.contrib.rnn.static_rnn(cell, self.proj_encoder_inputs, dtype=tf.float32)
            #if num_layers > 1 : final_state = final_state[-1]
        else:
            encoder_outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, self.proj_encoder_inputs, dtype=tf.float32)
            if num_layers > 1: 
                final_state = []
                for ind, fw in enumerate(output_state_fw):
                    bw = output_state_bw[ind]
                    final_state.append(tf.concat([fw, bw], axis=1))
                 #output_state_fw, output_state_bw = output_state_fw[-1], output_state_bw[-1]
            if FLAGS.cell_type == "LSTM": # DOESN'T WORK FOR NUM_LAYERS > 1
                if num_layers == 1: output_state_fw, output_state_bw = [output_state_fw], [output_state_bw]
                final_state = []
                for ind, fw in enumerate(output_state_fw):
                    bw = output_state_bw[ind]
                    o_fw_c, o_fw_m  = tf.unstack(fw, axis=0)
                    o_bw_c, o_bw_m  = tf.unstack(bw, axis=0)
                    o_c =  tf.concat([o_fw_c, o_bw_c], axis=1)
                    o_m =  tf.concat([o_fw_m, o_bw_m], axis=1)   
                    final_state.append(tf.contrib.rnn.LSTMStateTuple(o_c, o_m))
                if num_layers == 1: final_state = final_state[0]
            elif num_layers <= 1:
                final_state = tf.concat([output_state_fw, output_state_bw], axis=1)
        
        output_size = size if not bidirect else 2*size
        #output_size = output_size*num_layers
        # Need a dummy output to point on it. End of decoding.
        encoder_outputs = [tf.zeros([FLAGS.batch_size, output_size])] + encoder_outputs #[tf.zeros([FLAGS.batch_size, output_size])] + encoder_outputs 
        # First calculate a concatenation of encoder outputs to put attention on.
        #print(encoder_outputs[1].get_shape(), output_state_fw.get_shape(), output_size)
        top_states = [tf.reshape(e, [-1, 1, output_size])
			      for e in encoder_outputs]
        attention_states = tf.concat(axis=1, values=top_states)

        with tf.variable_scope("decoder"):
            outputs, states, _, _ = pointer_decoder(
                self.proj_decoder_inputs, final_state, attention_states, decoder_cell, cell_type=FLAGS.cell_type, num_glimpses=FLAGS.num_glimpses, num_layers=num_layers)
        #print("DECODING")
        with tf.variable_scope("decoder", reuse=True):
            predictions, _, inps, cp = pointer_decoder(
                self.proj_decoder_inputs, final_state, attention_states, decoder_cell, feed_prev=True, one_hot=FLAGS.encoder_attn_1hot,  cell_type=FLAGS.cell_type, num_glimpses=FLAGS.num_glimpses, num_layers=num_layers)

        self.predictions = predictions
        #self.cps = tf.transpose(tf.stack(cp), (1, 0))
        self.outputs = outputs
        self.inps = inps
        # move code below to a separate function as in TF examples

    def getCell(self, size):
        if FLAGS.cell_type == "LSTM":
           cell = tf.contrib.rnn.LSTMCell(size, initializer=tf.random_uniform_initializer(-0.08, 0.08)) #based on paper recommendations
        else:
           cell = tf.contrib.rnn.GRUCell(size)#, initializer=tf.contrib.layers.variance_scaling_initializer())
        if FLAGS.dp < 0.0:
           return cell
        return tf.contrib.rnn.DropoutWrapper(cell,  variational_recurrent=True, input_size=tf.TensorShape(( FLAGS.fc_dim)),input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob, dtype=tf.float32)
           
    def create_feed_dict(self, encoder_input_data, decoder_input_data, decoder_target_data, keep_prob):
        feed_dict = {}
        for placeholder, data in zip(self.encoder_inputs, encoder_input_data):
            feed_dict[placeholder] = data

        for placeholder, data in zip(self.decoder_inputs, decoder_input_data):
            feed_dict[placeholder] = data

        for placeholder, data in zip(self.decoder_targets, decoder_target_data):
            feed_dict[placeholder] = data

        for placeholder in self.target_weights:
            feed_dict[placeholder] = np.ones([self.batch_size, 1])

        #for placeholder, data in zip(self.fnames, fnames_data):
        #    feed_dict[placeholder] = data

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


    def save_saliency(self, saliencies, idx):
        print("Saving saliencies...")
        fname = self.fnames[0]
        seqlen = FLAGS.max_steps
       
        # Hacky: Reloading images due to issue with encoder_inputs as tf Tensors. 
        img = dg.readImg(fname)
        args = ([img], FLAGS.puzzle_height, FLAGS.puzzle_width, (FLAGS.image_dim, FLAGS.image_dim, 3))
        img = dg.getReshapedImages(args)[0]
        args = (FLAGS.puzzle_height, FLAGS.puzzle_width, img, (FLAGS.image_dim, FLAGS.image_dim, 3))
        imgs = np.array(fv.splitImage(*args))
   
        nrows = 1 + seqlen
        ncols = seqlen
        fig, ax = plt.subplots(nrows, ncols, figsize=(20, 20),  squeeze=False)
        for i in range(nrows):
            for t in range(ncols):
                if i == 0:
                    ax[i, t].imshow(imgs[t]/255.0)
                    ax[i, t].axis("off")
                    ax[i, t].set_title("Image Chip " + str(t))
                    continue
                ax[i, t].set_title("Saliency " + "(Time %d, Chip %d)" % (i, t))
                ax[i, t].imshow(saliencies[(i - 1)*seqlen +  t], cmap=plt.cm.hot)
                ax[i, t].axis('off')
        
        save_fname = os.path.basename(fname) + getModelStr() #fname.replace("/", ".")
        plt.savefig("saliencies/" + str(idx) + "-" + save_fname + ".png")
    
    def step(self, optim, nb_epochs, lr_decay_period, reg, use_cnn, resnet_cnn, model_str, load_frm_ckpts, tune_vgg=False):
        loss = 0.0
        for output, target, weight in zip(self.outputs, self.decoder_targets, self.target_weights):
            loss += tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target) * weight

        loss = tf.reduce_mean(loss) / (FLAGS.max_steps  + 1)
        reg_loss = 0
        var_list = []
        special = {} 
        for tf_var in tf.trainable_variables():
            #print("trainable : ", tf_var.name)
            #if "Variable_" not in tf_var.name: to_load[tf_var.name] = tf_var
            if("fc_vgg" in tf_var.name): special[tf_var.name] = tf_var
            if ('fc_resnet' in tf_var.name):
                special[tf_var.name] = tf_var
            if not ('Bias' in tf_var.name):
                if use_cnn and ( not ('vgg_16' in tf_var.name) or tune_vgg) and (not resnet_cnn or not('resnet_v1_50' in tf_var.name)):
                    if 'vgg_16' in tf_var.name : print('Added vgg weights for training')
                    if 'resnet_v1_50' in tf_var.name: print ('Adding resnet weight')
                    var_list.append(tf_var)
                    reg_loss += tf.nn.l2_loss(tf_var)
                else:
                    var_list.append(tf_var)
                    reg_loss += tf.nn.l2_loss(tf_var)
        
        loss += reg * reg_loss if FLAGS.dp < 0.0 else 0.0
        tf.summary.scalar('train_loss', loss) 

        test_loss = 0.0
        for output, target, weight in zip(self.predictions, self.decoder_targets, self.target_weights):
            test_loss += tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target) * weight

        test_loss = tf.reduce_mean(test_loss) / (FLAGS.max_steps  + 1)
        tf.summary.scalar('test_loss', test_loss) # Sanya

        optimizer = self.getOptimizer(optim) #tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)

        print("Adding Histogram of Training Variables.")
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

        train_loss_value = 0.0
        test_loss_value = 0.0

        correct_order = 0
        all_order = 0
        epoch_data = []
        #model_str = 'CNN_max_steps4_rnn_size-400_learning_rate-0.0001_fc_dim-256_num-glimpses-0_reg-0.001_optimizer-Adam_bidirect-True_cell-type-GRU_num_layers-2_tuneVGG_used-attn-one-hot'
        ckpt_file = CKPT_DIR + "/" + model_str + "/" + model_str
        specials_file = CKPT_DIR + "/" + model_str + "/specials"
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        test_acc = []
        n_accs = []
        d_accs = []
        all_fnames = []
        special_saver = tf.train.Saver(special)
        with tf.Session(config=config) as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter("/tmp/" + model_str + "/train", sess.graph)
            test_writer = tf.summary.FileWriter("/tmp/" + model_str + "/test", sess.graph)

            init = tf.global_variables_initializer()
            sess.run(init)
            if use_cnn: self.inputfn(sess)
            if(load_frm_ckpts): 
                print("Restoring Ckpts at : ", ckpt_file)
                saver.restore(sess, ckpt_file)
            
            for i in range(nb_epochs):
                if i > 0 and (i % lr_decay_period == 0):
                    sess.run([self.learning_rate_decay_op])

                encoder_input_data, decoder_input_data, targets_data, self.fnames = dataset.next_batch(
                    FLAGS.batch_size, FLAGS.max_steps)

                if not FLAGS.test_mode:# Train
                    feed_dict = self.create_feed_dict(encoder_input_data, decoder_input_data, targets_data, FLAGS.dp)
                    d_x, d_reg, l, summary = sess.run([loss, reg_loss, train_op, merged], feed_dict=feed_dict)
                    train_loss_value = d_x #0.9 * train_loss_value + 0.1 * d_x
                    #train_writer.add_summary(summary, i)

                    if i % 1 == 0:
                        print('Step: %d' % i)
                        print("Train: ", train_loss_value - reg *d_reg, " reg:",  reg *d_reg)
                    encoder_input_data, decoder_input_data, targets_data, self.fnames = dataset.next_batch(FLAGS.batch_size, FLAGS.max_steps, train_mode=False)
                else:
                    encoder_input_data, decoder_input_data, targets_data, fnames  = dataset.next_batch(FLAGS.batch_size, FLAGS.max_steps, train_mode=False)

                # Test
                feed_dict = self.create_feed_dict(
                    encoder_input_data, decoder_input_data, targets_data, 1.0)
                # inps_ = sess.run(self.inps, feed_dict=feed_dict)

                predictions, targets_onehot = sess.run([self.predictions, self.decoder_targets], feed_dict=feed_dict)   
                targets = np.argmax(targets_onehot, axis=0)[0]
                # SALIENCY START
                show_saliency = True
                if show_saliency and (i % 20 == 0):
                    grads = []
                    inps = self.encoder_inputs 
                    for ind in targets:
                        if ind == 0: continue
                        ind -= 1
                        targ = tf.reduce_sum(self.outputs[ind][0, :])
                        grad = tf.gradients(targ, inps)
                        grad = [grad[j][0] for j in range(FLAGS.max_steps)]
                        grads.extend(grad)
                    grad_arr = sess.run([grads], feed_dict=feed_dict)
                    saliencies = []
                    for grad in grad_arr:
                        grad = np.abs(grad)
                        saliency = np.max(grad, axis=3)
                        saliencies.extend(saliency)
                    self.save_saliency(saliencies, i)
                
                test_loss_value, summary, indices = sess.run([test_loss, merged, self.cps], feed_dict=feed_dict)
                # SALIENCY END
                    
                test_writer.add_summary(summary, i)
                if i % 1 == 0:
                    print("Test: ", test_loss_value)
                
                predictions_order = np.concatenate([np.expand_dims(prediction, 0) for prediction in predictions])
                #print("Here are the predictions", np.transpose(predictions_order, (1, 0, 2))[0])
                #print ("Here are the maxinds ", indices[0])
                predictions_order = np.argmax(predictions_order, 2).transpose(1, 0)[:, 0:FLAGS.max_steps] - 1

                input_order = np.concatenate([np.expand_dims(target, 0) for target in targets_data])
                input_order = np.argmax(input_order, 2).transpose(1, 0)[:, 0:FLAGS.max_steps] - 1
                if FLAGS.test_mode:
                    all_fnames.extend(fnames)
                    for correct, pred in zip(input_order, predictions_order):
                        correct, pred = np.array(correct), np.array(pred)
                        n_accs.append(NeighborAccuracy(correct, pred, (FLAGS.puzzle_height, FLAGS.puzzle_width)))
                        d_accs.append(directAccuracy(correct, pred))
                total_neighbor_acc = 0.0
                total_direct_acc = 0.0
                for correct, pred in zip(input_order, predictions_order):
                    correct, pred = np.array(correct), np.array(pred)
                    total_neighbor_acc += NeighborAccuracy(correct, pred, (FLAGS.puzzle_height, FLAGS.puzzle_width))
                    total_direct_acc  += directAccuracy(correct, pred)
                print("Avg neighbor acc = ", total_neighbor_acc/len(input_order), "Avg direct Accuracy = ", total_direct_acc/len(input_order))
                epoch_data.append([train_loss_value, test_loss_value,total_neighbor_acc/len(input_order), total_direct_acc/len(input_order)])
                np.save(CKPT_DIR + "/" + model_str + '/epoch_data_' + model_str, epoch_data)
                test_acc.append(total_direct_acc)
                if i > 0 and max(test_acc) <= total_direct_acc and (not FLAGS.test_mode):
                    print("We are saving the checkpoints")
                    saver.save(sess, ckpt_file)
                    special_saver.save(sess, specials_file)
                    # print(encoder_input_data, decoder_input_data, targets_data)
                    # print(inps_)
        print("neighbor acc :", np.mean(n_accs), "direct acc: ", np.mean(d_accs))
        np.save("n_accs" + model_str, n_accs)
        np.save("d_accs" + model_str, d_accs)
        np.save("fnames" + model_str, all_fnames)

def getModelStr():
    model_str = "FIXED_PRES_RANGE_CNN_" if FLAGS.use_cnn else "MLP_"
    model_str += "max_steps" + str(FLAGS.max_steps) + "_rnn_size-" + str(FLAGS.rnn_size) + "_learning_rate-" + str(FLAGS.learning_rate) + "_fc_dim-" + str(FLAGS.fc_dim) + "_num-glimpses-" + str(FLAGS.num_glimpses)
    model_str += "_reg-" + str(FLAGS.reg) if FLAGS.dp < 0.0 else "_dp-" + str(FLAGS.dp)
    model_str += "_optimizer-" + FLAGS.optimizer + "_bidirect-" +  str(FLAGS.bidirect) + "_cell-type-" + FLAGS.cell_type + "_num_layers-" + str(FLAGS.num_layers)
    if FLAGS.tune_vgg: model_str += '_tuneVGG'
    if FLAGS.resnet_cnn: model_str += '_resnetCNN'
    if FLAGS.encoder_attn_1hot: model_str += '_used-attn-one-hot'
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
        pointer_network = PointerNetwork(FLAGS.max_steps, FLAGS.input_dim, FLAGS.rnn_size,
                                         FLAGS.num_layers, 5, FLAGS.batch_size, FLAGS.learning_rate, \
                                        FLAGS.lr_decay, FLAGS.inter_dim, \
                                        FLAGS.fc_dim, FLAGS.use_cnn, FLAGS.image_dim, FLAGS.vgg_dim, FLAGS.bidirect)
        dataset = DataGenerator(FLAGS.puzzle_height, FLAGS.puzzle_width, FLAGS.input_dim, FLAGS.use_cnn, FLAGS.image_dim, test_mode = FLAGS.test_mode)
        pointer_network.step(FLAGS.optimizer, FLAGS.nb_epochs, FLAGS.lr_decay_period, FLAGS.reg, FLAGS.use_cnn, model_str,FLAGS.load_from_ckpts, tune_vgg=FLAGS.tune_vgg)
