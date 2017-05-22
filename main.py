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

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 32, 'Batch size.  ')
flags.DEFINE_integer('max_steps', 9, 'Maximum number of pieces in puzzle')
flags.DEFINE_integer('rnn_size', 200, 'RNN size.  ')
flags.DEFINE_integer('puzzle_width', 3, 'Puzzle Width')
flags.DEFINE_integer('puzzle_height', 3, 'Puzzle Height')
flags.DEFINE_integer('image_dim', 64, 'If use_cnn is set to true, we use this as the dimensions of each piece image')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_integer('inter_dim', 4096, 'Dimension of intermediate state - if using fully connected' )
flags.DEFINE_integer('fc_dim', 512, 'Dimension of final pre-encoder state - if using fully connected')
flags.DEFINE_bool('use_cnn', False, 'Use CNN for creating image embeddings')
flags.DEFINE_integer('input_dim', 12288, 'Dimensionality of input images - use if flattened')
flags.DEFINE_integer('vgg_dim', 2048, 'Dimensionality flattnened vgg pool feature')
flags.DEFINE_string('optimizer', 'Adam', 'Optimizer to use for training')
flags.DEFINE_integer('nb_epochs', 1000, 'the number of epochs to run')
flags.DEFINE_float('lr_decay', 0.95, 'the decay rate of the learning rate')
flags.DEFINE_integer('lr_decay_period', 20, 'the number of iterations after which to decay learning rate.')
flags.DEFINE_float('reg', 5e-4, 'regularization on model parameters')
flags.DEFINE_bool('use_cnn', False, 'Whether to use CNN or MLP for input dimensionality reduction')

class PointerNetwork(object):
    def __init__(self, max_len, input_size, size, num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, use_cnn, inter_dim, fc_dim, use_cnn, image_dim, vgg_dim):
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
        self.init =  tf.contrib.layers.variance_scaling_initializer()
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        cell = tf.contrib.rnn.GRUCell(size)
        decoder_cell = tf.contrib.rnn.GRUCell(size)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_targets = []
        self.target_weights = []
        for i in range(max_len):
            size = [batch_size, input_size] if not use_cnn else [batch_size, image_dim, image_dim, 3]
            self.encoder_inputs.append(tf.placeholder(
                tf.float32, size, name="EncoderInput%d" % i))

        for i in range(max_len + 1):
            size = [batch_size, input_size] if not use_cnn else [batch_size, image_dim, image_dim, 3]
            self.decoder_inputs.append(tf.placeholder(
                tf.float32, [batch_size, input_size], name="DecoderInput%d" % i))
            self.decoder_targets.append(tf.placeholder(
                tf.float32, [batch_size, max_len + 1], name="DecoderTarget%d" % i))  # one hot
            self.target_weights.append(tf.placeholder(
                tf.float32, [batch_size, 1], name="TargetWeight%d" % i))

        # Encoder
        # Neeed to pass both encode inputs and everything through a dense layer.
        if use_cnn:
            # Encoder 
            cnn_f_extractor = CNN_FeatureExtractor()
            stacked_ins = tf.stack(self.encoder_inputs)
            stacked_ins = tf.reshape(static_rnn, [-1, image_dim, image_dim, 3])
            features = cnn_f_extractor.getCNNFEatures(stacked_ins, vgg_dim, fc_dim,self.init)
            features = tf.reshape(features, [-1, batch_size, fc_dim])
            self.proj_encoder_inputs = tf.unstack(features)

            # Decoder
            stacked_ins = tf.stack(self.decoder_inputs)
            stacked_ins = tf.reshape(static_rnn, [-1, image_dim, image_dim, 3])
            inputfn, features = cnn_f_extractor.getCNNFEatures(stacked_ins, vgg_dim, fc_dim,self.init, get_inputfn = True)
            self.inputfn = inputfn
            features = tf.reshape(features, [-1, batch_size, fc_dim])
            self.proj_decoder_inputs = tf.unstack(features)
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
        encoder_outputs, final_state = tf.contrib.rnn.static_rnn(cell, self.proj_encoder_inputs, dtype=tf.float32)

        # Need a dummy output to point on it. End of decoding.
        encoder_outputs = [tf.zeros([FLAGS.batch_size, cell.output_size])] + encoder_outputs

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [tf.reshape(e, [-1, 1, cell.output_size])
                      for e in encoder_outputs]
        attention_states = tf.concat(axis=1, values=top_states)

        with tf.variable_scope("decoder"):
            outputs, states, _ = pointer_decoder(
                self.proj_decoder_inputs, final_state, attention_states, decoder_cell)

        with tf.variable_scope("decoder", reuse=True):
            predictions, _, inps = pointer_decoder(
                self.proj_decoder_inputs, final_state, attention_states, decoder_cell, feed_prev=True)

        self.predictions = predictions

        self.outputs = outputs
        self.inps = inps
        # move code below to a separate function as in TF examples

    def create_feed_dict(self, encoder_input_data, decoder_input_data, decoder_target_data):
        feed_dict = {}
        for placeholder, data in zip(self.encoder_inputs, encoder_input_data):
            feed_dict[placeholder] = data

        for placeholder, data in zip(self.decoder_inputs, decoder_input_data):
            feed_dict[placeholder] = data

        for placeholder, data in zip(self.decoder_targets, decoder_target_data):
            feed_dict[placeholder] = data

        for placeholder in self.target_weights:
            feed_dict[placeholder] = np.ones([self.batch_size, 1])

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


    def step(self, optim, nb_epochs, lr_decay_period, reg, use_cnn):

        loss = 0.0
        for output, target, weight in zip(self.outputs, self.decoder_targets, self.target_weights):
            loss += tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target) * weight

        loss = tf.reduce_mean(loss)
        reg_loss = 0
        var_list = []

        for tf_var in tf.trainable_variable():
            if not ('Bias' in tf_var.name):
                if use_cnn and not('vgg_16' in tf_var.name)
                    var_list.append(tf_var)
                    reg_loss += tf.nn.l2_loss(tf_var)
                else:
                    var_list.append(tf_var)
                    reg_loss += tf.nn.l2_loss(tf_var)
        
        loss += reg * reg_loss

        test_loss = 0.0
        for output, target, weight in zip(self.predictions, self.decoder_targets, self.target_weights):
            test_loss += tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target) * weight

        test_loss = tf.reduce_mean(test_loss)

        optimizer = self.getOptimizer(optim) #tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)

        train_loss_value = 0.0
        test_loss_value = 0.0

        correct_order = 0
        all_order = 0

        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("/tmp/pointer_logs", sess.graph)
            init = tf.global_variables_initializer()
            sess.run(init)
            if use_cnn: self.inputfn(sess)
            for i in range(nb_epochs):
                if i > 0 and (i % lr_decay_period == 0):
                    sess.run([self.learning_rate_decay_op])

                encoder_input_data, decoder_input_data, targets_data = dataset.next_batch(
                    FLAGS.batch_size, FLAGS.max_steps)

                # Train
                feed_dict = self.create_feed_dict(
                    encoder_input_data, decoder_input_data, targets_data)
                d_x, l = sess.run([loss, train_op], feed_dict=feed_dict)
                train_loss_value = d_x #0.9 * train_loss_value + 0.1 * d_x

                if i % 1 == 0:
                    print('Step: %d' % i)
                    print("Train: ", train_loss_value)

                encoder_input_data, decoder_input_data, targets_data = dataset.next_batch(
                    FLAGS.batch_size, FLAGS.max_steps, train_mode=False)
                # Test
                feed_dict = self.create_feed_dict(
                    encoder_input_data, decoder_input_data, targets_data)
                # inps_ = sess.run(self.inps, feed_dict=feed_dict)

                predictions = sess.run(self.predictions, feed_dict=feed_dict)

                test_loss_value = sess.run(test_loss, feed_dict=feed_dict) #0.9 * test_loss_value + 0.1 * sess.run(test_loss, feed_dict=feed_dict)

                if i % 1 == 0:
                    print("Test: ", test_loss_value)

                predictions_order = np.concatenate([np.expand_dims(prediction, 0) for prediction in predictions])
                predictions_order = np.argmax(predictions_order, 2).transpose(1, 0)[:, 0:FLAGS.max_steps]

                input_order = np.concatenate([np.expand_dims(target, 0) for target in targets_data])
                input_order = np.argmax(input_order, 2).transpose(1, 0)[:, 1:] - 1
                # input_order = np.concatenate(
                #     [np.expand_dims(encoder_input_data_, 0) for encoder_input_data_ in encoder_input_data])
                # input_order = np.argsort(input_order, 0).squeeze().transpose(1, 0) + 1


                correct_order += np.sum(np.all(predictions_order == input_order,
                                               axis=1))
                all_order += FLAGS.batch_size

                if i % 1 == 0:
                    print('Correct order / All order: %f' % (correct_order / all_order))
                    correct_order = 0
                    all_order = 0

                    # print(encoder_input_data, decoder_input_data, targets_data)
                    # print(inps_)


if __name__ == "__main__":
    # TODO: replace other with params
    pointer_network = PointerNetwork(FLAGS.max_steps, FLAGS.input_dim, FLAGS.rnn_size,
                                     1, 5, FLAGS.batch_size, FLAGS.learning_rate, \
                                    FLAGS.lr_decay, FLAGS.use_cnn, FLAGS.inter_dim, \
                                    FLAGS.fc_dim, FLAGS.use_cnn, FLAGS.image_dim, FLAGS.vgg_dim)
    dataset = DataGenerator(FLAGS.puzzle_width, FLAGS.puzzle_width, FLAGS.input_dim)
    pointer_network.step(FLAGS.optimizer, FLAGS.nb_epochs, FLAGS.lr_decay_period, FLAGS.reg, FLAGS.use_cnn)
