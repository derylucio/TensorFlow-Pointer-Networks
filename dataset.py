from __future__ import absolute_import, division, print_function

import numpy as np
import sys

from datagenerator import getData

class DataGenerator(object):

    def __init__(self, puzzle_height, puzzle_width, input_dim, use_cnn, image_dim):
        self.puzzle_width = puzzle_width
        self.puzzle_height = puzzle_height
        self.use_cnn = use_cnn
        self.image_dim = image_dim
        self.data = getData(puzzle_height, puzzle_width, use_cnn=self.use_cnn)
        self.data['train'] = (self.data['train'][0], np.argmax( self.data['train'][1] , axis = 2), self.data['train'][2])
        self.data['val'] = (self.data['val'][0], np.argmax(self.data['val'][1] , axis=2), self.data['val'][2])
        self.input_dim = input_dim
        self.curr_train_pos = 0
        self.curr_test_pos = 0

    def next_batch(self, batch_size, N, train_mode=True):
        """Return the next `batch_size` examples from this data set."""

        # A sequence of random numbers from [0, 1]
        reader_input_batch = []

        # Sorted sequence that we feed to encoder
        # In inference we feed an unordered sequence again
        decoder_input_batch = []

        # Ordered sequence where one hot vector encodes position in the input array
        writer_outputs_batch = []
        size = [batch_size, self.input_dim] if not self.use_cnn else [batch_size, self.image_dim, self.image_dim, 3]
        for _ in range(N):
            reader_input_batch.append(np.zeros(size))
        for _ in range(N + 1):
            decoder_input_batch.append(np.zeros(size))
            writer_outputs_batch.append(np.zeros([batch_size, N + 1]))

        mode_string = 'train' if train_mode else 'val'
        x, y, _ = self.data[mode_string]
        if train_mode:
            self.curr_train_pos += 1
            if batch_size*self.curr_train_pos >= len(x): self.curr_train_pos = 0
        else:
            self.curr_test_pos += 1
            if batch_size*self.curr_test_pos >= len(x): self.curr_test_pos = 0
        pos = self.curr_train_pos if train_mode else self.curr_test_pos
        x, y = x[pos*batch_size:(pos  + 1)*batch_size], y[pos*batch_size:(pos  + 1)*batch_size]
	
        for b in range(batch_size):
            for i in range(N):
                reader_input_batch[i][b] = x[b][i]
                if train_mode:
                    index = np.where(y[b] == i)[0][0]
                    decoder_input_batch[i + 1][b] = x[b][index]
                else:
                    decoder_input_batch[i + 1][b] = x[b][i]
                writer_outputs_batch[i][b, y[b][i] + 1] = 1.0

            # Points to the stop symbol
            writer_outputs_batch[N][b, 0] = 1.0
        return reader_input_batch, decoder_input_batch, writer_outputs_batch
    

#datagen = DataGenerator(2, 2, 12288, False, 64)
#r_in, d_in, r_out  = datagen.next_batch(2, 4, train_mode=False)
#print(np.argmax(np.transpose(r_out, (1, 0, 2)), axis=2) - 1)
