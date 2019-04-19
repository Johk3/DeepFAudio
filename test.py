# Basic libraries
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

tf.reset_default_graph()
tf.set_random_seed(2016)
np.random.seed(2016)

# LSTM-autoencoder
from LSTMAutoencoder import *
print("\n\n\n\n")

data = pickle.load(open("data/save.p", "rb"))

# Constants
batch_num = 77
hidden_num = 12
step_num = 8
elem_num = 1
iteration = 10000

#data = data[len(data)%step_num:]
data = tf.convert_to_tensor(data, dtype=tf.float32)
print(data.get_shape().as_list())
# placeholder list
p_input = data

# p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]
p_inputs = []
for t in tf.split(p_input, step_num, 1):
    # Ignoring squeezing since it removes the necessary shape (77, ?)
    p_inputs.append(t)
print(p_inputs)

cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
try:
    ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell, decode_without_input=True)
except Exception as e:
    print("WARNING ", e)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(iteration):
        """Random sequences.
          Every sequence has size batch_num * step_num * elem_num 
          Each step number increases 1 by 1.
          An initial number of each sequence is in the range from 0 to 19.
          (ex. [8. 9. 10. 11. 12. 13. 14. 15])
        """
        r = np.random.randint(20, size=batch_num).reshape([batch_num, 1, 1])
        r = np.tile(r, (1, step_num, elem_num))
        d = np.linspace(0, step_num, step_num, endpoint=False).reshape([1, step_num, elem_num])
        d = np.tile(d, (batch_num, 1, 1))
        random_sequences = r + d
        
        random_sequences = random_sequences[:, :, 0]

        
        (loss_val, _) = sess.run([ae.loss, ae.train], {p_input: random_sequences})
        if i % 1000 == 0:
            print('iter %d:' % (i), loss_val)
        if i % 10000 == 0:
            save_path = saver.save(sess, "/tmp/model.ckpt")
            print("Model saved in path: {}".format(save_path))
            

    (input_, output_) = sess.run([ae.input_, ae.output_], {p_input: r + d})
    print('train result :')
    print('input :', input_[0, :, :].flatten())
    print('output :', output_[0, :, :].flatten())
