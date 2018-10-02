"""
Richard Wang
"""

import tensorflow as tf
import sys
from myLib import *
import scipy.io as scipy_io
import os
import pdb

conv2d_transpose = tf.contrib.layers.conv2d_transpose
conv2d = tf.contrib.layers.conv2d

height = 16 # STFT 40-200Hz, 10Hz/band
width = 70 # 26701
depth = 64
height_ = 128 # 0-6kHz
width_ = 50 # 26701
depth_ = 1
LOG_DIR = './log'
LOG_DIR_RESULT = LOG_DIR+'/result/'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(LOG_DIR_RESULT):
    os.makedirs(LOG_DIR_RESULT)


def network(input, keep_prob):
    out = []
    current = tf.concat([input[:,-1::-1,:,:],input],1)
    out.append(current)
    # current = conv2d(current,num_outputs=128,kernel_size=[4,4],rate=2,scope = 'dilate1')
    current = conv2d_transpose(current, num_outputs=64, kernel_size=[4,5], stride=[2,1], scope='conv_1')
    out.append(current)
    # current = conv2d(current,num_outputs=32,kernel_size=[4,4],rate=4,scope = 'dilate2')
    current = conv2d_transpose(current, num_outputs=64, kernel_size=[4,5], stride=[2,1], scope='conv_2')
    out.append(current)
    # current = conv2d(current,num_outputs=8,kernel_size=[4,4],rate=8,scope = 'dilate3')
    current = conv2d_transpose(current, num_outputs=64, kernel_size=[4,5], stride=[2,1], scope='conv_3')
    current = conv2d(current, num_outputs=128, kernel_size=[4,5], stride=[1,1], scope='conv_4')
    current = conv2d(current, num_outputs=128, kernel_size=[4,5], stride=[1,1], scope='conv_5')
    current = conv2d(current, num_outputs=depth_, kernel_size=[4,5], stride=[1,1], scope='conv_6', activation_fn = None)
    out.append(current)
    # current = tf.nn.tanh(current)
    out.append(current)  
    return current[:,128:,10:60,:],out

# def network(input, keep_prob):
#     out = []
#     # current = conv2d(current,num_outputs=128,kernel_size=[4,4],rate=2,scope = 'dilate1')
#     current = input
#     current = conv2d(current, num_outputs=32, kernel_size=[16,7], stride=[1,1], scope='conv_1', padding = 'VALID')
#     out.append(current)
#     # current = conv2d(current,num_outputs=32,kernel_size=[4,4],rate=4,scope = 'dilate2')
#     current = conv2d(current, num_outputs=64, kernel_size=[1,8], stride=[1,1], scope='conv_2',padding = 'VALID')
#     out.append(current)
#     # current = conv2d(current,num_outputs=8,kernel_size=[4,4],rate=8,scope = 'dilate3')
#     current = conv2d(current, num_outputs=256, kernel_size=[1,8], stride=[1,1], scope='conv_3',padding = 'VALID', activation_fn = None)
#     out.append(current)
#     current = tf.nn.tanh(current)
#     out.append(current)
#     current_transpose = tf.transpose(current,perm = [0,2,3,1])
#     current_reshape = tf.reshape(current_transpose,[tf.shape(current_transpose)[0],tf.shape(current_transpose)[1],height_,depth_])
#     current = tf.transpose(current_reshape,perm=[0,2,1,3])
#     return current,out


# Placeholders
x = tf.placeholder(tf.float32, shape=[None, height, width, depth]) # [None, 28*28]
# print(x.get_shape)
y_ = tf.placeholder(tf.float32, shape=[None, height_, width_, depth_])  # [None, 10]
# y_conv = tf.placeholder(tf.float32, shape=[None, height_, width_, depth_])
# print(y_.get_shape)
deno = tf.placeholder(tf.float32)
sub = tf.placeholder(tf.float32)
idx = tf.placeholder(tf.float32, shape=[10])
keep_prob = tf.placeholder(tf.float32)

# Reshape 'x' and 'y' to a 4D tensor (2nd dim=width, 3rd dim=height, 4th dim=Channel)
x_spec = tf.reshape(x, [-1,height,width,depth])
# print(x_spec.get_shape)
y_spec_ = tf.reshape(y_, [-1,height_,width_,depth_])
# print(y_spec_.get_shape)
y_ = y_spec_

pred,out = network(x_spec,keep_prob)

losses = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_[:,3:,:], predictions=pred[:,3:,:]))
# losses = tf.reduce_mean(tf.losses.absolute_difference(labels=y_[:,:,:], predictions=pred[:,:,:]))

train_step = tf.train.AdamOptimizer(1e-3).minimize(losses)  # 1e-4


accuracy = losses
tf.summary.scalar('Losses:', losses)

saver = tf.train.Saver()

sess = tf.InteractiveSession()


## Run
sess.run(tf.global_variables_initializer())


# Include keep_prob in feed_dict to control dropout rate.
net,nst,nett,nstt = read_all_data()
for i in range(150):
    batch = get_data_train(50, 10, net, nst) ###########################################################################
    # Logging every ?th iteration in the training process.
    if i%5 == 0:
        # train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
        train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1], deno: batch[2], sub: batch[3], idx: batch[4], keep_prob: 1.0})
        print("Step %d, Training Losses %g" %(i, train_accuracy))
    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], deno: batch[2], sub: batch[3], keep_prob: 0.5})


# Evaluate our accuracy on the test data

for i in range(10):
    testset = get_data_test(sess, 50, 10, nett,nstt) ###########################################################################

    # test_accuracy = accuracy.eval(feed_dict={x: testset[0], y_: testset[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
    test_accuracy, final_result, out_, seg_idx, ddd, sss = sess.run([accuracy, pred, out, idx, deno, sub], feed_dict={x: batch[0], y_: batch[1], deno: testset[2], sub: testset[3], idx: testset[4], keep_prob: 1.0})


    result_index = np.asarray(seg_idx)

    # scipy_io.savemat(LOG_DIR_RESULT+'pred_index'+str(i)+'.mat', dict([('pred_index',result_index)]))
    scipy_io.savemat(LOG_DIR_RESULT+'GT_STFT'+str(i)+'.mat', dict([('GT_STFT',batch[1])]))
    scipy_io.savemat(LOG_DIR_RESULT+'input_STFT'+str(i)+'.mat', dict([('input_STFT',batch[0])]))
    scipy_io.savemat(LOG_DIR_RESULT+'out'+str(i)+'.mat',dict([('out0',out_[0]),('out1',out_[1]),('out2',out_[2]),('out3',out_[3])]))

    result_to_save = np.asarray(final_result)
    scipy_io.savemat(LOG_DIR_RESULT+'pred_STFT'+str(i)+'.mat', dict([('pred_STFT',result_to_save)]))

    print("Avg Test Losses %g" %(test_accuracy))


# pdb.set_trace()


saver.save(sess, LOG_DIR+'/model')

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())  
writer.close()  



"""
ECOG 16*t*128
  maybe 16*t*96
Speech 128*t*2

1st 32*t*32
2nd 64*t*8
3rd 128*t*2

channel 128 -> 32 -> 8 -> 2
  or 96 -> 32 -> 8 -> 2
"""