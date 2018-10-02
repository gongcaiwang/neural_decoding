"""
Richard Wang
"""

import tensorflow as tf
import sys
from myLib_1d import *
import scipy.io as scipy_io
import os
import pdb

# conv1d_transpose = tf.contrib.layers.conv2d_transpose
conv = tf.contrib.slim.convolution

# height = 16 # STFT 40-200Hz, 10Hz/band
width = 160+64+56 # ecog segment width 26701
depth = 64
# height_ = 128 # 0-6kHz
width_ = 160/4 # spech segment width 26701
depth_ = 128*2#256#15#32
# frame_length = 64
# frame_step = 32
LOG_DIR = './log'
LOG_DIR_RESULT = LOG_DIR+'/result/'
L2_REG_WEIGHT = 0.01

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(LOG_DIR_RESULT):
    os.makedirs(LOG_DIR_RESULT)

def reshape_result(final_result):
    final_result1 = np.concatenate((np.zeros((final_result.shape[0],final_result.shape[1],1)),final_result[:,:,:depth_/2]),axis=2)
    final_result2 = np.concatenate((np.zeros((final_result.shape[0],final_result.shape[1],1)),final_result[:,:,depth_/2:]),axis=2)
    final_result = np.concatenate((
                                    np.power(10,final_result1[:,:,:,np.newaxis])/10,
                                    final_result2[:,:,:,np.newaxis]),axis=3)
    return final_result

def network(input, keep_prob):
    out = []
    # current = conv2d(current,num_outputs=128,kernel_size=[4,4],rate=2,scope = 'dilate1')
    current = input
    current = conv(current, num_outputs=64, kernel_size=32, stride=2, scope='conv_1')
    out.append(current)
    # current = conv2d(current,num_outputs=32,kernel_size=[4,4],rate=4,scope = 'dilate2')
    current = conv(current, num_outputs=128, kernel_size=32, stride=2, scope='conv_2')
    out.append(current)
    # current = conv2d(current,num_outputs=8,kernel_size=[4,4],rate=8,scope = 'dilate3')
    current = conv(current, num_outputs=256, kernel_size=32, stride=1, scope='conv_3')
    out.append(current)
    current = conv(current, num_outputs=512, kernel_size=8, stride=1, scope='conv_4')
    out.append(current)
    current = conv(current, num_outputs=depth_, kernel_size=3, stride=1, scope='conv_5')
    # current = tf.nn.tanh(current)
    out.append(current)
    return current[:,15:-15],out#current[:,8:-8],out

def network_dilate(input, keep_prob):
    out = []
    # current = conv2d(current,num_outputs=128,kernel_size=[4,4],rate=2,scope = 'dilate1')
    current = input
    current = conv(current, num_outputs=64, kernel_size=8, rate=1, stride=1, scope='conv_01')
    current = conv(current, num_outputs=64, kernel_size=8, rate=2, stride=1, scope='conv_02')
    current = conv(current, num_outputs=64, kernel_size=8, rate=4, stride=1, scope='conv_03')
    current = conv(current, num_outputs=64, kernel_size=8, rate=8, stride=1, scope='conv_04')
    current = conv(current, num_outputs=64, kernel_size=8, stride=2, scope='conv_1')
    out.append(current)
    # current = conv2d(current,num_outputs=32,kernel_size=[4,4],rate=4,scope = 'dilate2')
    current = conv(current, num_outputs=128, kernel_size=8, stride=2, scope='conv_2')
    out.append(current)
    # current = conv2d(current,num_outputs=8,kernel_size=[4,4],rate=8,scope = 'dilate3')
    current = conv(current, num_outputs=256, kernel_size=8, stride=1, scope='conv_3')
    out.append(current)
    current = conv(current, num_outputs=512, kernel_size=8, stride=1, scope='conv_4')
    out.append(current)
    current = conv(current, num_outputs=depth_, kernel_size=3, stride=1, scope='conv_5')
    # current = tf.nn.tanh(current)
    out.append(current)
    return current[:,15:-15],out#current[:,8:-8],out

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, width, depth]) # [None, 28*28]
y_ = tf.placeholder(tf.float32, shape=[None, width_, depth_])  # [None, 10]
# y_spectrom = tf.contrib.signal.stft(tf.squeeze(y_),frame_length=64,frame_step=32)


deno = tf.placeholder(tf.float32)
sub = tf.placeholder(tf.float32)
idx = tf.placeholder(tf.float32, shape=[10])
keep_prob = tf.placeholder(tf.float32)

# Reshape 'x' and 'y' to a 4D tensor (2nd dim=width, 3rd dim=height, 4th dim=Channel)
# x_spec = tf.reshape(x, [-1,height,width,depth])
# print(x_spec.get_shape)
# y_spec_ = tf.reshape(y_, [-1,height_,width_,depth_])
# print(y_spec_.get_shape)

pred,out = network_dilate(x,keep_prob)

l2_loss = tf.add_n([tf.nn.l2_loss(v)
                    for v in tf.trainable_variables()
                    if not('bias' in v.name)])

losses = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=pred)) + L2_REG_WEIGHT * l2_loss
# losses = tf.reduce_mean(tf.losses.absolute_difference(labels=y_[:,:,:], predictions=pred[:,:,:]))

train_step = tf.train.AdamOptimizer(1e-3).minimize(losses)  # 1e-4

tf.summary.scalar('Losses:', losses)
merged = tf.summary.merge_all()

saver = tf.train.Saver()

sess = tf.InteractiveSession()


## Run
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph) 

# Include keep_prob in feed_dict to control dropout rate.
ecog_train, spkr_train, ecog_test, spkr_test, spkrspec_train, spkrspec_test = read_all_data()

for i in range(1500): 
    batch = get_batch(ecog_train, spkrspec_train, seg_length=160, batch_size=10) ###########################################################################
    # Logging every ?th iteration in the training process.
    if i%5 == 0:
        # train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
        train_loss,l2_loss_ = sess.run([losses,l2_loss], feed_dict={ x:batch[0],
                                                        y_: batch[1],
                                                        keep_prob: 1.0})
        
        print("Step %d, Training Losses %g, l2_loss %g" %(i, train_loss, l2_loss_))
    _,summary = sess.run([train_step,merged],feed_dict={ x: batch[0], 
                                    y_: batch[1], 
                                    keep_prob: 0.5})
    summary_writer.add_summary(summary, i)


# Evaluate our accuracy on the test data

for i in range(10):
    testset = get_batch(ecog_test, spkrspec_test, seg_length=160, batch_size=10) ###########################################################################

    # test_accuracy = accuracy.eval(feed_dict={x: testset[0], y_: testset[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
    test_losses, final_result, out_ = sess.run([losses, pred, out], 
                                                                    feed_dict={ x: testset[0], 
                                                                                y_: testset[1],
                                                                                keep_prob: 1.0})
    final_result_train, out_train = sess.run([pred, out], 
                                                                    feed_dict={ x: batch[0], 
                                                                                y_: batch[1],
                                                                                keep_prob: 1.0})

    # final_result = reshape_result(final_result)
    # final_result_train = reshape_result(final_result_train)
    # gt_save_test = reshape_result(testset[1])
    # gt_save_train = reshape_result(batch[1])
    gt_save_test = testset[1]
    gt_save_train = batch[1]
    # result_index = np.asarray(seg_idx)
    # scipy_io.savemat(LOG_DIR_RESULT+'pred_index'+str(i)+'.mat', dict([('pred_index',result_index)]))
    scipy_io.savemat(LOG_DIR_RESULT+'GT_STFT_test'+str(i)+'.mat', dict([('GT_STFT_test',gt_save_test)]))
    scipy_io.savemat(LOG_DIR_RESULT+'input_STFT_test'+str(i)+'.mat', dict([('input_STFT_test',testset[0])]))
    scipy_io.savemat(LOG_DIR_RESULT+'out_test'+str(i)+'.mat',dict([('out0_test',out_[0]),('out1_test',out_[1]),('out2_test',out_[2]),('out3_test',out_[3])]))
    result_to_save = np.asarray(final_result)
    scipy_io.savemat(LOG_DIR_RESULT+'pred_STFT_test'+str(i)+'.mat', dict([('pred_STFT_test',result_to_save)]))

    scipy_io.savemat(LOG_DIR_RESULT+'GT_STFT_train'+str(i)+'.mat', dict([('GT_STFT_train',gt_save_train)]))
    scipy_io.savemat(LOG_DIR_RESULT+'input_STFT_train'+str(i)+'.mat', dict([('input_STFT_train',batch[0])]))
    scipy_io.savemat(LOG_DIR_RESULT+'out_train'+str(i)+'.mat',dict([('out0_train',out_train[0]),('out1_train',out_train[1]),('out2_train',out_train[2]),('out3_train',out_train[3])]))
    result_to_save = np.asarray(final_result_train)
    scipy_io.savemat(LOG_DIR_RESULT+'pred_STFT_train'+str(i)+'.mat', dict([('pred_STFT_train',result_to_save)]))

    print("Avg Test Losses %g" %(test_losses))


# pdb.set_trace()


saver.save(sess, LOG_DIR+'/model')

 
summary_writer.close()  



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