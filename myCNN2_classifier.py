"""
Richard Wang
"""

import tensorflow as tf
import sys
from myLib_classifier import *
import scipy.io as scipy_io
import os
import pdb
import re
import time
from ops import *

timestr = time.strftime("%Y%m%d-%H%M%S")
file_name =  os.path.basename(sys.argv[0])
file_name = re.sub('.py','',file_name)

# conv1d_transpose = tf.contrib.layers.conv2d_transpose
conv = tf.contrib.slim.convolution
def deconv(input, num_outputs, kernel_size, stride, scope):
    with tf.variable_scope(scope):
        input = tf.expand_dims(input, axis = 1)
        output = tf.contrib.slim.conv2d_transpose(input,num_outputs,[1,filter_width],[1,stride],scope=scope)
        output = tf.squeeze(output,axis = 1)
    return output

 
# height = 16 # STFT 40-200Hz, 10Hz/band
width = 512+64 # ecog segment width 26701
width_show = 512+64
depth = 64
# height_ = 128 # 0-6kHz
width_ = 512/4 # spech segment width 26701
width_show_ = 512/4
depth_ = 32#128#128#256#15#32 
# frame_length = 64
# frame_step = 32
LOG_DIR = './log/'+file_name+'_'+timestr
LOG_DIR_RESULT = LOG_DIR+'/result/'
dilation_channels = 32
residual_channels = 16
skip_channels = 512
filter_width = 4
L2_REG_WEIGHT = 1.#0.001
initial_filter_width_ = 32
quantization_channels = 2**8
disc_channels = 32 
batch_size = 32
from ops import *

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(LOG_DIR_RESULT):
    os.makedirs(LOG_DIR_RESULT)


def classifier(input_batch, train_phase, reuse=None, name = None):
    with tf.variable_scope('classifier', reuse = reuse):
        current = input_batch
        current = OptimizedResBlockDisc1(current, num_outputs=disc_channels, kernel_size = 8, scope = 'disc_block0',mode ='classifier',pool='MAX')
        current = res_block_dis(current, num_outputs = disc_channels, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block1',mode='classifier',pool='MAX')
        current = res_block_dis(current, num_outputs = disc_channels*2, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block2',mode='classifier',pool='MAX')
        current = res_block_dis(current, num_outputs = disc_channels*4, kernel_size = 16 ,train_phase = train_phase, resample='down', scope = 'disc_block3',mode='classifier',pool='MAX')
        current = res_block_dis(current, num_outputs = disc_channels*4, kernel_size = 16 ,train_phase = train_phase, resample='down', scope = 'disc_block4',mode='classifier',pool='MAX')
        current = res_block_dis(current, num_outputs = disc_channels*4, kernel_size = 16 ,train_phase = train_phase, resample='down', scope = 'disc_block5',mode='classifier',pool='MAX')
        # current = res_block_dis_gate(current, num_outputs = disc_channels*8, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block4',mode='discriminator')
        # current = res_block_dis_gate(current, num_outputs = disc_channels*16, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block5',mode='discriminator')
        # current = res_block_dis(current, num_outputs = disc_channels*4, kernel_size = 16, train_phase = train_phase, scope = 'disc_block4',mode='discriminator')
        # current = res_block_dis(current, num_outputs = disc_channels*4, kernel_size = 16, train_phase = train_phase, scope = 'disc_block5',mode='discriminator')
        # current = res_block_dis(current, num_outputs = disc_channels*4, kernel_size = 16, train_phase = train_phase, scope = 'disc_block6',mode='discriminator')
        current = norm_fn_critic(current, reuse = reuse, scope = 'norm1')
        current = nonlinearity(current,mode='classifier')
        # current = tf.reduce_mean(current, axis = 1)
        current = tf.layers.flatten(current)
        # current = tf.layers.dense(current, disc_channels*4, name = 'dense1')
        # current = norm_fn_critic(current, reuse = reuse, scope = 'norm2')
        # current = nonlinearity(current,mode='discriminator')
        current = tf.layers.dense(current, 2, name = 'dense2')
    return current

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, width, depth]) # [None, 28*28]
l = tf.placeholder(tf.float32, shape=[None, 2])  # [None, 10]


keep_prob = tf.placeholder(tf.float32)

# Reshape 'x' and 'y' to a 4D tensor (2nd dim=width, 3rd dim=height, 4th dim=Channel)
# x_spec = tf.reshape(x, [-1,height,width,depth])
# print(x_spec.get_shape)
# y_spec_ = tf.reshape(l, [-1,height_,width_,depth_])
# print(y_spec_.get_shape)

pred = classifier(x,train_phase=True, reuse = None)
# pred_test = network_wavenet(x, keep_prob,train_phase=False, reuse = True)
# # pred_test = network_wavenet(x, keep_prob,train_phase=True, reuse = True)
# pred_show = network_wavenet(x_show, keep_prob,train_phase=False, reuse = True)
# # pred_show = network_wavenet(x_show, keep_prob,train_phase=True, reuse = True)
pred_label = tf.argmax(
                    tf.cast(
                        tf.nn.softmax(tf.cast(pred, tf.float64)), tf.float32),
                    axis =-1)
l_label = tf.argmax(l,axis=-1)
accuracy = tf.reduce_mean(tf.to_float(tf.equal(pred_label, l_label)))
l2_loss = tf.add_n([tf.nn.l2_loss(v)
                    for v in tf.trainable_variables()
                    if not('bias' in v.name)])
# y_one_hot = tf.one_hot(l,depth = quantization_channels, dtype = tf.float32) 

losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                        logits=pred,
                                        labels=l))
losses = losses + L2_REG_WEIGHT * l2_loss
# losses = tf.reduce_mean(losses)
# losses = tf.reduce_mean(tf.losses.mean_squared_error(labels=l, predictions=pred)) + L2_REG_WEIGHT * l2_loss
losses_test = losses
losses_show = losses
# losses_test = tf.reduce_mean(tf.losses.mean_squared_error(labels=l, predictions=pred_test))
# losses_show = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_show_, predictions=pred_show))
# losses = tf.reduce_mean(tf.losses.absolute_difference(labels=l[:,:,:], predictions=pred[:,:,:]))

train_step = tf.train.AdamOptimizer(1e-3).minimize(losses)  # 1e-4

training_summary = tf.summary.scalar("training_loss", losses)
validation_summary = tf.summary.scalar("validation_loss", losses_test)

# merged = tf.summary.merge_all()

saver = tf.train.Saver()

sess = tf.Session()


## Run
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph) 

# Include keep_prob in feed_dict to control dropout rate.
ecog_train, spkr_train, ecog_test, spkr_test, spkrspec_train, spkrspec_test,label_train,label_test,start_ind_train,start_ind_test,target_train,target_test,peseudo_train,peseudo_test = read_all_data()
for i in range(1000): #820
    batch = get_batch(ecog_train, spkrspec_train,label_train, target_train,peseudo_train,start_ind_train,seg_length=512, batch_size=batch_size, threshold = 0.0,mode='train') ###########################################################################
    # Logging every ?th iteration in the training process.
    if i%10 == 0:
        # train_accuracy = accuracy.eval(feed_dict={x:batch[0], l: batch[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
        train_loss,l2_loss_,accuracy_train_,summary_train = sess.run([losses_test,l2_loss,accuracy,training_summary], feed_dict={ x:batch[0],
                                                        l: batch[3],
                                                        keep_prob: 1.0})
        testset = get_batch(ecog_test, spkrspec_test,label_test,target_test,peseudo_test,start_ind_test, seg_length=512, batch_size=batch_size, threshold = 1.2,mode='test')
        test_loss,accuracy_test_,summary_validate = sess.run([losses_test,accuracy,validation_summary], feed_dict={ x:testset[0],
                                                        l: testset[3],
                                                        keep_prob: 1.0})
        summary_writer.add_summary(summary_train, i)
        summary_writer.add_summary(summary_validate, i)
        print("Step %d, Training Losses %g, Test Losses %g, l2_loss %g" %(i, train_loss, test_loss, l2_loss_))
        print('accuracy_train %g, accuracy_test %g' %(accuracy_train_,accuracy_test_))
    sess.run(train_step,feed_dict={ x: batch[0], 
                                    l: batch[3], 
                                    keep_prob: 0.5})


# Evaluate our accuracy on the test data

for i in range(10):
    testset = get_batch(ecog_test, spkrspec_test, label_test,target_test,peseudo_test,start_ind_test,seg_length=512, batch_size=batch_size, threshold = 1.2,mode='test') ###########################################################################
    batch = get_batch(ecog_train, spkrspec_train, label_train, target_train,peseudo_train,start_ind_train ,seg_length=512, batch_size=batch_size, threshold = 1.2,mode='test')
    # test_accuracy = accuracy.eval(feed_dict={x: testset[0], l: testset[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
    # test_losses, final_result = sess.run([losses_show, pred_show], 
    #                                     feed_dict={ x_show: testset[0], 
    #                                                 y_show_: testset[1],
    #                                                 keep_prob: 1.0})
    # final_result_train = sess.run(pred_show, 
    #                             feed_dict={ x_show: batch[0], 
    #                                         y_show_: batch[1],
    #                                         keep_prob: 1.0})
    test_losses, final_result = sess.run([losses, pred_label], 
                                        feed_dict={ x: testset[0], 
                                                    l: testset[3],
                                                    keep_prob: 1.0})
    final_result_train = sess.run(pred_label, 
                                feed_dict={ x: batch[0], 
                                            l: batch[3],
                                            keep_prob: 1.0})
    print 'test:\n', np.argmax(testset[3],axis=-1),final_result,'\ntrain:\n', np.argmax(batch[3],axis=-1),final_result_train

    gt_save_test = testset[3]
    gt_save_train = batch[3]
    # final_result = reshape_result(final_result)
    # final_result_train = reshape_result(final_result_train)
    # gt_save_test = reshape_result(testset[1])
    # gt_save_train = reshape_result(batch[1])
    # gt_save_test = mu_law_decode(testset[1],quantization_channels)
    # gt_save_train = mu_law_decode(batch[1],quantization_channels)
    # final_result_train = mu_law_decode(final_result_train,quantization_channels)[:,:,np.newaxis]
    # final_result = mu_law_decode(final_result,quantization_channels)[:,:,np.newaxis]
    # result_index = np.asarray(seg_idx)
    # scipy_io.savemat(LOG_DIR_RESULT+'pred_index'+str(i)+'.mat', dict([('pred_index',result_index)]))
    scipy_io.savemat(LOG_DIR_RESULT+'GT_STFT_test'+str(i)+'.mat', dict([('GT_STFT_test',gt_save_test)]))
    # scipy_io.savemat(LOG_DIR_RESULT+'input_STFT_test'+str(i)+'.mat', dict([('input_STFT_test',testset[0])]))
    # scipy_io.savemat(LOG_DIR_RESULT+'out_test'+str(i)+'.mat',dict([('out0_test',out_[0]),('out1_test',out_[1]),('out2_test',out_[2]),('out3_test',out_[3])]))
    result_to_save = np.asarray(final_result)
    scipy_io.savemat(LOG_DIR_RESULT+'pred_STFT_test'+str(i)+'.mat', dict([('pred_STFT_test',result_to_save)]))

    scipy_io.savemat(LOG_DIR_RESULT+'GT_STFT_train'+str(i)+'.mat', dict([('GT_STFT_train',gt_save_train)]))
    # scipy_io.savemat(LOG_DIR_RESULT+'input_STFT_train'+str(i)+'.mat', dict([('input_STFT_train',batch[0])]))
    # scipy_io.savemat(LOG_DIR_RESULT+'out_train'+str(i)+'.mat',dict([('out0_train',out_train[0]),('out1_train',out_train[1]),('out2_train',out_train[2]),('out3_train',out_train[3])]))
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