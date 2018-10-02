"""
Richard Wang
"""

import tensorflow as tf
import sys
from myLib_wavenet_stft_2d import *
from ops import *
import scipy.io as scipy_io
import os
import pdb
import re
import time

timestr = time.strftime("%Y%m%d-%H%M%S")
file_name =  os.path.basename(sys.argv[0])
file_name = re.sub('.py','',file_name)
# conv1d_transpose = tf.contrib.layers.conv2d_transpose
conv = tf.contrib.slim.convolution
deconv = tf.contrib.slim.conv2d_transpose
# def deconv(input, num_outputs, kernel_size, stride, scope):
#     output = tf.contrib.slim.conv2d_transpose(input,num_outputs,[1,filter_width],[1,stride],scope=scope)
#     return output
 
 
# height = 16 # STFT 40-200Hz, 10Hz/band
width = 1600+64 # ecog segment width 26701
width_show = 1600+64
depth = 64
# height_ = 128 # 0-6kHz
width_ = 1600/4 # spech segment width 26701
width_show_ = 1600/4
depth_ = 32#128#128#256#15#32 
dilations = [1, 2, 4, 8, 16, 32, 64,
             1, 2, 4, 8, 16, 32, 64,
             1, 2, 4, 8, 16, 32, 64,
             1, 2, 4, 8, 16, 32, 64,
             1, 2, 4, 8, 16, 32, 64,
             1, 2, 4, 8, 16, 32, 64,]
# frame_length = 64
# frame_step = 32
LOG_DIR = './log/'+file_name+'_'+timestr
LOG_DIR_RESULT = LOG_DIR+'/result/'
dilation_channels = 32
residual_channels = 16
skip_channels = 512
filter_width = 4
L2_REG_WEIGHT = 0#0.001
initial_filter_width_ = 32
quantization_channels = 2**8

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

def network_wavenet(input_batch, keep_prob, train_phase, reuse=None):
    with tf.variable_scope('generator',reuse=reuse):
        current_layer = tf.expand_dims(input_batch,-2)
        current_layer = conv(current_layer, num_outputs = 64, kernel_size = [6,1], scope = 'conv1')       
        current_layer = deconv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res1', stride = [1,2],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm1'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res2', rate = [2,1],activation_fn=None)
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res3', rate = [4,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm2'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res4', rate = [8,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm3'))
        current_layer = deconv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res5',stride = [1,2],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm4'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res6', rate = [2,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm5'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res7', rate = [4,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm6'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res8', rate = [8,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm7'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res9', rate = [16,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm8'))
        current_layer = deconv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res10',  stride = [1,2],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm9'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res11', rate = [2,2],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm10'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res12', rate = [4,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm11'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res13', rate = [8,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm12'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res13_1', rate = [16,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm13'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res13_2', rate = [32,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm14'))
        current_layer = deconv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res14', stride = [1,2],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm15'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res15', rate = [2,2],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm16'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res16', rate = [4,4],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm17'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res17', rate = [8,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm18'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res18', rate = [16,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm19'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res19', rate = [32,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm20'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res20', rate = [64,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm21'))
        current_layer = deconv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res21',  stride = [1,2],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm22'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res22', rate = [2,2],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm23'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res23', rate = [4,4],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm24'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res24', rate = [8,8],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm25'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res25', rate = [14,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm26'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,2], scope = 'res26', rate = [32,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm27'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,4], scope = 'res27', rate = [64,1],activation_fn=None)
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm28'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,4], scope = 'res28', rate = [128,1],activation_fn=None)
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,4], scope = 'res29',activation_fn=None, stride = [2,1])
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm29'))
        current_layer = conv(current_layer, num_outputs=32,kernel_size=[2,4], scope = 'res30',activation_fn=None, stride = [2,1])
        current_layer = tf.nn.relu(tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm30'))
        current_layer = tf.squeeze(conv(current_layer, num_outputs=1,kernel_size=1, scope = 'res31', rate = 1,activation_fn=None))
    return current_layer[:,8:-8]


# Placeholders
x = tf.placeholder(tf.float32, shape=[None, width, depth]) # [None, 28*28]
y_ = tf.placeholder(tf.float32, shape=[None, width_, depth_])  # [None, 10]
# x_show = tf.placeholder(tf.float32, shape=[None, width_show, depth]) # [None, 28*28]
# y_show_ = tf.placeholder(tf.float32, shape=[None, width_show_, depth_])  # [None, 10]
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

pred = network_wavenet(x, keep_prob,train_phase=True, reuse = None)
# pred_test = network_wavenet(x, keep_prob,train_phase=False, reuse = True)
# # pred_test = network_wavenet(x, keep_prob,train_phase=True, reuse = True)
# pred_show = network_wavenet(x_show, keep_prob,train_phase=False, reuse = True)
# # pred_show = network_wavenet(x_show, keep_prob,train_phase=True, reuse = True)
pred_wav = tf.argmax(
                    tf.cast(
                        tf.nn.softmax(tf.cast(pred, tf.float64)), tf.float32),
                    axis =-1)
l2_loss = tf.add_n([tf.nn.l2_loss(v)
                    for v in tf.trainable_variables()
                    if not('bias' in v.name)])
# y_one_hot = tf.one_hot(y_,depth = quantization_channels, dtype = tf.float32) 

# losses = tf.nn.softmax_cross_entropy_with_logits(
#                                         logits=tf.reshape(pred,[-1,quantization_channels]),
#                                         labels=tf.reshape(y_one_hot,[-1,quantization_channels]))
# losses = tf.reduce_mean(losses)
losses = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=pred)) + L2_REG_WEIGHT * l2_loss
# loss_bias = tf.reshape(tf.constant(np.array(xrange(depth_))+1.0,dtype = tf.float32),[1,1,-1,1])
# loss_bias = tf.reshape(tf.constant(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0])*20.0+1.0,dtype = tf.float32),[1,1,-1])
# losses = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_*loss_bias, predictions=pred*loss_bias)) + L2_REG_WEIGHT * l2_loss
losses_test = losses
losses_show = losses
# losses_test = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=pred_test))
# losses_show = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_show_, predictions=pred_show))
# losses = tf.reduce_mean(tf.losses.absolute_difference(labels=y_[:,:,:], predictions=pred[:,:,:]))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(losses)  # 1e-4

training_summary = tf.summary.scalar("training_loss", losses_test)
validation_summary = tf.summary.scalar("validation_loss", losses_test)

# merged = tf.summary.merge_all()

saver = tf.train.Saver()

sess = tf.InteractiveSession()


## Run
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph) 

# Include keep_prob in feed_dict to control dropout rate.
ecog_train, spkr_train, ecog_test, spkr_test, spkrspec_train, spkrspec_test = read_all_data()
for i in range(1500): #820
    batch = get_batch(ecog_train, spkrspec_train, seg_length=1600 , batch_size=10, threshold = 0.0) ###########################################################################
    # Logging every ?th iteration in the training process.
    if i%10 == 0:
        # train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
        train_loss,l2_loss_,summary_train = sess.run([losses_test,l2_loss,training_summary], feed_dict={ x:batch[0],
                                                        y_: batch[1],
                                                        keep_prob: 1.0})
        testset = get_batch(ecog_test, spkrspec_test, seg_length=1600, batch_size=10, threshold = 1.2)
        test_loss,summary_validate = sess.run([losses_test,validation_summary], feed_dict={ x:testset[0],
                                                        y_: testset[1],
                                                        keep_prob: 1.0})
        summary_writer.add_summary(summary_train, i)
        summary_writer.add_summary(summary_validate, i)
        print("Step %d, Training Losses %g, Test Losses %g, l2_loss %g" %(i, train_loss, test_loss, l2_loss_))
    sess.run(train_step,feed_dict={ x: batch[0], 
                                    y_: batch[1], 
                                    keep_prob: 0.5})


# Evaluate our accuracy on the test data

for i in range(10):
    testset = get_batch(ecog_test, spkrspec_test, seg_length=1600, batch_size=10, threshold = 1.2) ###########################################################################
    batch = get_batch(ecog_train, spkrspec_train, seg_length=1600, batch_size=10, threshold = 1.2)
    # test_accuracy = accuracy.eval(feed_dict={x: testset[0], y_: testset[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
    # test_losses, final_result = sess.run([losses_show, pred_show], 
    #                                     feed_dict={ x_show: testset[0], 
    #                                                 y_show_: testset[1],
    #                                                 keep_prob: 1.0})
    # final_result_train = sess.run(pred_show, 
    #                             feed_dict={ x_show: batch[0], 
    #                                         y_show_: batch[1],
    #                                         keep_prob: 1.0})
    test_losses, final_result = sess.run([losses, pred], 
                                        feed_dict={ x: testset[0], 
                                                    y_: testset[1],
                                                    keep_prob: 1.0})
    final_result_train = sess.run(pred, 
                                feed_dict={ x: batch[0], 
                                            y_: batch[1],
                                            keep_prob: 1.0})

    gt_save_test = testset[1]
    gt_save_train = batch[1]
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
    scipy_io.savemat(LOG_DIR_RESULT+'input_STFT_test'+str(i)+'.mat', dict([('input_STFT_test',testset[0])]))
    # scipy_io.savemat(LOG_DIR_RESULT+'out_test'+str(i)+'.mat',dict([('out0_test',out_[0]),('out1_test',out_[1]),('out2_test',out_[2]),('out3_test',out_[3])]))
    result_to_save = np.asarray(final_result)
    scipy_io.savemat(LOG_DIR_RESULT+'pred_STFT_test'+str(i)+'.mat', dict([('pred_STFT_test',result_to_save)]))

    scipy_io.savemat(LOG_DIR_RESULT+'GT_STFT_train'+str(i)+'.mat', dict([('GT_STFT_train',gt_save_train)]))
    scipy_io.savemat(LOG_DIR_RESULT+'input_STFT_train'+str(i)+'.mat', dict([('input_STFT_train',batch[0])]))
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