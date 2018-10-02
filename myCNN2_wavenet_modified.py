"""
Richard Wang
"""

import tensorflow as tf
import sys
from myLib_wavenet import *
import scipy.io as scipy_io
import os
import pdb

# conv1d_transpose = tf.contrib.layers.conv2d_transpose
conv = tf.contrib.slim.convolution
def deconv(input, num_outputs, kernel_size, stride, scope, activation_fn = None):
    with tf.variable_scope(scope):
        input = tf.expand_dims(input, axis = 1)
        output = tf.contrib.slim.conv2d_transpose(input,num_outputs,[1,filter_width],[1,stride],scope=scope, activation_fn=activation_fn)
        output = tf.squeeze(output,axis = 1)
    return output

# height = 16 # STFT 40-200Hz, 10Hz/band
width = 320+64 # ecog segment width 26701
depth = 64
# height_ = 128 # 0-6kHz
width_ = 320*32 # spech segment width 26701
depth_ = 1#128#256#15#32
dilations =  [[1, 2, 4, 8, 16, 32, 64,128,256,512],
              [1, 2, 4, 8, 16, 32, 64,128,256,512],
              [1, 2, 4, 8, 16, 32, 64,128,256,512],
              [1, 2, 4, 8, 16, 32, 64,128,256,512],
              [1, 2, 4, 8, 16, 32, 64,128,256,512]]
# frame_length = 64
# frame_step = 32
LOG_DIR = './log'
LOG_DIR_RESULT = LOG_DIR+'/result/'
dilation_channels = 32
residual_channels = 16
skip_channels = 512
filter_width = 2
L2_REG_WEIGHT = 0#0.01
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

def upsample(input,size):
    size = [1,size]
    input = tf.expand_dims(input,axis=1)
    out = tf.image.resize_images( input,
                                size,
                                method=ResizeMethod.BILINEAR,
                                align_corners=False)
    out = tf.squeeze(out,axis = 1)
    return out


def res_block(input_batch,stride_index,layer_index,dilation,scope = 'residual_block'):
    with tf.variable_scope(scope):
        num_outputs = dilation_channels
        kernel_size = filter_width
        conv_filter = conv(input_batch, num_outputs, kernel_size, 
                            rate = dilation, activation_fn = None, 
                            scope = 'stride{}_dilation_conv{}'.format(stride_index,layer_index))
        conv_gate = conv(input_batch, num_outputs, kernel_size, 
                                rate = dilation, activation_fn = None, 
                                scope = 'stride{}_dilation_gate{}'.format(stride_index,layer_index))
        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)
        transformed = conv(out,
                            residual_channels, 
                            kernel_size = 1,
                            activation_fn = None,
                            scope="stride{}_dense{}".format(stride_index,layer_index))
        if stride_index==0:
            num_skip_ = skip_channels*2
        else:
            num_skip_ = skip_channels
        skip_contribution = conv(out, 
                                num_skip_, 
                                kernel_size = 1,
                                activation_fn = None,
                                scope="stride{}_skip{}".format(stride_index,layer_index))
    return skip_contribution, input_batch+transformed

def network_wavenet(input_batch):
    skip_contributions=[]
    initial_filter_width = initial_filter_width_
    current_layer = input_batch
    # current_layer = deconv(current_layer,
    #                 skip_channels,
    #                 kernel_size = 16,
    #                 stride = 4, scope = 'upstride1')
    # current_layer = deconv(current_layer,
    #             skip_channels,
    #             kernel_size = 16,
    #             stride = 4,scope = 'upstride2')
    # current_layer = deconv(current_layer,
    #             skip_channels,
    #             kernel_size = 8,
    #             stride = 2,scope = 'upstride3')
    current_layer = conv(current_layer,
                        residual_channels,
                        initial_filter_width,
                        activation_fn=None,
                        scope = 'initial_layer'
                        )
    with tf.variable_scope('dilated_stack'):
        for stride_index in xrange(len(dilations)):
            outputs = []
            for layer_index in xrange(len(dilations[0])):
                dilation = dilations[stride_index][layer_index]
                with tf.variable_scope('stride{}layer{}'.format(stride_index,layer_index)):
                    output, current_layer = res_block(
                            current_layer, stride_index, layer_index, dilation)                
                    outputs.append(output)
            skip_contributions.append((outputs))
            current_layer = deconv(current_layer,
                                    residual_channels,
                                    kernel_size = 8,
                                    activation_fn = None,
                                    stride = 2, scope = 'upstride{}'.format(stride_index))

    with tf.variable_scope('postprocessing'):
        totals = []
        # todo: make the process nonlinear
        for i in xrange(len(dilations)):
            total = sum(skip_contributions[i])
            totals.append(total)
        for i in xrange(len(dilations)):
            input1 = totals[i]
            if i == 0:
                input1 = input1 
            else:
                input1 = tf.concat([input1,current_output],2)
            out1 = conv(input1,
                        num_outputs=skip_channels*2,
                        kernel_size = 4,
                        activation_fn=None,
                        scope = 'wavelety{}'.format(i))
            current_output = tf.reshape(out1,[-1,out1.get_shape().as_list()[1]*2,out1.get_shape().as_list()[2]/2])

        transformed1 = tf.nn.relu(current_output)
        current_layer = transformed1
        current_layer = conv(current_layer,
                    skip_channels,
                    kernel_size=1,scope = 'postprocess1')
        conv2 = conv(current_layer,
                    quantization_channels,
                    kernel_size=1,
                    activation_fn = None,
                    scope = 'postprocess2')

    return conv2[:,32*32:-32*32]


# Placeholders
x = tf.placeholder(tf.float32, shape=[None, width, depth]) # [None, 28*28]
y_ = tf.placeholder(tf.int32, shape=[None, width_, depth_])  # [None, 10]
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

pred = network_wavenet(x)
pred_wav = tf.argmax(
                    tf.cast(
                        tf.nn.softmax(tf.cast(pred, tf.float64)), tf.float32),
                    axis =-1)

l2_loss = tf.add_n([tf.nn.l2_loss(v)
                    for v in tf.trainable_variables()
                    if not('bias' in v.name)])
y_one_hot = tf.one_hot(y_,depth = quantization_channels, dtype = tf.float32) 

losses = tf.nn.softmax_cross_entropy_with_logits(
                                        logits=tf.reshape(pred,[-1,quantization_channels]),
                                        labels=tf.reshape(y_one_hot,[-1,quantization_channels]))
losses = tf.reduce_mean(losses)
# losses = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=pred)) + L2_REG_WEIGHT * l2_loss
# losses = tf.reduce_mean(tf.losses.absolute_difference(labels=y_[:,:,:], predictions=pred[:,:,:]))

train_step = tf.train.AdamOptimizer(1e-3).minimize(losses)  # 1e-4

training_summary = tf.summary.scalar("training_loss", losses)
validation_summary = tf.summary.scalar("validation_loss", losses)

saver = tf.train.Saver()

sess = tf.InteractiveSession()


## Run
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph) 

# Include keep_prob in feed_dict to control dropout rate.
ecog_train, spkr_train, ecog_test, spkr_test, spkrspec_train, spkrspec_test = read_all_data()
for i in range(3000): 
    batch = get_batch(ecog_train, spkrspec_train, seg_length=320, batch_size=10, threshold = 0) ###########################################################################
    # Logging every ?th iteration in the training process.
    if i%5 == 0:
        # train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
        train_loss,l2_loss_,summary_train = sess.run([losses,l2_loss,training_summary], feed_dict={ x:batch[0],
                                                        y_: batch[1]})
        testset = get_batch(ecog_test, spkrspec_test, seg_length=320, batch_size=10, threshold = 1.2)
        test_loss,summary_validate = sess.run([losses,validation_summary], feed_dict={ x:testset[0],
                                                        y_: testset[1]})
        summary_writer.add_summary(summary_train, i)
        summary_writer.add_summary(summary_validate, i)
        print("Step %d, Training Losses %g, Test Losses %g, l2_loss %g" %(i, train_loss, test_loss, l2_loss_))
    sess.run(train_step,feed_dict={ x: batch[0], 
                                    y_: batch[1], 
                                    })


# Evaluate our accuracy on the test data

for i in range(10):
    testset = get_batch(ecog_test, spkrspec_test, seg_length=320, batch_size=10) ###########################################################################

    # test_accuracy = accuracy.eval(feed_dict={x: testset[0], y_: testset[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
    test_losses, final_result = sess.run([losses, pred_wav], 
                                        feed_dict={ x: testset[0], 
                                                    y_: testset[1],
                                                    })
    final_result_train = sess.run(pred_wav, 
                                feed_dict={ x: batch[0], 
                                            y_: batch[1],
                                            })

    # final_result = reshape_result(final_result)
    # final_result_train = reshape_result(final_result_train)
    # gt_save_test = reshape_result(testset[1])
    # gt_save_train = reshape_result(batch[1])
    gt_save_test = mu_law_decode(testset[1],quantization_channels)
    gt_save_train = mu_law_decode(batch[1],quantization_channels)
    final_result_train = mu_law_decode(final_result_train,quantization_channels)[:,:,np.newaxis]
    final_result = mu_law_decode(final_result,quantization_channels)[:,:,np.newaxis]
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