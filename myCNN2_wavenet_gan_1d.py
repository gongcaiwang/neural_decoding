"""
Richard Wang
"""

import tensorflow as tf
import sys
from myLib_wavenet_gan_1d import *
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
# def deconv(input, num_outputs, kernel_size, stride, scope):
#     with tf.variable_scope(scope):
#         input = tf.expand_dims(input, axis = 1)
#         output = tf.contrib.slim.conv2d_transpose(input,num_outputs,[1,filter_width],[1,stride],scope=scope)
#         output = tf.squeeze(output,axis = 1)
#     return output

# height = 16 # STFT 40-200Hz, 10Hz/band
width = (512+64)*8 # ecog segment width 26701
depth = 64
# height_ = 128 # 0-6kHz
width_ = 512*32 # spech segment width 26701
depth_ = 1#128#256#15#32
dilations = [ 1, 4, 16, 64, 256, 1024, 4096,
              1, 4, 16, 64, 256, 1024, 4096,
              ]
# frame_length = 64
# frame_step = 32
LOG_DIR = './log/'+file_name+'_'+timestr
LOG_DIR_RESULT = LOG_DIR+'/result/'
noise_dims = 64
n_critic = 5
batch_size = 10
dilation_channels = 32
residual_channels = 16
disc_channels = 32
skip_channels = 512
filter_width = 4
L2_REG_WEIGHT = 0#0.01
initial_filter_width_ = 32
quantization_channels = 2**8
lr_init = 5e-5
decay_steps = 20000

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

from ops import *

def gradient_penalty(real, fake, f):
    def interpolate(a, b):
        shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
        alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.get_shape().as_list())
        return inter

    x = interpolate(real, fake)
    pred = f(x,train_phase=True,reuse=True)
    gradients = tf.gradients(pred, x)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, x.shape.ndims)))
    # gp = tf.reduce_mean((slopes - 1.)**2)
    gp = tf.reduce_mean((tf.maximum(0., slopes - 1)) ** 2)
    return gp

def generator_wave(input_batch, keep_prob, train_phase, reuse=None, name = None):
    with tf.variable_scope('generator', reuse = reuse):
        outputs = []
        initial_filter_width = initial_filter_width_
        current_layer = input_batch
        with tf.variable_scope('initial_layer',reuse = reuse):
            current_layer = tf.layers.dense(current_layer, 256*8, name = 'dense1',activation=functools.partial(nonlinearity,mode = 'generator'))
            current_layer = tf.reshape(current_layer,[-1,256,8])
            current_layer = deconv(current_layer,
                                residual_channels,
                                initial_filter_width,
                                stride = 2,
                                scope = 'conv0'
                                )
            current_layer = deconv(current_layer,
                                residual_channels,
                                initial_filter_width,
                                activation_fn=None,
                                stride = 4,
                                scope = 'conv1'
                                )
            current_layer = deconv(current_layer,
                                residual_channels,
                                initial_filter_width,
                                activation_fn=None,
                                stride = 4,
                                scope = 'conv2'
                                )
            current_layer = res_block_dis(current_layer, num_outputs=residual_channels,kernel_size=16,train_phase=train_phase, scope = 'res0_0', rate = 1, resample='up', mode ='generator')
            # current_layer = res_block_dis(current_layer, num_outputs=32,kernel_size=4,train_phase=train_phase, scope = 'res0_1', rate = 1, resample='up', mode ='generator')
        with tf.variable_scope('dilated_stack',reuse = reuse):
            for layer_index, dilation in enumerate(dilations):
                with tf.variable_scope('layer{}'.format(layer_index)):
                    output, current_layer = res_block(
                            current_layer, layer_index, dilation, keep_prob,train_phase,reuse)                
                    outputs.append(output)

        with tf.variable_scope('postprocessing',reuse = reuse):
            total = sum(outputs)
            transformed1 = nonlinearity(total,mode = 'generator')
            current_layer = transformed1
            # current_layer = conv(current_layer,
            #                 skip_channels,
            #                 kernel_size = 8,
            #                 activation_fn = None, 
            #                 stride = 2, scope = 'downstride1')
            current_layer = tf.contrib.layers.batch_norm(current_layer, decay = 0.999, is_training = train_phase, reuse = reuse,fused = True, scope = 'downstride1_norm')
            current_layer = nonlinearity(current_layer,mode = 'generator')
            current_layer = tf.nn.dropout(current_layer,keep_prob)

            # current_layer = conv(current_layer,
            #                 skip_channels,
            #                 kernel_size = 8,
            #                 activation_fn = None, 
            #                 stride = 2, scope = 'downstride2')
            current_layer = tf.contrib.layers.batch_norm(current_layer, decay = 0.999, is_training = train_phase, reuse = reuse,fused = True, scope = 'downstride2_norm')
            current_layer = nonlinearity(current_layer,mode = 'generator')
            current_layer = tf.nn.dropout(current_layer,keep_prob)

            current_layer = conv(current_layer,
                                skip_channels,
                                kernel_size=1,
                                activation_fn = None, 
                                scope = 'postprocess1')
            current_layer = tf.contrib.layers.batch_norm(current_layer, decay = 0.999, is_training = train_phase, reuse = reuse,fused = True, scope = 'postprocess1_norm')
            current_layer = nonlinearity(current_layer,mode = 'generator')
            # current_layer = tf.nn.dropout(current_layer,keep_prob)

            conv2 = conv(current_layer,
                        depth_,
                        kernel_size=1,
                        activation_fn = None,
                        scope = 'postprocess2')

    return conv2

def discriminator(input_batch, train_phase, reuse=None, name = None):
    with tf.variable_scope('discriminator', reuse = reuse):
        current = input_batch
        current = OptimizedResBlockDisc1(current, num_outputs=disc_channels, kernel_size = 32, scope = 'disc_block0',mode ='discriminator')
        current = res_block_dis(current, num_outputs = disc_channels, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block1',mode='discriminator',rate=2)
        current = res_block_dis(current, num_outputs = disc_channels*2, kernel_size = 8, train_phase = train_phase, resample='down', scope = 'disc_block2',mode='discriminator',rate=4)
        current = res_block_dis(current, num_outputs = disc_channels*4, kernel_size = 8, train_phase = train_phase, resample='down', scope = 'disc_block3',mode='discriminator',rate=8)
        current = res_block_dis(current, num_outputs = disc_channels*8, kernel_size = 8, train_phase = train_phase, resample='down', scope = 'disc_block4',mode='discriminator',rate=16)
        # current = res_block_dis_gate(current, num_outputs = disc_channels*8, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block4',mode='discriminator')
        # current = res_block_dis_gate(current, num_outputs = disc_channels*16, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block5',mode='discriminator')
        # current = res_block_dis(current, num_outputs = disc_channels*4, kernel_size = 16, train_phase = train_phase, scope = 'disc_block4',mode='discriminator')
        # current = res_block_dis(current, num_outputs = disc_channels*4, kernel_size = 16, train_phase = train_phase, scope = 'disc_block5',mode='discriminator')
        # current = res_block_dis(current, num_outputs = disc_channels*4, kernel_size = 16, train_phase = train_phase, scope = 'disc_block6',mode='discriminator')
        current = norm_fn_critic(current, reuse = reuse, scope = 'norm1')
        current = nonlinearity(current,mode='discriminator')
        # current = tf.reduce_mean(current, axis = 1)
        current = tf.layers.flatten(current)
        # current = tf.layers.dense(current, disc_channels*4, name = 'dense1')
        # current = norm_fn_critic(current, reuse = reuse, scope = 'norm2')
        # current = nonlinearity(current,mode='discriminator')
        current = tf.layers.dense(current, 1, name = 'dense2')
    return current

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, width, depth]) # [None, 28*28]
# y = tf.placeholder(tf.int32, shape=[None, width_, depth_])  # [None, 10]
y = tf.placeholder(tf.float32, shape=[None, width_, depth_])  # [None, 10]
# y_spectrom = tf.contrib.signal.stft(tf.squeeze(y),frame_length=64,frame_step=32)
z = tf.placeholder(tf.float32, shape=[None, noise_dims])
lr = tf.placeholder(tf.float32,shape = [])

# deno = tf.placeholder(tf.float32)
# sub = tf.placeholder(tf.float32)
# idx = tf.placeholder(tf.float32, shape=[10])
keep_prob = tf.placeholder(tf.float32)

# Reshape 'x' and 'y' to a 4D tensor (2nd dim=width, 3rd dim=height, 4th dim=Channel)
# x_spec = tf.reshape(x, [-1,height,width,depth])
# print(x_spec.get_shape)
# y_spec_ = tf.reshape(y, [-1,height_,width_,depth_])
# print(y_spec_.get_shape)

pred = generator_wave(z, keep_prob,train_phase=True, reuse = None, name='G1')
# pred_wav = tf.argmax(
#                     tf.cast(
#                         tf.nn.softmax(tf.cast(pred, tf.float64)), tf.float32),
#                     axis =-1)

l2_loss = tf.add_n([tf.nn.l2_loss(v)
                    for v in tf.trainable_variables()
                    if not('bias' in v.name)])
# y_one_hot = tf.one_hot(y,depth = quantization_channels, dtype = tf.float32) 

# losses = tf.nn.softmax_cross_entropy_with_logits(
#                                         logits=tf.reshape(pred,[-1,quantization_channels]),
#                                         labels=tf.reshape(y_one_hot,[-1,quantization_channels]))
# losses = tf.reduce_mean(losses)

disc_realfake = discriminator(tf.concat([y,pred],axis=0), train_phase = True, name='D1')
disc_real = disc_realfake[:batch_size]
disc_fake = disc_realfake[batch_size:]
wd = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
gp = gradient_penalty(y, pred, discriminator)


D_loss = -wd + gp*10.

losses = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=pred)) + L2_REG_WEIGHT * l2_loss
# losses = tf.reduce_mean(tf.losses.absolute_difference(labels=y[:,:,:], predictions=pred[:,:,:]))

G_loss_gan = - tf.reduce_mean(disc_fake)

G_loss =  G_loss_gan#G_loss_gan + G_loss_gan_2 + L2_REG_WEIGHT * l2_loss
D_loss = D_loss#D_loss + D_loss_2
theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
dopt = tf.train.AdamOptimizer(learning_rate=lr, beta1 = 0., beta2 = 0.9)#.minimize(D_loss, var_list=theta_D)
gopt = tf.train.AdamOptimizer(learning_rate=lr, beta1 = 0., beta2 = 0.9)#.minimize(G_loss, var_list=theta_G)
ggrads = gopt.compute_gradients(G_loss,var_list=theta_G)
train_step_G = gopt.apply_gradients(ggrads)
dgrads = dopt.compute_gradients(D_loss,var_list=theta_D)
train_step_D = dopt.apply_gradients(dgrads)
summ = []
# summ.append(tf.summary.scalar("sup_loss", losses_sup)) 
summ.append(tf.summary.scalar("gp", gp))
summ.append(tf.summary.scalar("wd", wd))
summ.append(tf.summary.scalar("G_loss", G_loss_gan))
summ.append(tf.summary.scalar("D_loss", D_loss))
training_summary_merge = tf.summary.merge(summ)
saver = tf.train.Saver()

sess = tf.InteractiveSession()


## Run
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph) 

# Include keep_prob in feed_dict to control dropout rate.
ecog_train, spkr_train, ecog_test, spkr_test, spkrspec_train, spkrspec_test = read_all_data()
for i in range(500): 
    lr_ = lr_init#lr_init*2.0**((-i*1.0)/(decay_steps*1.0))
    noise = np.random.standard_normal([batch_size,noise_dims])
    noise = noise / np.linalg.norm(noise,axis=1)[:,np.newaxis]
    batch = get_batch(ecog_train, spkrspec_train, seg_length=512*8, batch_size=batch_size,threshold=0.) ###########################################################################
    # Logging every ?th iteration in the training process.
    if i%10 == 0:
        # train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
        D_loss_,G_loss_,gp_,wd_,training_summary_merge_ = sess.run([D_loss,G_loss,gp,wd,training_summary_merge], feed_dict={ x:batch[0],
                                                                    y: batch[1],
                                                                    z: noise,
                                                                    keep_prob: 1.0})
        
        print("Step%d D_loss_ %g, G_loss_ %g, gp_ %g, wd_ %g" %(i,D_loss_,G_loss_,gp_,wd_))
        summary_writer.add_summary(training_summary_merge_, i)
    for j in range(n_critic):
        _ = sess.run([train_step_D],feed_dict={ x: batch[0], 
                                                            y: batch[1], 
                                                            z: noise,
                                                            lr: lr_,
                                                            keep_prob: 0.5})
    _ = sess.run([train_step_G],feed_dict={ x: batch[0], 
                                                            y: batch[1], 
                                                            z: noise,
                                                            lr: lr_,
                                                            keep_prob: 0.5})


# Evaluate our accuracy on the test data
    if i%500== 499:
        LOG_DIR_RESULT_step = LOG_DIR_RESULT + '/step_' + str(i) +'/'
        if not os.path.exists(LOG_DIR_RESULT_step):
            os.makedirs(LOG_DIR_RESULT_step)
        # testset = get_batch(ecog_test, spkrspec_test, seg_length=512*8, batch_size=batch_size,threshold=0.) ###########################################################################

        # test_accuracy = accuracy.eval(feed_dict={x: testset[0], y: testset[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
        # test_losses, final_result = sess.run([losses, pred_wav], 
        #                                     feed_dict={ x: testset[0], 
        #                                                 y: testset[1],
        #                                                 })
        # final_result_train = sess.run(pred_wav, 
        #                             feed_dict={ x: batch[0], 
        #                                         y: batch[1],
        #                                         })

        final_result = sess.run(pred, 
                                feed_dict={ #x: testset[0], 
                                            #y: testset[1],
                                            z: noise,
                                            keep_prob: 1.0
                                            })
        # final_result_train = sess.run(pred, 
        #                             feed_dict={ x: batch[0], 
        #                                         y: batch[1],
        #                                         })
        # final_result = reshape_result(final_result)
        # final_result_train = reshape_result(final_result_train)
        # gt_save_test = reshape_result(testset[1])
        # gt_save_train = reshape_result(batch[1])
        # gt_save_test = mu_law_decode(testset[1],quantization_channels)
        gt_save_train = mu_law_decode(batch[1],quantization_channels)
        # final_result_train = mu_law_decode(final_result_train,quantization_channels)[:,:,np.newaxis]
        final_result = mu_law_decode(final_result,quantization_channels)[:,:,np.newaxis]
        # result_index = np.asarray(seg_idx)
        # scipy_io.savemat(LOG_DIR_RESULT+'pred_index'+str(i)+'.mat', dict([('pred_index',result_index)]))
        # scipy_io.savemat(LOG_DIR_RESULT+'GT_STFT_test'+str(i)+'.mat', dict([('GT_STFT_test',gt_save_test)]))
        # scipy_io.savemat(LOG_DIR_RESULT+'input_STFT_test'+str(i)+'.mat', dict([('input_STFT_test',testset[0])]))
        # scipy_io.savemat(LOG_DIR_RESULT+'out_test'+str(i)+'.mat',dict([('out0_test',out_[0]),('out1_test',out_[1]),('out2_test',out_[2]),('out3_test',out_[3])]))
        result_to_save = np.asarray(final_result)
        scipy_io.savemat(LOG_DIR_RESULT+'pred_STFT_test'+'.mat', dict([('pred_STFT_test',result_to_save)]))

        scipy_io.savemat(LOG_DIR_RESULT+'GT_STFT_train'+str(i)+'.mat', dict([('GT_STFT_train',gt_save_train)]))
        # # scipy_io.savemat(LOG_DIR_RESULT+'input_STFT_train'+str(i)+'.mat', dict([('input_STFT_train',batch[0])]))
        # # scipy_io.savemat(LOG_DIR_RESULT+'out_train'+str(i)+'.mat',dict([('out0_train',out_train[0]),('out1_train',out_train[1]),('out2_train',out_train[2]),('out3_train',out_train[3])]))
        # result_to_save = np.asarray(final_result_train)
        # scipy_io.savemat(LOG_DIR_RESULT+'pred_STFT_train'+str(i)+'.mat', dict([('pred_STFT_train',result_to_save)]))

        # print("Avg Test Losses %g" %(test_losses))
        saver.save(sess, LOG_DIR+'/model')

# pdb.set_trace()



 
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