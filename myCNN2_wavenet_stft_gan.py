"""
Richard Wang
"""

import tensorflow as tf
import sys
from myLib_wavenet_stft_gan import *
import scipy.io as scipy_io
import functools
import os
import pdb
import re
import time

timestr = time.strftime("%Y%m%d-%H%M%S")
file_name =  os.path.basename(sys.argv[0])
file_name = re.sub('.py','',file_name)

# conv1d_transpose = tf.contrib.layers.conv2d_transpose
# height = 16 # STFT 40-200Hz, 10Hz/band
width = 512+64 # ecog segment width 26701
width_show = 1600+64
depth = 64
# height_ = 128 # 0-6kHz
width_ = 512/4 # spech segment width 26701
width_show_ = 1600/4
depth_ = 32#128#128#256#15#32 
dilations = [1, 4, 16, 64, 256]
n_critic = 5
batch_size = 32
# frame_length = 64
# frame_step = 32
LOG_DIR = './log/'+file_name+'_'+timestr
LOG_DIR_RESULT = LOG_DIR+'/result/'
dilation_channels = 32
residual_channels = 16
disc_channels = 32
skip_channels = 512
filter_width = 4
L2_REG_WEIGHT = 0#0.001
initial_filter_width_ = 32
quantization_channels = 2**8
lr_init = 5e-5
decay_steps = 1200
from ops import *

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(LOG_DIR_RESULT):
    os.makedirs(LOG_DIR_RESULT)

import shutil
for file in os.listdir('.'):
    if file.endswith(".py"):
        shutil.copy(file,LOG_DIR+'/'+file)

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
    # gp = tf.reduce_mean(tf.clip_by_value(slopes - 1., 0, np.infty)**2)
    gp = tf.reduce_mean((tf.maximum(0., slopes - 1)) ** 2)
    return gp

def reshape_result(final_result):
    final_result1 = np.concatenate((np.zeros((final_result.shape[0],final_result.shape[1],1)),final_result[:,:,:depth_/2]),axis=2)
    final_result2 = np.concatenate((np.zeros((final_result.shape[0],final_result.shape[1],1)),final_result[:,:,depth_/2:]),axis=2)
    final_result = np.concatenate((
                                    np.power(10,final_result1[:,:,:,np.newaxis])/10,
                                    final_result2[:,:,:,np.newaxis]),axis=3)
    return final_result

# def res_block(input_batch,layer_index,dilation,keep_prob, train_phase, reuse=None, scope = 'residual_block'):
#     with tf.variable_scope(scope,reuse = reuse):
#         num_outputs = dilation_channels
#         kernel_size = filter_width
#         conv_filter = conv(input_batch, num_outputs, kernel_size, 
#                             rate = dilation, activation_fn = None, 
#                             scope = 'dilation_conv{}'.format(layer_index))
#         conv_filter = tf.contrib.layers.batch_norm(conv_filter, decay = 0.95, scale = True, is_training = train_phase, reuse = reuse, scope = 'dilation_conv_norm',fused = True)
#         conv_gate = conv(input_batch, num_outputs, kernel_size, 
#                                 rate = dilation, activation_fn = None, 
#                                 scope = 'dilation_gate{}'.format(layer_index))
#         conv_gate = tf.contrib.layers.batch_norm(conv_gate, decay = 0.95, scale = True, is_training = train_phase, reuse = reuse, scope = 'dilation_gate_norm',fused = True)        
#         out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)
#         out = tf.nn.dropout(out,keep_prob)
#         transformed = conv(out,
#                             residual_channels, 
#                             kernel_size = 1,
#                             activation_fn = None,
#                             scope="dense{}".format(layer_index))
#         # transformed = tf.nn.dropout(transformed,keep_prob)
#         skip_contribution = conv(out, 
#                                 skip_channels, 
#                                 kernel_size = 1,
#                                 activation_fn = None,
#                                 scope="skip{}".format(layer_index))
#         # skip_contribution = tf.nn.dropout(skip_contribution,keep_prob)
#     return skip_contribution, input_batch+transformed

def generator_wave(input_batch, keep_prob, train_phase, reuse=None):
    with tf.variable_scope('generator', reuse = reuse):
        outputs = []
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
        with tf.variable_scope('initial_layer',reuse = reuse):
            current_layer = conv(current_layer,
                                residual_channels,
                                initial_filter_width,
                                activation_fn=None,
                                )
        with tf.variable_scope('dilated_stack',reuse = reuse):
            for layer_index, dilation in enumerate(dilations):
                with tf.variable_scope('layer{}'.format(layer_index)):
                    output, current_layer = res_block(
                            current_layer, layer_index, dilation, keep_prob,train_phase,reuse)                
                    outputs.append(output)

        with tf.variable_scope('postprocessing',reuse = reuse):
            total = sum(outputs)
            transformed1 = nonlinearity(total,mode='generator')
            current_layer = transformed1
            current_layer = conv(current_layer,
                            skip_channels,
                            kernel_size = 8,
                            activation_fn = None, 
                            stride = 2, scope = 'downstride1')
            current_layer = tf.contrib.layers.batch_norm(current_layer, decay = 0.999, is_training = train_phase, reuse = reuse, scope = 'downstride1_norm',fused = True, updates_collections=None)
            current_layer = nonlinearity(current_layer,mode='generator')
            current_layer = tf.nn.dropout(current_layer,keep_prob)

            current_layer = conv(current_layer,
                            skip_channels,
                            kernel_size = 8,
                            activation_fn = None, 
                            stride = 2, scope = 'downstride2')
            current_layer = tf.contrib.layers.batch_norm(current_layer, decay = 0.999, is_training = train_phase, reuse = reuse, scope = 'downstride2_norm',fused = True, updates_collections=None)
            current_layer = nonlinearity(current_layer,mode='generator')
            current_layer = tf.nn.dropout(current_layer,keep_prob)

            current_layer = conv(current_layer,
                                skip_channels,
                                kernel_size=1,
                                activation_fn = None, 
                                scope = 'postprocess1')
            current_layer = tf.contrib.layers.batch_norm(current_layer, decay = 0.999, is_training = train_phase, reuse = reuse, scope = 'postprocess1_norm',fused = True, updates_collections=None)
            current_layer = nonlinearity(current_layer,mode='generator')
            # current_layer = tf.nn.dropout(current_layer,keep_prob)

            conv2 = conv(current_layer,
                        depth_,
                        kernel_size=1,
                        activation_fn = None,
                        scope = 'postprocess2')

    return conv2[:,8:-8]

# def ConvMeanPool(inputs, num_outputs ,kernel_size, reuse = None, scope = 'ConvMeanPool'):
#     with tf.variable_scope(scope, reuse = reuse):
#         output = conv(inputs, num_outputs=num_outputs, kernel_size = kernel_size, activation_fn=None)
#         output = tf.nn.pool(output,[2],'AVG','SAME')
#     return output

# def MeanPoolConv(inputs, num_outputs ,kernel_size, reuse = None, scope = 'MeanPoolConv'):
#     with tf.variable_scope(scope, reuse = reuse):
#         output = inputs
#         output = tf.nn.pool(output,[2],'AVG','SAME')
#         output = conv(output, num_outputs=num_outputs, kernel_size = kernel_size, activation_fn=None)
#     return output


# def res_block_dis(inputs, num_outputs ,kernel_size, train_phase, scope, resample=None, reuse = None):
#     with tf.variable_scope(scope, reuse = reuse):
#         if resample == 'down':
#             conv_1 = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_1')
#             conv_2 = functools.partial(ConvMeanPool, num_outputs=num_outputs,kernel_size = kernel_size, scope = 'conv_2')
#             conv_shortcut = ConvMeanPool
#         elif resample=='up':
#             conv_1 = functools.partial(deconv, num_outputs=num_outputs,kernel_size=kernel_size, stride=2, activation_fn = None, scope = 'conv_1')
#             conv_2 = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_2')
#             conv_shortcut = functools.partial(deconv,stride=2, activation_fn = None)
#         elif resample == None:
#             conv_1 = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_1')
#             conv_2 = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_2')
#             conv_shortcut = functools.partial(conv,activation_fn = None)
#         if inputs.get_shape().as_list()[-1] == num_outputs and resample == None:
#             shortcut = inputs
#         else:
#             shortcut = conv_shortcut(inputs, num_outputs=num_outputs, kernel_size=1, scope='shortcut')

#         output = inputs
#         output = tf.contrib.layers.layer_norm(output, reuse = reuse, scope = 'norm1')
#         output = nonlinearity(output)
#         output = conv_1(output, scope='conv1')
#         output = tf.contrib.layers.layer_norm(output,reuse = reuse, scope = 'norm2')
#         output = nonlinearity(output)
#         output = conv_2(output, scope='conv2')

#         return shortcut + output

# def OptimizedResBlockDisc1(inputs, num_outputs, kernel_size, scope, reuse = None):
#     with tf.variable_scope(scope, reuse = reuse):
#         shortcut = MeanPoolConv(inputs, num_outputs=num_outputs, kernel_size=1, scope='shortcut')
#         output = inputs
#         output = conv(output, num_outputs= num_outputs, kernel_size = kernel_size, scope = 'conv1')
#         output = nonlinearity(output,mode='discriminator')
#         output = ConvMeanPool(output,num_outputs=num_outputs, kernel_size=kernel_size, scope='conv2')
#         return shortcut + output
    

def discriminator(input_batch, train_phase, reuse=None):
    with tf.variable_scope('discriminator', reuse = reuse):
        current = input_batch
        current = OptimizedResBlockDisc1(current, num_outputs=disc_channels, kernel_size = 8, scope = 'disc_block0',mode ='discriminator')
        current = res_block_dis(current, num_outputs = disc_channels, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block1',mode='discriminator')
        current = res_block_dis(current, num_outputs = disc_channels*2, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block2',mode='discriminator')
        current = res_block_dis(current, num_outputs = disc_channels*4, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block3',mode='discriminator')
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
y_ = tf.placeholder(tf.float32, shape=[None, width_, depth_])  # [None, 10]
lr = tf.placeholder(tf.float32,shape = [])
# x_show = tf.placeholder(tf.float32, shape=[None, width_show, depth]) # [None, 28*28]
# y_show_ = tf.placeholder(tf.float32, shape=[None, width_show_, depth_])  # [None, 10]
# l = tf.placeholder(tf.int32, shape= [None,])
# label = tf.one_hot(l, 2)

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

# x = tf.random_normal(tf.shape(x))
pred = generator_wave(x, keep_prob,train_phase=True, reuse = None)
# pred_test = generator_wave(x, keep_prob,train_phase=False, reuse = True)
# disc_real = discriminator(y_, train_phase = True, reuse=None)
# disc_fake = discriminator(pred, train_phase = True, reuse=True)
disc_realfake = discriminator(tf.concat([y_,pred],axis=0), train_phase = True)
disc_real = disc_realfake[:batch_size]
disc_fake = disc_realfake[batch_size:]
# disc_fake = discriminator(pred, train_phase = False, reuse=True)
# disc_real = discriminator(y_, train_phase = False, reuse=True)
# label_fake = tf.argmax(disc_fake, axis = -1)
# label_real = tf.argmax(disc_real, axis = -1)
# accuracy = tf.reduce_mean(tf.to_float(tf.equal(label_fake, tf.zeros_like(label_fake))) + tf.to_float(tf.equal(label_real, tf.ones_like(label_real))))/2

# pred_show = generator_wave(x_show, keep_prob,train_phase=True, reuse = True)
# pred_wav = tf.argmax(
#                     tf.cast(
#                         tf.nn.softmax(tf.cast(pred, tf.float64)), tf.float32),
#                     axis =-1)
# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_real,labels= tf.one_hot(tf.ones_like(tf.argmax(disc_real,axis=-1)),2)))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake,labels= tf.one_hot(tf.zeros_like(tf.argmax(disc_fake,axis=-1)),2)))
# D_loss = (D_loss_real+D_loss_fake)/2

wd = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
gp = gradient_penalty(y_, pred, discriminator)
D_loss = -wd + gp*10.

l2_loss = tf.add_n([tf.nn.l2_loss(v)
                    for v in tf.trainable_variables()
                    if not('bias' in v.name)])
# G_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake,labels= tf.one_hot(tf.ones_like(tf.argmax(disc_fake,axis=-1)),2)))
G_loss_gan = - tf.reduce_mean(disc_fake)

# y_one_hot = tf.one_hot(y_,depth = quantization_channels, dtype = tf.float32) 

# losses = tf.nn.softmax_cross_entropy_with_logits(
#                                         logits=tf.reshape(pred,[-1,quantization_channels]),
#                                         labels=tf.reshape(y_one_hot,[-1,quantization_channels]))
# losses = tf.reduce_mean(losses)
# G_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=pred)) + L2_REG_WEIGHT * l2_loss + G_loss_gan
G_loss =  G_loss_gan + 20*tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=pred))
losses_sup = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=pred))
# losses_test = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=pred_test))
# losses_show = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_show_, predictions=pred_show))
# losses = tf.reduce_mean(tf.losses.absolute_difference(labels=y_[:,:,:], predictions=pred[:,:,:])
theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
dopt = tf.train.AdamOptimizer(learning_rate=lr, beta1 = 0.,beta2 = 0.9)#.minimize(D_loss, var_list=theta_D)
gopt = tf.train.AdamOptimizer(learning_rate=lr, beta1 = 0.,beta2 = 0.9)#.minimize(G_loss, var_list=theta_G)
# with tf.control_dependencies(update_ops):
ggrads = gopt.compute_gradients(G_loss,var_list=theta_G)
train_step_G = gopt.apply_gradients(ggrads)
gnorm = tf.global_norm([i[0] for i in ggrads])
dgrads = dopt.compute_gradients(D_loss,var_list=theta_D)
train_step_D = dopt.apply_gradients(dgrads)
dnorm = tf.global_norm([i[0] for i in dgrads])
    # train_step_G = tf.train.AdamOptimizer(1e-3).minimize(G_loss)  # 1e-4

summ = []

summ.append(tf.summary.scalar("sup_loss", losses_sup)) 
summ.append(tf.summary.scalar("gp", gp)) 
summ.append(tf.summary.scalar("wd", wd)) 
summ.append(tf.summary.scalar("G_loss", G_loss_gan)) 
summ.append(tf.summary.scalar("D_loss", D_loss)) 
training_summary_merge = tf.summary.merge(summ)
validation_summary = tf.summary.scalar("validation_loss", losses_sup)
# merged = tf.summary.merge_all()

saver = tf.train.Saver()

sess = tf.InteractiveSession()


## Run
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph) 

# Include keep_prob in feed_dict to control dropout rate.
ecog_train, spkr_train, ecog_test, spkr_test, spkrspec_train, spkrspec_test = read_all_data()
for i in range(500000): #820
    lr_ = lr_init#np.maximum(lr_init*2.0**((-i*1.0)/(decay_steps*1.0)),5e-5)
    batch = get_batch(ecog_train, spkrspec_train, seg_length=512, batch_size=batch_size, threshold = 0.0) ###########################################################################
    # Logging every ?th iteration in the training process.
    if i%10 == 0:
        # train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
        sup_loss,l2_loss_,D_loss_,G_loss_gan_,summary_train, gnorm_,dnorm_, gp_, wd_  = sess.run([losses_sup,l2_loss,D_loss,G_loss_gan,training_summary_merge, gnorm, dnorm, gp, wd], 
                                                                                                    feed_dict={ x:batch[0],
                                                                                                                y_: batch[1],
                                                                                                                keep_prob: 1.0})
        testset = get_batch(ecog_test, spkrspec_test, seg_length=512, batch_size=batch_size, threshold = 1.2)
        test_loss,summary_validate = sess.run([losses_sup,validation_summary], feed_dict={ x:testset[0],
                                                        y_: testset[1],
                                                        keep_prob: 1.0})
        summary_writer.add_summary(summary_train, i)
        summary_writer.add_summary(summary_validate, i)
        print("Step %d, Training Losses-- sup: %g, l2_loss: %g, g_gan_loss: %g, d_gan_loss: %g, wd: %g;    Test Losses-- %g" %(i, sup_loss, l2_loss_, G_loss_gan_,  D_loss_ , wd_, test_loss ))
        print 'gnorm ', gnorm_, ' dnorm', dnorm_, 'gp', gp_

    # D_loss_,G_loss_gan_ = sess.run([D_loss,G_loss], 
    #                                 feed_dict={ x:batch[0],
    #                                             y_: batch[1],
    #                                             keep_prob: 1.0})
    
    # if D_loss_ * 2 < G_loss_gan_:
    #     train_d = False
    #     pass
    # if train_d:
    #     _, G_loss_gan_, D_loss_ = sess.run([train_step_D,G_loss_gan,D_loss],feed_dict={ x: batch[0], 
    #                                     y_: batch[1], 
    #                                     keep_prob: 0.5})

    # if G_loss_gan_ * 1.5 < D_loss_:
    #     train_g = False
    #     pass
    # if train_g:
    #     _, G_loss_gan_, D_loss_ =  sess.run([train_step_G,G_loss_gan,D_loss],feed_dict={ x: batch[0], 
    #                                     y_: batch[1], 
    #                                     keep_prob: 0.5})
    for j in range(n_critic):
        _, G_loss_gan_, D_loss_ = sess.run([train_step_D,G_loss_gan,D_loss],feed_dict={ x: batch[0], 
                                        y_: batch[1],
                                        lr: lr_,
                                        keep_prob: 0.5})
    _, G_loss_gan_, D_loss_ =  sess.run([train_step_G,G_loss_gan,D_loss],feed_dict={ x: batch[0], 
                                y_: batch[1],
                                lr: lr_,
                                keep_prob: 0.5})

    if i%500== 499:
        LOG_DIR_RESULT_step = LOG_DIR_RESULT + '/step_' + str(i) +'/'
        if not os.path.exists(LOG_DIR_RESULT_step):
            os.makedirs(LOG_DIR_RESULT_step)
        for j in range(10):
            testset = get_batch(ecog_test, spkrspec_test, seg_length=512, batch_size=batch_size, threshold = 1.2) ###########################################################################
            batch = get_batch(ecog_train, spkrspec_train, seg_length=512, batch_size=batch_size, threshold = 1.2)
            # test_accuracy = accuracy.eval(feed_dict={x: testset[0], y_: testset[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
            test_losses, final_result = sess.run([losses_sup, pred], 
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
            # scipy_io.savemat(LOG_DIR_RESULT+'pred_index'+str(j)+'.mat', dict([('pred_index',result_index)]))
            scipy_io.savemat(LOG_DIR_RESULT_step+'GT_STFT_test'+str(j)+'.mat', dict([('GT_STFT_test',gt_save_test)]))
            # scipy_io.savemat(LOG_DIR_RESULT_step+'input_STFT_test'+str(j)+'.mat', dict([('input_STFT_test',testset[0])]))
            # scipy_io.savemat(LOG_DIR_RESULT_step+'out_test'+str(j)+'.mat',dict([('out0_test',out_[0]),('out1_test',out_[1]),('out2_test',out_[2]),('out3_test',out_[3])]))
            result_to_save = np.asarray(final_result)
            scipy_io.savemat(LOG_DIR_RESULT_step+'pred_STFT_test'+str(j)+'.mat', dict([('pred_STFT_test',result_to_save)]))

            scipy_io.savemat(LOG_DIR_RESULT_step+'GT_STFT_train'+str(j)+'.mat', dict([('GT_STFT_train',gt_save_train)]))
            # scipy_io.savemat(LOG_DIR_RESULT_step+'input_STFT_train'+str(j)+'.mat', dict([('input_STFT_train',batch[0])]))
            # scipy_io.savemat(LOG_DIR_RESULT_step+'out_train'+str(j)+'.mat',dict([('out0_train',out_train[0]),('out1_train',out_train[1]),('out2_train',out_train[2]),('out3_train',out_train[3])]))
            result_to_save = np.asarray(final_result_train)
            scipy_io.savemat(LOG_DIR_RESULT_step+'pred_STFT_train'+str(j)+'.mat', dict([('pred_STFT_train',result_to_save)]))

            print("Avg Test Losses %g" %(test_losses))
        saver.save(sess, LOG_DIR+'/model'+'_step_'+str(i))


 
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