"""
Richard Wang
"""

import tensorflow as tf
import sys
from myLib_wavenet_stft_gan_z import *

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
noise_dims = 4
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
    gp = tf.reduce_mean((tf.maximum(0., slopes - 1)) ** 2)
    return gp

def reshape_result(final_result):
    final_result1 = np.concatenate((np.zeros((final_result.shape[0],final_result.shape[1],1)),final_result[:,:,:depth_/2]),axis=2)
    final_result2 = np.concatenate((np.zeros((final_result.shape[0],final_result.shape[1],1)),final_result[:,:,depth_/2:]),axis=2)
    final_result = np.concatenate((
                                    np.power(10,final_result1[:,:,:,np.newaxis])/10,
                                    final_result2[:,:,:,np.newaxis]),axis=3)
    return final_result



# def generator_wave(input_batch, keep_prob, train_phase, reuse=None, name=None):
#     current = input_batch
#     with tf.variable_scope('generator_common_layers',reuse=reuse):
#         current = tf.layers.dense(current, 16*32, name = 'dense1',activation=functools.partial(nonlinearity,mode = 'generator'))
#         current = tf.reshape(current,[-1,16,32])
#         current = conv(current,num_outputs = 32, kernel_size = 6, scope = 'conv1',activation_fn=None)       
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res1', rate = 1, resample='up', mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res2', rate = 2, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res3', rate = 4, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res4', rate = 8, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res5', rate = 1, resample='up', mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res6', rate = 2, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res7', rate = 4, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res8', rate = 8, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res9', rate = 16, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res10', rate = 1, resample='up', mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res11', rate = 2, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res12', rate = 4, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res13', rate = 8, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res14', rate = 16, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res15', rate = 32, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res16', rate = 1, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res17', rate = 2, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res18', rate = 4, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res19', rate = 8, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res20', rate = 16, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res21', rate = 32, mode ='generator')
#         common = current
#     with tf.variable_scope('generator_domain_layers_1'+name,reuse=reuse):
#         current = common
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res22', rate = 1, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res23', rate = 2, mode ='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res24', rate = 4, mode ='generator')
#         current = norm_fn(current, is_training = train_phase, reuse = reuse, scope = 'norm')
#         current = nonlinearity(current, mode ='generator')
#         out1 = conv(current, num_outputs=32,kernel_size=1, scope = 'out', activation_fn=None)
#     # with tf.variable_scope('generator_domain_layers_2'+name,reuse=reuse):
#     #     current = common
#     #     current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res22', rate = 1)
#     #     current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res23', rate = 2)
#     #     current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res24', rate = 4)
#     #     current = norm_fn(current,  is_training = train_phase, reuse = reuse, scope = 'norm')
#     #     current = nonlinearity(current)
#     #     out2 = conv(current, num_outputs=32,kernel_size=1, scope = 'out', activation_fn=None)
#     return out1

def generator_wave(input_batch, keep_prob, train_phase, reuse=None, name = None):
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
            # current_layer = tf.layers.dense(current_layer, 128*32, name = 'dense1',activation=functools.partial(nonlinearity,mode = 'generator'))
            # current_layer = tf.reshape(current_layer,[-1,128,32])
            current_layer = tf.layers.dense(current_layer, 64*16, name = 'dense1',activation=functools.partial(nonlinearity,mode = 'generator'))
            current_layer = tf.reshape(current_layer,[-1,64,16])
            current_layer = conv(current_layer,
                                residual_channels,
                                initial_filter_width,
                                activation_fn=None,
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

    return conv2#conv2[:,8:-8]


def discriminator1(input_batch, train_phase, reuse=None, name = None):
    with tf.variable_scope('discriminator1', reuse = reuse):
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

def discriminator2(input_batch, train_phase, reuse=None, name = None):
    with tf.variable_scope('discriminator2', reuse = reuse):
        current = input_batch
        current = OptimizedResBlockDisc1(current, num_outputs=disc_channels, kernel_size = 8, scope = 'disc_block0',mode ='discriminator')
        current = res_block_dis(current, num_outputs = disc_channels*2, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block1',mode='discriminator')
        current = res_block_dis(current, num_outputs = disc_channels*4, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block2',mode='discriminator')
        current = res_block_dis(current, num_outputs = disc_channels*8, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block3',mode='discriminator')
        # current = res_block_dis(current, num_outputs = disc_channels*8, kernel_size = 4, train_phase = train_phase, resample='down', scope = 'disc_block4',mode='discriminator')
        # current = res_block_dis(current, num_outputs = disc_channels*16, kernel_size = 4, train_phase = train_phase, resample='down', scope = 'disc_block5',mode='discriminator')
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

# def discriminator(input_batch, train_phase, reuse=None, share_params=False, name=None):
#     current = input_batch
#     with tf.variable_scope('discriminator_domain_layers'+name, reuse = reuse):
#         current = OptimizedResBlockDisc1(current, num_outputs=disc_channels, kernel_size = 8, scope = 'disc_block0')
#         # current = res_block_dis(current, num_outputs = disc_channels, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block1')
#     with tf.variable_scope('discriminator_common_layers', reuse = (share_params or reuse)):
#         # current = OptimizedResBlockDisc1(current, num_outputs=disc_channels, kernel_size = 8, scope = 'disc_block0')
#         current = res_block_dis(current, num_outputs = disc_channels, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block1')
#         current = res_block_dis(current, num_outputs = disc_channels, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block2')
#         current = res_block_dis(current, num_outputs = disc_channels, kernel_size = 16, train_phase = train_phase, resample='down', scope = 'disc_block3')
#         current = norm_fn(current, is_training = train_phase, reuse = reuse, scope = 'norm1')
#         current = nonlinearity(current)
#         current = tf.reduce_mean(current, axis = 1)
#         current = tf.layers.dense(current, disc_channels, name = 'dense1')
#         current = norm_fn(current, is_training = train_phase, reuse = reuse, scope = 'norm2')
#         current = nonlinearity(current)
#         current = tf.layers.dense(current, 1, name = 'dense2')
#     return current


# Placeholders
x = tf.placeholder(tf.float32, shape=[None, width, depth]) # [None, 28*28]
y_1 = tf.placeholder(tf.float32, shape=[None, width_, depth_])  # [None, 10]
y_2 = tf.placeholder(tf.float32, shape=[None, width_, depth_])  # [None, 10]
z = tf.placeholder(tf.float32, shape=[None, noise_dims])
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
# pred_1 = generator_wave(z, keep_prob,train_phase=True, reuse = None, name='G1')
pred_1 = generator_wave(z, keep_prob,train_phase=True, reuse = None, name='G1')
# pred_2 = generator_wave(z, keep_prob,train_phase=True, name='G2')
# pred_1 = generator_wave(z, keep_prob,train_phase=True, reuse = True, name='G1')
pred_2=pred_1
# pred_test = generator_wave(x, keep_prob,train_phase=False, reuse = True)
# disc_real_1 = discriminator1(y_1, train_phase = True, reuse=None,name='D1')
# disc_fake_1 = discriminator1(pred_1, train_phase = True, reuse = True, name='D1')
disc_realfake_1 = discriminator1(tf.concat([y_1,pred_1],axis=0), train_phase = True, name='D1')
disc_real_1 = disc_realfake_1[:batch_size]
disc_fake_1 = disc_realfake_1[batch_size:]
disc_realfake_2 = discriminator2(tf.concat([y_2,pred_2],axis=0), train_phase = True, name='D2')
disc_real_2 = disc_realfake_2[:batch_size]
disc_fake_2 = disc_realfake_2[batch_size:]
# disc_real_2 = discriminator2(y_2, train_phase = True,name='D2')
# disc_fake_2 = discriminator2(pred_2, train_phase = True, reuse = True, name='D2')


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

wd_1 = tf.reduce_mean(disc_real_1) - tf.reduce_mean(disc_fake_1)
gp_1 = gradient_penalty(y_1, pred_1, discriminator1)
wd_2 = tf.reduce_mean(disc_real_2) - tf.reduce_mean(disc_fake_2)
gp_2 = gradient_penalty(y_2, pred_2, discriminator2)

D_loss_1 = -wd_1 + gp_1*10.
D_loss_2 = -wd_2 + gp_2*10.

l2_loss = tf.add_n([tf.nn.l2_loss(v)
                    for v in tf.trainable_variables()
                    if not('bias' in v.name)])
# G_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake,labels= tf.one_hot(tf.ones_like(tf.argmax(disc_fake,axis=-1)),2)))
G_loss_gan_1 = - tf.reduce_mean(disc_fake_1)
G_loss_gan_2 = - tf.reduce_mean(disc_fake_2)

# y_one_hot = tf.one_hot(y_,depth = quantization_channels, dtype = tf.float32) 

# losses = tf.nn.softmax_cross_entropy_with_logits(
#                                         logits=tf.reshape(pred,[-1,quantization_channels]),
#                                         labels=tf.reshape(y_one_hot,[-1,quantization_channels]))
# losses = tf.reduce_mean(losses)
# G_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=pred)) + L2_REG_WEIGHT * l2_loss + G_loss_gan
# G_loss =  tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=pred)) + G_loss_gan
G_loss =  G_loss_gan_1#G_loss_gan_1 + G_loss_gan_2 + L2_REG_WEIGHT * l2_loss
D_loss = D_loss_1#D_loss_1 + D_loss_2
losses_sup = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_1, predictions=pred_1))
# losses_test = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=pred_test))
# losses_show = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_show_, predictions=pred_show))
# losses = tf.reduce_mean(tf.losses.absolute_difference(labels=y_[:,:,:], predictions=pred[:,:,:])
theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator1') #+ tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator2')
theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
dopt = tf.train.AdamOptimizer(learning_rate=lr, beta1 = 0., beta2 = 0.9)#.minimize(D_loss, var_list=theta_D)
gopt = tf.train.AdamOptimizer(learning_rate=lr, beta1 = 0., beta2 = 0.9)#.minimize(G_loss, var_list=theta_G)
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
summ.append(tf.summary.scalar("gp_1", gp_1))
summ.append(tf.summary.scalar("wd_1", wd_1))
summ.append(tf.summary.scalar("G_loss_1", G_loss_gan_1))
summ.append(tf.summary.scalar("D_loss_1", D_loss_1))
summ.append(tf.summary.scalar("gp_2", gp_2))
summ.append(tf.summary.scalar("wd_2", wd_2))
summ.append(tf.summary.scalar("G_loss_2", G_loss_gan_2))
summ.append(tf.summary.scalar("D_loss_2", D_loss_2))
validation_summary = tf.summary.scalar("validation_loss", losses_sup)
training_summary_merge = tf.summary.merge(summ)
# merged = tf.summary.merge_all()

saver = tf.train.Saver()

sess = tf.InteractiveSession()


## Run
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph) 

# Include keep_prob in feed_dict to control dropout rate.
# ecog_train, spkr_train, ecog_test, spkr_test, spkrspec_train, spkrspec_train2,spkrspec_test,good_idxs_train,good_idxs_test = read_all_data()
ecog_train, spkr_train, ecog_test, spkr_test, spkrspec_train, spkrspec_train2,spkrspec_test = read_all_data()
for i in range(500000): #820
    lr_ = lr_init#np.maximum(lr_init*2.0**((-i*1.0)/(decay_steps*1.0)),5e-5)
    noise = np.random.standard_normal([batch_size,noise_dims])
    noise = noise / np.linalg.norm(noise,axis=1)[:,np.newaxis]
    batch = get_batch(ecog_train, spkrspec_train, spkrspec_train2,seg_length=512, batch_size=batch_size, threshold = 0., threshold2=0.) ###########################################################################
    # batch = get_batch(ecog_train, spkrspec_train, spkrspec_train2,good_idxs_train,seg_length=512, batch_size=batch_size, threshold = 0., threshold2=0.) ###########################################################################
    # Logging every ?th iteration in the training process.
    if i%10 == 0:
        # train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
        sup_loss,l2_loss_,D_loss_1_,G_loss_gan_1_,D_loss_2_,G_loss_gan_2_,summary_train, gnorm_,dnorm_, gp_1_, wd_1_, gp_2_, wd_2_  = sess.run([losses_sup,l2_loss,D_loss_1,G_loss_gan_1,D_loss_2,G_loss_gan_2,training_summary_merge, gnorm, dnorm, gp_1, wd_1,gp_2, wd_2], 
                                                                                                    feed_dict={ x:batch[0],
                                                                                                                y_1: batch[1],
                                                                                                                y_2: batch[2],
                                                                                                                z: noise,
                                                                                                                keep_prob: 1.0})
        # testset = get_batch(ecog_test, spkrspec_test,spkrspec_test,good_idxs_test,seg_length=512, batch_size=batch_size, threshold = 1.0,threshold2 = 1.)
        testset = get_batch(ecog_test, spkrspec_test,spkrspec_test,seg_length=512, batch_size=batch_size, threshold = 1.0,threshold2 = 1.)
        test_loss,summary_validate = sess.run([losses_sup,validation_summary], feed_dict={ x:testset[0],
                                                                                            y_1: testset[1],
                                                                                            y_2: batch[2],
                                                                                            z: noise,
                                                                                            keep_prob: 1.0})
        summary_writer.add_summary(summary_train, i)
        summary_writer.add_summary(summary_validate, i)
        print("Step %d, Training Losses-- sup: %g, l2_loss: %g, d_gan_loss_1: %g, g_gan_loss_1: %g,d_gan_loss_2: %g, g_gan_loss_2: %g, wd_1: %g,wd_2: %g;    Test Losses-- %g" %(i, sup_loss, l2_loss_,  D_loss_1_ , G_loss_gan_1_,D_loss_2_ , G_loss_gan_2_, wd_1_,wd_2_, test_loss ))
        print 'gnorm ', gnorm_, ' dnorm', dnorm_, 'gp_1', gp_1_,'gp_2', gp_2_, 'lr', lr_
        # print '\nfake_label:', label_fake_, '\nreal_label:', label_real_, '\n' # ,'disc_fake_:', disc_fake_,  'disc_real_:\n', disc_real_test_
        # print 'accuracy', accuracy_

    # D_loss_,G_loss_gan_ = sess.run([D_loss,G_loss], 
    #                                 feed_dict={ x:batch[0],
    #                                             y_1: batch[1],
    #                                             keep_prob: 1.0})
    
    # if D_loss_ * 2 < G_loss_gan_:
    #     train_d = False
    #     pass
    # if train_d:
    #     _, G_loss_gan_, D_loss_ = sess.run([train_step_D,G_loss_gan,D_loss],feed_dict={ x: batch[0], 
    #                                     y_1: batch[1], 
    #                                     keep_prob: 0.5})

    # if G_loss_gan_ * 1.5 < D_loss_:
    #     train_g = False
    #     pass
    # if train_g:
    #     _, G_loss_gan_, D_loss_ =  sess.run([train_step_G,G_loss_gan,D_loss],feed_dict={ x: batch[0], 
    #                                     y_1: batch[1], 
    #                                     keep_prob: 0.5})
    for j in range(n_critic):
        _ = sess.run([train_step_D],feed_dict={ x: batch[0], 
                                        y_1: batch[1], 
                                        y_2: batch[2],
                                        z: noise,
                                        lr: lr_,
                                        keep_prob: 0.5})
    _ =  sess.run([train_step_G],feed_dict={ x: batch[0], 
                                y_1: batch[1],
                                y_2: batch[2],
                                z: noise,
                                lr: lr_,
                                keep_prob: 0.5})

    if i%500== 499:
        LOG_DIR_RESULT_step = LOG_DIR_RESULT + '/step_' + str(i) +'/'
        if not os.path.exists(LOG_DIR_RESULT_step):
            os.makedirs(LOG_DIR_RESULT_step)
        for j in range(10):
            testset = get_batch(ecog_test, spkrspec_test,spkrspec_test,seg_length=512, batch_size=batch_size, threshold = 1.,threshold2 = 1.) ###########################################################################
            batch = get_batch(ecog_train, spkrspec_train,spkrspec_train,seg_length=512, batch_size=batch_size, threshold = 0.,threshold2 = 0.)
            # testset = get_batch(ecog_test, spkrspec_test,spkrspec_test,good_idxs_test,seg_length=512, batch_size=batch_size, threshold = 1.,threshold2 = 1.) ###########################################################################
            # batch = get_batch(ecog_train, spkrspec_train,spkrspec_train,good_idxs_train,seg_length=512, batch_size=batch_size, threshold = 0.,threshold2 = 0.)
            # test_accuracy = accuracy.eval(feed_dict={x: testset[0], y_1: testset[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
            # test_losses, final_result_1, final_result_2 = sess.run([losses_sup, pred_1, pred_2], 
            #                                     feed_dict={ #x: testset[0], 
            #                                                 y_1: testset[1],
            #                                                 z: noise,
            #                                                 keep_prob: 1.0})
            test_losses, final_result_1= sess.run([losses_sup, pred_1], 
                                                feed_dict={ x: testset[0], 
                                                            y_1: testset[1],
                                                            z: noise,
                                                            keep_prob: 1.0})
            # final_result_train = sess.run(pred_1, 
            #                             feed_dict={ #x: batch[0], 
            #                                         y_1: batch[1],
            #                                         z: noise,
            #                                         keep_prob: 1.0})

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
            result_to_save_1 = np.asarray(final_result_1)
            scipy_io.savemat(LOG_DIR_RESULT_step+'pred_STFT_test_1_'+str(j)+'.mat', dict([('pred_STFT_test_1',result_to_save_1)]))
            # result_to_save_2 = np.asarray(final_result_2)
            # scipy_io.savemat(LOG_DIR_RESULT_step+'pred_STFT_test_2_'+str(j)+'.mat', dict([('pred_STFT_test_2',result_to_save_2)]))

            # scipy_io.savemat(LOG_DIR_RESULT_step+'GT_STFT_train'+str(j)+'.mat', dict([('GT_STFT_train',gt_save_train)]))
            # # scipy_io.savemat(LOG_DIR_RESULT_step+'input_STFT_train'+str(j)+'.mat', dict([('input_STFT_train',batch[0])]))
            # # scipy_io.savemat(LOG_DIR_RESULT_step+'out_train'+str(j)+'.mat',dict([('out0_train',out_train[0]),('out1_train',out_train[1]),('out2_train',out_train[2]),('out3_train',out_train[3])]))
            # result_to_save = np.asarray(final_result_train)
            # scipy_io.savemat(LOG_DIR_RESULT_step+'pred_STFT_train'+str(j)+'.mat', dict([('pred_STFT_train',result_to_save)]))

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