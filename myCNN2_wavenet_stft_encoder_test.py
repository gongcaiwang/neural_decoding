"""
Richard Wang
"""

import tensorflow as tf
import sys
from myLib_wavenet_stft_encoder import *
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
def deconv(input, num_outputs, kernel_size, stride, scope):
    with tf.variable_scope(scope):
        input = tf.expand_dims(input, axis = 1)
        output = tf.contrib.slim.conv2d_transpose(input,num_outputs,[1,filter_width],[1,stride],scope=scope)
        output = tf.squeeze(output,axis = 1)
    return output
 
 
# height = 16 # STFT 40-200Hz, 10Hz/band
width = (512)/4 # ecog segment width 26701
width_show = (512)/4
depth = 64
# height_ = 128 # 0-6kHz
width_ = 512/4 # spech segment width 26701
width_show_ = 9289-32#512/4
depth_ = 32#128#128#256#15#32 
dilations = [1, 2, 4, 8, 16,
             1, 2, 4, 8, 16,]
             # 1, 2, 4, 8, 16, 32, 64,
             # 1, 2, 4, 8, 16, 32, 64,
             # 1, 2, 4, 8, 16, 32, 64,
             # 1, 2, 4, 8, 16, 32, 64,]
# frame_length = 64
# frame_step = 32
LOG_DIR = './log/'+file_name+'_'+timestr
LOG_DIR_RESULT = LOG_DIR+'/result/'
# LOAD_DIR = './log/'+'myCNN2_wavenet_stft_encoder_20180619-004708'+'/model_step_2999' #wavenet
# LOAD_DIR = './log/'+'myCNN2_wavenet_stft_encoder_20180619-215247'+'/model_step_2999' #cnn
# LOAD_DIR = './log/'+'myCNN2_wavenet_stft_cv_hd06_encode_20180714-234843'+'/model_step_3000_cv0' #cnn hd06
LOAD_DIR = './log/'+'myCNN2_wavenet_stft_cv_hd06_encode_20180715-002056'+'/model_step_3000_cv0' #cnn hd06 l2_reg
dilation_channels = 32
residual_channels = 16
skip_channels = 512#64
filter_width = 2
L2_REG_WEIGHT = 0#0.001
initial_filter_width_ = 32
quantization_channels = 2**8
lr = 1e-4

from ops import *

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(LOG_DIR_RESULT):
    os.makedirs(LOG_DIR_RESULT)

import shutil
for file in os.listdir('.'):
    if file.endswith(".py"):
        shutil.copy(file,LOG_DIR+'/'+file)

def causal_conv(value, filter_, dilation=1, name='causal_conv'):
    with tf.name_scope(name):
        filter_width = tf.shape(filter_)[0]
        if dilation > 1:
            transformed = time_to_batch(value, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1,
                                padding='SAME')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(value, filter_, stride=1, padding='SAME')
        # Remove excess elements at the end.
        out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, out_width, -1])
        return result

def reshape_result(final_result):
    final_result1 = np.concatenate((np.zeros((final_result.shape[0],final_result.shape[1],1)),final_result[:,:,:depth_/2]),axis=2)
    final_result2 = np.concatenate((np.zeros((final_result.shape[0],final_result.shape[1],1)),final_result[:,:,depth_/2:]),axis=2)
    final_result = np.concatenate((
                                    np.power(10,final_result1[:,:,:,np.newaxis])/10,
                                    final_result2[:,:,:,np.newaxis]),axis=3)
    return final_result

def res_block(input_batch,layer_index,dilation,keep_prob, train_phase, reuse=None, scope = 'residual_block'):
    with tf.variable_scope(scope,reuse = reuse):
        num_outputs = dilation_channels
        kernel_size = filter_width
        conv_filter = conv(input_batch, num_outputs, kernel_size, 
                            rate = dilation, activation_fn = None, 
                            scope = 'dilation_conv{}'.format(layer_index))
        conv_filter = tf.contrib.layers.batch_norm(conv_filter, decay = 0.95, scale = True, is_training = train_phase, reuse = reuse, scope = 'dilation_conv_norm')
        conv_gate = conv(input_batch, num_outputs, kernel_size, 
                                rate = dilation, activation_fn = None, 
                                scope = 'dilation_gate{}'.format(layer_index))
        conv_gate = tf.contrib.layers.batch_norm(conv_gate, decay = 0.95, scale = True, is_training = train_phase, reuse = reuse, scope = 'dilation_gate_norm')        
        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)
        out = tf.nn.dropout(out,keep_prob)
        transformed = conv(out,
                            residual_channels, 
                            kernel_size = 1,
                            activation_fn = None,
                            scope="dense{}".format(layer_index))
        # transformed = tf.nn.dropout(transformed,keep_prob)
        skip_contribution = conv(out, 
                                skip_channels, 
                                kernel_size = 1,
                                activation_fn = None,
                                scope="skip{}".format(layer_index))
        # skip_contribution = tf.nn.dropout(skip_contribution,keep_prob)
    return skip_contribution, input_batch+transformed

# def network_wavenet(input_batch, keep_prob, train_phase, reuse=None):
#     outputs = []
#     initial_filter_width = initial_filter_width_
#     current_layer = input_batch
#     # current_layer = deconv(current_layer,
#     #                 skip_channels,
#     #                 kernel_size = 16,
#     #                 stride = 4, scope = 'upstride1')
#     # current_layer = deconv(current_layer,
#     #             skip_channels,
#     #             kernel_size = 16,
#     #             stride = 4,scope = 'upstride2')
#     # current_layer = deconv(current_layer,
#     #             skip_channels,
#     #             kernel_size = 8,
#     #             stride = 2,scope = 'upstride3')
#     with tf.variable_scope('initial_layer',reuse = reuse):
#         current_layer = conv(current_layer,
#                             residual_channels,
#                             initial_filter_width,
#                             activation_fn=None,
#                             )
#     with tf.variable_scope('dilated_stack',reuse = reuse):
#         for layer_index, dilation in enumerate(dilations):
#             with tf.variable_scope('layer{}'.format(layer_index)):
#                 output, current_layer = res_block(
#                         current_layer, layer_index, dilation, keep_prob,train_phase)                
#                 outputs.append(output)

#     with tf.variable_scope('postprocessing',reuse = reuse):
#         total = sum(outputs)
#         transformed1 = tf.nn.relu(total)
#         current_layer = transformed1
#         # current_layer = conv(current_layer,
#         #                 skip_channels,
#         #                 kernel_size = 8,
#         #                 activation_fn = None, 
#         #                 stride = 2, scope = 'downstride1')
#         # current_layer = tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'downstride1_norm')
#         # current_layer = tf.nn.relu(current_layer)
#         # current_layer = tf.nn.dropout(current_layer,keep_prob)

#         # current_layer = conv(current_layer,
#         #                 skip_channels,
#         #                 kernel_size = 8,
#         #                 activation_fn = None, 
#         #                 stride = 2, scope = 'downstride2')
#         # current_layer = tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'downstride2_norm')
#         # current_layer = tf.nn.relu(current_layer)
#         # current_layer = tf.nn.dropout(current_layer,keep_prob)

#         current_layer = conv(current_layer,
#                             skip_channels,
#                             kernel_size=1,
#                             activation_fn = None, 
#                             scope = 'postprocess1')
#         current_layer = tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, scope = 'postprocess1_norm')
#         current_layer = tf.nn.relu(current_layer)
#         # current_layer = tf.nn.dropout(current_layer,keep_prob)

#         conv2 = conv(current_layer,
#                     depth,
#                     kernel_size=1,
#                     activation_fn = None,
#                     scope = 'postprocess2')

#     return conv2#[:,:-16]

def network_wavenet(input_batch, keep_prob, train_phase, reuse=None, name='G'):
    with tf.variable_scope('generator_common_layers',reuse=reuse):
        current = conv(input_batch,num_outputs=depth,kernel_size=100, scope = 'out', activation_fn=None,stride=1)
    return current#current[:,8:-8]

# def network_wavenet(input_batch, keep_prob, train_phase, reuse=None, name='G'):
#     with tf.variable_scope('generator_common_layers',reuse=reuse):
#         current = tf.layers.flatten(input_batch)
#         current = tf.layers.dense(current,(width_)*depth_)
#         current = tf.reshape(current,[-1,width_,depth_])
#     return current#current[:,8:-8]
 
# def network_wavenet(input_batch, keep_prob, train_phase, reuse=None, name='G'):
#     current = input_batch
#     with tf.variable_scope('generator_common_layers',reuse=reuse):     
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res1', rate = 1,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res2', rate = 2,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res3', rate = 4,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res4', rate = 8,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res5', rate = 1, resample='down',mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res6', rate = 2,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res7', rate = 4,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res8', rate = 8,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res9', rate = 16,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res10', rate = 1, resample='down',mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res11', rate = 2,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res12', rate = 4,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res13', rate = 8,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res14', rate = 16,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res15', rate = 32,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res16', rate = 1,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res17', rate = 2,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res18', rate = 4,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res19', rate = 8,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res20', rate = 16,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res21', rate = 32,mode='generator')
#         current = tf.contrib.layers.batch_norm(current, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm1')
#         common = nonlinearity(current,mode='generator')
#     with tf.variable_scope('generator_domain_layers_1'+name,reuse=reuse):
#         out1 = conv(common, num_outputs=32,kernel_size=1, scope = 'out', activation_fn=None)
#     # with tf.variable_scope('generator_domain_layers_2'+name,reuse=reuse):
#     #     out2 = conv(common, num_outputs=32,kernel_size=1, scope = 'out', activation_fn=None)
#     return out1[:,8:-8]

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, width, depth]) # [None, 28*28]
# x = tf.placeholder(tf.float32, shape=[None, width+100-1, depth]) # [None, 28*28]
y_ = tf.placeholder(tf.float32, shape=[None, width_, depth_])  # [None, 10]
y_show = tf.placeholder(tf.float32, shape=[None, width_show_, depth_])  # [None, 10]
# y_show = tf.placeholder(tf.float32, shape=[None, width_show_-100+1, depth_])  # [None, 10]
x_show = tf.placeholder(tf.float32, shape=[None, width_show_+16, depth]) # [None, 28*28]
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

pred = network_wavenet(y_, keep_prob,train_phase=True, reuse = None)
# pred_show = network_wavenet(x_show, keep_prob,train_phase=True, reuse = True)
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

window = tf.reshape(tf.constant(np.array([-0.5,1]),dtype=tf.float32),[2,1,1,1])
pred_reshape = tf.expand_dims(pred,axis=-1)
y_reshape = tf.expand_dims(y_,axis=-1)
# norminator = tf.nn.conv2d(y_reshape**2,window,[1,1,1,1],'SAME')
# losses = tf.reduce_mean((tf.reduce_mean(tf.squeeze((tf.nn.conv2d(y_reshape-pred_reshape,window,[1,1,1,1],'SAME'))**2,axis=-1),axis=-1))**0.5)
# denorminator = tf.nn.conv2d((y_reshape-pred_reshape)**2,window,[1,1,1,1],'SAME')
# losses = tf.reduce_mean(-10*tf.log(norminator/denorminator)/tf.log(10.))
# losses = tf.reduce_mean((y_-pred)**2)/tf.reduce_mean(y_**2) + L2_REG_WEIGHT * l2_loss
# losses = tf.reduce_mean((tf.reduce_mean((y_-pred)**2,axis=-1))**0.5) + L2_REG_WEIGHT * l2_loss 
losses = tf.reduce_mean(tf.losses.mean_squared_error(labels=x, predictions=pred)) + L2_REG_WEIGHT * l2_loss
losses_test = losses
# losses_show = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_show, predictions=pred_show))
# losses_test = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=pred_test))
# losses_show = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_show_, predictions=pred_show))
# losses = tf.reduce_mean(tf.losses.absolute_difference(labels=y_[:,:,:], predictions=pred[:,:,:]))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(lr).minimize(losses)  # 1e-4

training_summary = tf.summary.scalar("training_loss", losses_test)
validation_summary = tf.summary.scalar("validation_loss", losses_test)
# validation_summary = tf.summary.scalar("validation_loss", losses_show)

# merged = tf.summary.merge_all()

saver = tf.train.Saver()
saver_load = tf.train.Saver([v for v in tf.trainable_variables()])

sess = tf.InteractiveSession()


## Run
sess.run(tf.global_variables_initializer())
saver_load.restore(sess, LOAD_DIR)
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph) 

# Include keep_prob in feed_dict to control dropout rate.
ecog_train, spkr_train, ecog_test, spkr_test, spkrspec_train, spkrspec_test,label_train,label_test,start_ind_train,start_ind_test,target_train,target_test,peseudo_train,peseudo_test,word_window_train,word_window_test  = read_all_data()

batch = np.zeros([32,128,32])
for i in range(32):
    batch[i,50:60,i] = 1.0

preded= sess.run(pred, 
                feed_dict={ y_: batch, 
                            keep_prob: 1.0})
result_to_save = np.asarray(preded)
scipy_io.savemat(LOG_DIR_RESULT+file_name+'_'+timestr+'_preded.mat', dict([('preded',result_to_save)]))

all_var = [v for v in tf.trainable_variables()]
strf = np.asarray(sess.run(all_var[0]))
scipy_io.savemat(LOG_DIR_RESULT+file_name+'_'+timestr+'_strf.mat', dict([('strf',strf)]))
