"""
Richard Wang
"""

import tensorflow as tf
import sys
from myLib_wavenet_stft_cv_graph2 import *
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
width = (512+64)/4 # ecog segment width 26701
width_show = (512+64)/4
# depth = 24
# height_ = 128 # 0-6kHz
width_ = 512/4 # spech segment width 26701
width_show_ = 9289-32#512/4
depth_ = 32#128#128#256#15#32 
num_words = 32
# dilations = [1, 2, 4, 8, 16, 32,
#              1, 2, 4, 8, 16, 32,]
# dilations = [1, 2, 4, 8, 16,
#              1, 2, 4, 8, 16,]
             # 1, 2, 4, 8, 16, 32, 64,
             # 1, 2, 4, 8, 16, 32, 64,
             # 1, 2, 4, 8, 16, 32, 64,
             # 1, 2, 4, 8, 16, 32, 64,]
dilations = [1, 2, 4, 8, 16,
             1, 2, 4, 8, 16,]
# frame_length = 64
# frame_step = 32
LOG_DIR = './log/'+file_name+'_'+timestr 
LOG_DIR_RESULT = LOG_DIR+'/result/'
dilation_channels = 32
residual_channels = 16
skip_channels = 64#512
filter_width = 2
L2_REG_WEIGHT = 0#0.001
L1_REG_WEIGHT = 0.000
Siamese_Margin = 0.6
SIM_WEIGHT=1.#1.0
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

def network_wavenet(input_batch, keep_prob, train_phase,reuse=None):
    outputs = []
    initial_filter_width = initial_filter_width_
    current_layer = tf.concat(input_batch,axis=0)
    transformed = current_layer
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
    ##############code block for mergenet############
    # transformed_results = []
    # for d in xrange(num_dataset):
    #     input_batch_d = input_batch[d]
    #     with tf.variable_scope('transformer_layer_'+str(d),reuse = reuse):
    #         current_layer = conv(input_batch_d,
    #                             48,
    #                             1,
    #                             activation_fn=tf.nn.relu,
    #                             )
    #         transformed_results+=[current_layer]
    # transformed = tf.concat(transformed_results,axis=0)
    # current_layer = transformed
    ################################################

    # transformed = tf.concat(current_layer,axis=0)
    # current_layer = conv(transformed,64,1,activation_fn=tf.nn.relu)
    
    
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
                        current_layer, layer_index, dilation, keep_prob,train_phase)                
                outputs.append(output)

    with tf.variable_scope('postprocessing',reuse = reuse):
        total = sum(outputs)
        transformed1 = tf.nn.relu(total)
        current_layer = transformed1
        # current_layer = conv(current_layer,
        #                 skip_channels,
        #                 kernel_size = 8,
        #                 activation_fn = None, 
        #                 stride = 2, scope = 'downstride1')
        # current_layer = tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'downstride1_norm')
        # current_layer = tf.nn.relu(current_layer)
        # current_layer = tf.nn.dropout(current_layer,keep_prob)

        # current_layer = conv(current_layer,
        #                 skip_channels,
        #                 kernel_size = 8,
        #                 activation_fn = None, 
        #                 stride = 2, scope = 'downstride2')
        # current_layer = tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'downstride2_norm')
        # current_layer = tf.nn.relu(current_layer)
        # current_layer = tf.nn.dropout(current_layer,keep_prob)

        current_layer = conv(current_layer,
                            skip_channels,
                            kernel_size=1,
                            activation_fn = None, 
                            scope = 'postprocess1')
        current_layer = tf.contrib.layers.batch_norm(current_layer, decay = 0.95, is_training = train_phase, scope = 'postprocess1_norm')
        current_layer = tf.nn.relu(current_layer)
        # current_layer = tf.nn.dropout(current_layer,keep_prob)

        conv2 = conv(current_layer,
                    depth_,
                    kernel_size=1,
                    activation_fn = None,
                    scope = 'postprocess2')

    return conv2[:,:-16],transformed

# def network_wavenet(input_batch, keep_prob, train_phase, reuse=None, name='G'):
#     with tf.variable_scope('generator_common_layers',reuse=reuse):
#         current = conv(input_batch,num_outputs=32,kernel_size=100, scope = 'out', activation_fn=None,stride=1,padding='SAME')
#     return current[:,:-16]#current[:,8:-8]

# def network_wavenet(input_batch, keep_prob, train_phase, reuse=None, name='G'):
#     with tf.variable_scope('generator_common_layers',reuse=reuse):
#         current = tf.layers.flatten(input_batch)
#         current = tf.layers.dense(current,(width_+16)*depth_)
#         current = tf.reshape(current,[-1,width_+16,depth_])
#     return current[:,:-16]#current[:,8:-8]
 
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
 
# norm_fn = functools.partial(tf.contrib.layers.batch_norm,decay=0.999,fused = True, updates_collections=None)

# def network_wavenet(input_batch, keep_prob, train_phase, reuse=None, name='G'):
#     current = input_batch
#     with tf.variable_scope('generator_common_layers',reuse=reuse):
#         current = conv(current,
#                     32,
#                     kernel_size=32,
#                     activation_fn = None,
#                     scope = 'postprocess1')
#         # current = norm_fn(current, is_training = train_phase, reuse = reuse, scope = 'norm0')
#         # current = nonlinearity(current,'generator')
#         # current = conv(current,
#         #             32,
#         #             kernel_size=16,
#         #             activation_fn = None, 
#         #             scope = 'postprocess2')
#         current = res_block_dis(current, num_outputs=32,kernel_size=8,train_phase=train_phase, scope = 'res1', rate = 1,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=8,train_phase=train_phase, scope = 'res2', rate = 1,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=8,train_phase=train_phase, scope = 'res3', rate = 1,mode='generator')
#         current = res_block_dis(current, num_outputs=32,kernel_size=8,train_phase=train_phase, scope = 'res4', rate = 1,mode='generator')
#         # current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res5', rate = 1,mode='generator')
#         # current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res6', rate = 1,mode='generator')
#         # current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res7', rate = 1,mode='generator')
#         # current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res8', rate = 1,mode='generator')
#         # current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res9', rate = 1,mode='generator')
#         # current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res10', rate = 1,mode='generator')
#         # current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res11', rate = 1,mode='generator')
#         # current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res12', rate = 1,mode='generator')
#         # current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res13', rate = 1,mode='generator')
#         # current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res14', rate = 1,mode='generator')
#         # current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res15', rate = 1,mode='generator')
#         # current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res16', rate = 1,mode='generator')
#         # current = tf.contrib.layers.batch_norm(current, decay = 0.95, is_training = train_phase, reuse = reuse, scope = 'norm1')
#         # common = nonlinearity(current,mode='generator')
#     # with tf.variable_scope('generator_domain_layers_1'+name,reuse=reuse):
#     #     out1 = conv(common, num_outputs=32,kernel_size=1, scope = 'out', activation_fn=None)
#     # with tf.variable_scope('generator_domain_layers_2'+name,reuse=reuse):
#     #     out2 = conv(common, num_outputs=32,kernel_size=1, scope = 'out', activation_fn=None)
#     return current[:,:-16]


ecog_alldataset, spkr_alldataset, start_ind_alldataset, word_alldataset = read_all_data()
# Placeholders
num_dataset = len(ecog_alldataset)
x = [tf.placeholder(tf.float32, shape=[None, width, ecog_alldataset[i].shape[-1]]) for i in xrange(num_dataset)] # [None, 28*28]
# x = tf.placeholder(tf.float32, shape=[None, width+100-1, depth]) # [None, 28*28]
y_ = tf.placeholder(tf.float32, shape=[None, width_, depth_])  # [None, 10]
word_window = tf.placeholder(tf.float32, shape=[None, width_, 1])  # [None, 10]
y_show = tf.placeholder(tf.float32, shape=[None, width_show_, depth_])  # [None, 10]
# y_show = tf.placeholder(tf.float32, shape=[None, width_show_-100+1, depth_])  # [None, 10]
# x_show = tf.placeholder(tf.float32, shape=[None, width_show_+16, depth]) # [None, 28*28]
# y_show_ = tf.placeholder(tf.float32, shape=[None, width_show_, depth_])  # [None, 10]
# y_spectrom = tf.contrib.signal.stft(tf.squeeze(y_),frame_length=64,frame_step=32)
input_dataset = tf.placeholder(tf.int32,shape = [None])
input_word = tf.placeholder(tf.int32,shape = [None])


deno = tf.placeholder(tf.float32)
sub = tf.placeholder(tf.float32)
idx = tf.placeholder(tf.float32, shape=[10])
keep_prob = tf.placeholder(tf.float32)

# Reshape 'x' and 'y' to a 4D tensor (2nd dim=width, 3rd dim=height, 4th dim=Channel)
# x_spec = tf.reshape(x, [-1,height,width,depth])
# print(x_spec.get_shape)
# y_spec_ = tf.reshape(y_, [-1,height_,width_,depth_])
# print(y_spec_.get_shape)

pred,transform = network_wavenet(x, keep_prob,train_phase=True, reuse = None)
# pred_show = network_wavenet(x_show, keep_prob,train_phase=True, reuse = True)
# pred_test = network_wavenet(x, keep_prob,train_phase=False, reuse = True)
# # pred_test = network_wavenet(x, keep_prob,train_phase=True, reuse = True)
# pred_show = network_wavenet(x_show, keep_prob,train_phase=False, reuse = True)
# # pred_show = network_wavenet(x_show, keep_prob,train_phase=True, reuse = True)
pred_wav = tf.argmax(
                    tf.cast(
                        tf.nn.softmax(tf.cast(pred, tf.float64)), tf.float32),
                    axis =-1)

words,word_id = tf.unique(input_word)
var_sim = tf.constant(0,dtype=tf.float32)
for w in range(num_words):
    # indx = tf.where(tf.equal(word_id,w))[:,0]
    # var_sim += tf.reduce_mean(tf.nn.moments(transform[indx[0]],axes=0)[1])*tf.to_float(tf.shape(indx))
    same_word_indx = [i*num_words+w for i in range(num_dataset)]
    var_sim += tf.reduce_mean(tf.nn.moments(transform[same_word_indx],axes=0)[1])*np.float32(len(same_word_indx))
var_sim /= np.float32(num_words*num_dataset)

var_distinct = tf.constant(0,dtype=tf.float32)
for w in range(num_words):
    diff_word_indx = [(i*num_words+w+i)%num_words for i in range(num_dataset)]
    var_distinct += tf.reduce_mean(tf.nn.moments(transform[diff_word_indx],axes=0)[1])*np.float32(len(diff_word_indx))
var_distinct /= np.float32(num_words*num_dataset)

loss_trans = 0#tf.maximum(var_sim-var_distinct + Siamese_Margin,0) + var_sim**2
# loss_trans = (1+40*var_sim**2)/(2*(var_distinct*3)**2) + tf.log(var_distinct*3)


l2_loss = tf.add_n([tf.nn.l2_loss(v)
                    for v in tf.trainable_variables()
                    if not('bias' in v.name)])
l1_regularizer = tf.contrib.layers.l1_regularizer(
   scale=L1_REG_WEIGHT, scope=None
)
l1_loss = tf.contrib.layers.apply_regularization(l1_regularizer, tf.trainable_variables())
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
losses = tf.reduce_mean((tf.reduce_mean((y_-pred)**2,axis=-1))**0.5) + L2_REG_WEIGHT * l2_loss + SIM_WEIGHT*loss_trans
# losses = tf.reduce_mean((y_ - pred)**2*(word_window*9+1.0)) + L2_REG_WEIGHT * l2_loss + l1_loss
# losses = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=pred)) + L2_REG_WEIGHT * l2_loss + l1_loss
losses_test = tf.reduce_mean((y_ - pred)**2)


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

sess = tf.InteractiveSession()


## Run


# Include keep_prob in feed_dict to control dropout rate.
for cv in range(1):
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    testset = get_batch(ecog_alldataset, spkr_alldataset, start_ind_alldataset, word_alldataset, mode = 'test',num_words=num_words,seg_length=width-64/4, threshold = 0.0)
    for i in range(3000): #820
        batch = get_batch(ecog_alldataset, spkr_alldataset, start_ind_alldataset, word_alldataset, mode = 'train',num_words=num_words,seg_length=width-64/4,threshold = 0.0)
        # batch = get_batch(ecog_train, spkrspec_train, filter_width=100,seg_length=width-64/4, batch_size=32, threshold = 0.0)
        # Logging every ?th iteration in the training process.
        if i%10 == 0:
            # train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
            feed = {}
            feed.update({i:d for i,d in zip(x,batch[0])})
            feed.update({y_: batch[1],input_word:batch[2],keep_prob: 1.0})
            train_loss,var_sim_train, var_distinct_train,l2_loss_,l1_loss_,summary_train = sess.run([losses, var_sim, var_distinct,l2_loss,l1_loss,training_summary], 
                                                                    feed_dict= feed)
            # train_loss,l2_loss_,l1_loss_,summary_train = sess.run([losses ,l2_loss,l1_loss,training_summary], 
            #                                                         feed_dict={ i:d for i,d in zip(x,batch[0]),
            #                                                                     y_: batch[1],
            #                                                                     input_word:batch[2],
            #                                                                     # word_window: batch[2],
            #                                                                     keep_prob: 1.0})

            # testset = get_batch_show(ecog_test, spkrspec_test,start_ind_test,seg_length=512/4)
            # testset_x = np.expand_dims(ecog_test[16:],0)
            # testset_y = np.expand_dims(spkrspec_test[0:-32],0)
            # testset_y = np.expand_dims(spkrspec_test[0+50:-32-50+1],0)
            # test_loss,summary_validate = sess.run([losses_show,validation_summary], feed_dict={ x_show:testset_x,
            #                                                 y_show: testset_y,
            #                                                 keep_prob: 1.0})
            feed = {}
            feed.update({i:d for i,d in zip(x,testset[0])})
            feed.update({y_: testset[1],input_word:testset[2],keep_prob: 1.0})
            test_loss,var_sim_test,var_distinct_test,summary_validate = sess.run([losses_test,var_sim,var_distinct,validation_summary],
                                                        feed_dict=feed)
            summary_writer.add_summary(summary_train, i) 
            summary_writer.add_summary(summary_validate, i)
            print("Step %d, Training Losses %g, Test Losses %g, var_sim_train %g, var_sim_test %g,var_distinct_train %g, var_distinct_test %g, l2_loss %g, l1_loss %g" %(i, train_loss, test_loss, var_sim_train,var_sim_test,var_distinct_train,var_distinct_test,l2_loss_,l1_loss_))
            
            feed = {}
            feed.update({i:d for i,d in zip(x,batch[0])})
            feed.update({y_: batch[1],input_word:batch[2],keep_prob: 0.5})        
        sess.run(train_step,feed_dict=feed)

        if i%50== 49:
            LOG_DIR_RESULT_step = LOG_DIR_RESULT + '/step_' + str(i) +'/'
            if not os.path.exists(LOG_DIR_RESULT_step):
                os.makedirs(LOG_DIR_RESULT_step)
            # for j in range(10):

            # testset = get_batch_show(ecog_test, spkrspec_test,start_ind_test,seg_length=512/4)
            # batch = get_batch_show(ecog_train, spkrspec_train,start_ind_train,seg_length=512)
            # testset = get_batch(ecog_test, spkrspec_test, seg_length=width-64, batch_size=32, threshold = 1.2) ###########################################################################
            batch = get_batch(ecog_alldataset, spkr_alldataset, start_ind_alldataset, word_alldataset , mode = 'train',num_words=num_words,seg_length=width-64/4, threshold = 0.0)
            # batch = get_batch(ecog_train, spkrspec_train,filter_width=100,seg_length=width-64/4, batch_size=32,threshold = 0.0)
            # test_accuracy = accuracy.eval(feed_dict={x: testset[0], y_1: testset[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
            # test_losses, final_result_1, final_result_2 = sess.run([losses_sup, pred_1, pred_2], 
            #                                     feed_dict={ #x: testset[0], 
            #                                                 y_1: testset[1],
            #                                                 z: noise,
            # #                                                 keep_prob: 1.0})
            # test_losses, final_result= sess.run([losses_show, pred_show], 
            #                                     feed_dict={ x_show: testset_x, 
            #                                                 y_show: testset_y,
            #                                                 keep_prob: 1.0})
            feed = {}
            feed.update({i:d for i,d in zip(x,testset[0])})
            feed.update({y_: testset[1],input_word:testset[2],keep_prob: 1.0})  
            test_losses, final_result= sess.run([losses_test, pred], 
                                                feed_dict=feed)
            feed = {}
            feed.update({i:d for i,d in zip(x,batch[0])})
            feed.update({y_: batch[1],input_word:batch[2],keep_prob: 1.0})   
            final_result_train = sess.run(pred, 
                                        feed_dict=feed)

            # gt_save_test = testset_y
            gt_save_test = testset[1]
            gt_save_train = batch[1]
            # scipy_io.savemat(LOG_DIR_RESULT+'pred_index'+str(j)+'.mat', dict([('pred_index',result_index)]))
            scipy_io.savemat(LOG_DIR_RESULT_step+file_name+'_'+timestr+'_step_'+str(i)+'_GT_STFT_test_cv_'+str(cv)+'.mat', dict([('GT_STFT_test',gt_save_test)]))
            # scipy_io.savemat(LOG_DIR_RESULT_step+'input_STFT_test'+str(j)+'.mat', dict([('input_STFT_test',testset[0])]))
            # scipy_io.savemat(LOG_DIR_RESULT_step+'out_test'+str(j)+'.mat',dict([('out0_test',out_[0]),('out1_test',out_[1]),('out2_test',out_[2]),('out3_test',out_[3])]))
            result_to_save = np.asarray(final_result)
            scipy_io.savemat(LOG_DIR_RESULT_step+file_name+'_'+timestr+'_step_'+str(i)+'_pred_STFT_test_cv_'+str(cv)+'_loss_'+str(test_losses)+'.mat', dict([('pred_STFT_test',result_to_save)]))
            # result_to_save_2 = np.asarray(final_result_2)
            # scipy_io.savemat(LOG_DIR_RESULT_step+'pred_STFT_test_2_'+str(j)+'.mat', dict([('pred_STFT_test_2',result_to_save_2)]))

            scipy_io.savemat(LOG_DIR_RESULT_step+file_name+'_'+timestr+'_step_'+str(i)+'_GT_STFT_train_cv_'+str(cv)+'.mat', dict([('GT_STFT_train',gt_save_train)]))
            # # scipy_io.savemat(LOG_DIR_RESULT_step+'input_STFT_train'+str(j)+'.mat', dict([('input_STFT_train',batch[0])]))
            # # scipy_io.savemat(LOG_DIR_RESULT_step+'out_train'+str(j)+'.mat',dict([('out0_train',out_train[0]),('out1_train',out_train[1]),('out2_train',out_train[2]),('out3_train',out_train[3])]))
            result_to_save = np.asarray(final_result_train)
            scipy_io.savemat(LOG_DIR_RESULT_step+file_name+'_'+timestr+'_step_'+str(i)+'_pred_STFT_train_cv_'+str(cv)+'.mat', dict([('pred_STFT_train',result_to_save)]))

            scipy_io.savemat(LOG_DIR_RESULT_step+file_name+'_'+timestr+'_step_'+str(i)+'mse_cv_'+str(cv)+'.mat', dict([('pred_STFT_train',test_losses)]))
            print("Avg Test Losses %g" %(test_losses))

    saver.save(sess, LOG_DIR+'/model'+'_step_3000_cv'+str(cv))

 
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