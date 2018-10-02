import tensorflow as tf
import functools
import numpy as np


dilation_channels = 32
residual_channels = 16
skip_channels = 512
filter_width = 2


conv = tf.contrib.slim.convolution

# def norm_fn(x,is_training=None, reuse = None, scope = None):
#     return batch_norm(x, train=is_training,reuse=reuse,name=scope)

def group_norm(x, G=32, eps=1e-5, reuse = None, scope='group_norm') :
    with tf.variable_scope(scope , reuse = reuse) :
        N, W, C = x.get_shape().as_list()
        G = min(G, C)

        x = tf.reshape(x, [-1, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 3], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma', [1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, C], initializer=tf.constant_initializer(0.0))


        x = tf.reshape(x, [-1, W, C]) * gamma + beta

    return x


norm_fn = functools.partial(tf.contrib.layers.batch_norm,decay=0.999,fused = True, updates_collections=None)#,updates_collections = None)
def norm_fn_critic(x, is_training = None, reuse = None, scope = None): 
    return group_norm(x,reuse=reuse,scope = scope)
    #return tf.contrib.layers.layer_norm(x, reuse = reuse, scope = scope)
# def norm_fn(input_batch,is_training=None, reuse = None, scope = None):
#     return input_batch

def nonlinearity(x,mode=None):
    if mode == 'discriminator':
        return tf.nn.softplus(2.*x+2.)/2.-1
    elif mode == 'generator':
        return tf.nn.elu(x)
    elif mode == 'classifier':
        return tf.nn.relu(x)

# def nonlinearity(x):
#     return tf.nn.leaky_relu(x)

def deconv(input, num_outputs, kernel_size, stride, activation_fn = nonlinearity, scope='deconv'):
    with tf.variable_scope(scope):
        input = tf.expand_dims(input, axis = 1)
        output = tf.contrib.slim.conv2d_transpose(input,num_outputs,[1,kernel_size],[1,stride],scope=scope)
        output = tf.squeeze(output,axis = 1)
    return output

def res_block(input_batch,layer_index,dilation,keep_prob, train_phase, reuse=None, scope = 'residual_block',local_condition=False, local_embedding=None):
    with tf.variable_scope(scope,reuse = reuse):
        num_outputs = dilation_channels
        kernel_size = filter_width
        conv_filter = conv(input_batch, num_outputs, kernel_size, 
                            rate = dilation, activation_fn = None, 
                            scope = 'dilation_conv{}'.format(layer_index))
        conv_filter = norm_fn(conv_filter, is_training = train_phase, reuse = reuse, scope = 'dilation_conv_norm')
        conv_gate = conv(input_batch, num_outputs, kernel_size, 
                                rate = dilation, activation_fn = None, 
                                scope = 'dilation_gate{}'.format(layer_index))
        conv_gate = norm_fn(conv_gate, is_training = train_phase, reuse = reuse, scope = 'dilation_gate_norm')
        if local_condition:
            conv_filter_lc = conv(local_embedding, num_outputs, 1, 
                            rate = 1, activation_fn = None, 
                            scope = 'dilation_conv_lc{}'.format(layer_index))
            conv_filter_lc = norm_fn(conv_filter_lc, is_training = train_phase, reuse = reuse, scope = 'dilation_conv_lc_norm')
            conv_gate_lc = conv(local_embedding, num_outputs, 1, 
                                rate = 1, activation_fn = None, 
                                scope = 'dilation_gate_lc{}'.format(layer_index))
            conv_gate_lc = norm_fn(conv_gate_lc, is_training = train_phase, reuse = reuse, scope = 'dilation_gate_lc_norm')

            out = tf.tanh(conv_filter + conv_filter_lc) * tf.sigmoid(conv_gate + conv_gate_lc)
        else:
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


def ConvMeanPool(inputs, num_outputs ,kernel_size, reuse = None, scope = 'ConvMeanPool',down=2,pool='AVG'):
    with tf.variable_scope(scope, reuse = reuse):
        output = conv(inputs, num_outputs=num_outputs, kernel_size = kernel_size, activation_fn=None)
        output = tf.nn.pool(output,[down],pool,'SAME',strides=[down])
    return output

def MeanPoolConv(inputs, num_outputs ,kernel_size, reuse = None, scope = 'MeanPoolConv',pool='AVG'):
    with tf.variable_scope(scope, reuse = reuse):
        output = inputs
        output = tf.nn.pool(output,[2],pool,'SAME',strides=[2])
        output = conv(output, num_outputs=num_outputs, kernel_size = kernel_size, activation_fn=None)
    return output


def res_block_dis(inputs, num_outputs ,kernel_size, train_phase, scope, resample=None, reuse = None, rate = 1, mode = None, down = 2, pool='AVG',dropout = None,keep_prob = 1.0):
    if mode == 'generator':
        normalize = norm_fn
    if mode == 'discriminator':
        normalize = norm_fn_critic
    if mode == 'classifier':
        normalize = norm_fn
    with tf.variable_scope(scope, reuse = reuse):
        if resample == 'down':
            conv_1 = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_1', rate = rate)
            conv_2 = functools.partial(ConvMeanPool, num_outputs=num_outputs,kernel_size = kernel_size, scope = 'conv_2',down=down,pool=pool)
            conv_shortcut = functools.partial(ConvMeanPool,kernel_size = 2,down=down,pool=pool)
        elif resample=='up':
            conv_1 = functools.partial(deconv, num_outputs=num_outputs,kernel_size=kernel_size, stride=2, activation_fn = None, scope = 'conv_1')
            conv_2 = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_2', rate = rate)
            conv_shortcut = functools.partial(deconv,stride=2, activation_fn = None,kernel_size=4)
        elif resample == None:
            conv_1 = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_1', rate = rate)
            conv_2 = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_2')
            conv_shortcut = functools.partial(conv,activation_fn = None, kernel_size=1)
        if inputs.get_shape().as_list()[-1] == num_outputs and resample == None:
            shortcut = inputs
        else:
            shortcut = conv_shortcut(inputs, num_outputs=num_outputs,  scope='shortcut')

        output = inputs
        output = normalize(output, is_training = train_phase, reuse = reuse, scope = 'norm1')
        output = nonlinearity(output,mode)
        if dropout is not None:
            output = tf.nn.dropout(output,keep_prob)
        output = conv_1(output, scope='conv1')
        output = normalize(output, is_training = train_phase, reuse = reuse, scope = 'norm2')
        output = nonlinearity(output,mode)
        if dropout is not None:
            output = tf.nn.dropout(output,keep_prob)
        output = conv_2(output, scope='conv2')

        return shortcut + output

def res_block_dis_gate(inputs, num_outputs ,kernel_size, train_phase, scope, resample=None, reuse = None, rate = 1, mode = None):
    if mode == 'generator':
        normalize = norm_fn
    if mode == 'discriminator':
        normalize = norm_fn_critic
    with tf.variable_scope(scope, reuse = reuse):
        if resample == 'down':
            conv_1_filter = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_1_filter', rate = rate)
            conv_1_gate = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_1_gate', rate = rate)
            conv_2 = functools.partial(ConvMeanPool, num_outputs=num_outputs,kernel_size = 1, scope = 'conv_2')
            conv_shortcut = functools.partial(ConvMeanPool, kernel_size = 2)
        elif resample=='up':
            conv_1_filter = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_1_filter', rate = rate)
            conv_1_gate = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_1_gate', rate = rate)
            conv_2 = functools.partial(deconv, num_outputs=num_outputs,kernel_size=4, stride=2, activation_fn = None, scope = 'conv_2')
            conv_shortcut = functools.partial(deconv,stride=2, activation_fn = None,kernel_size=4)
        elif resample == None:
            conv_1_filter = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_1_filter', rate = rate)
            conv_1_gate = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_1_gate', rate = rate)
            conv_2 = functools.partial(conv,num_outputs=num_outputs,kernel_size = kernel_size, activation_fn = None, scope = 'conv_2')
            conv_shortcut = functools.partial(conv,activation_fn = None, kernel_size=1)
        if inputs.get_shape().as_list()[-1] == num_outputs and resample == None:
            shortcut = inputs
        else:
            shortcut = conv_shortcut(inputs, num_outputs=num_outputs, scope='shortcut')

        output = inputs
        output = normalize(output, is_training = train_phase, reuse = reuse, scope = 'norm1')
        output = nonlinearity(output,mode)
        conv_filtered = normalize(conv_1_filter(output, scope='conv1_filter'),is_training = train_phase, reuse = reuse, scope = 'norm1_filter')
        conv_gated = normalize(conv_1_gate(output, scope='conv1_gate'),is_training = train_phase, reuse = reuse, scope = 'norm1_gate')
        output = tf.tanh(conv_filtered) * tf.sigmoid(conv_gated)
        # output = normalize(output, is_training = train_phase, reuse = reuse, scope = 'norm2')
        # output = nonlinearity(output,mode)
        output = conv_2(output, scope='conv2')

        return shortcut + output


def OptimizedResBlockDisc1(inputs, num_outputs, kernel_size, scope, reuse = None, mode=None,pool='AVG'):
    with tf.variable_scope(scope, reuse = reuse):
        shortcut = MeanPoolConv(inputs, num_outputs=num_outputs, kernel_size=1, scope='shortcut',pool=pool)
        output = inputs
        output = conv(output, num_outputs= num_outputs, kernel_size = kernel_size, scope = 'conv1')
        output = nonlinearity(output,mode)
        output = ConvMeanPool(output,num_outputs=num_outputs, kernel_size=kernel_size, scope='conv2',pool=pool)
        return shortcut + output


def cor_coef(gt,x):
    sample_num = gt.shape[1]*gt.shape[0]
    gt = gt.reshape([-1,gt.shape[-1]])
    gt = (gt-gt.mean(axis=0,keepdims=True))/np.sqrt(gt.var(axis=0,keepdims=True))
    x = x.reshape([-1,x.shape[-1]])
    x = (x-x.mean(axis=0,keepdims=True))/np.sqrt(x.var(axis=0,keepdims=True))

    correlate_coef = np.zeros([gt.shape[-1]])
    for i in range(gt.shape[-1]):
        cor = np.correlate(gt[:,i],x[:,i])
        correlate_coef[i] = cor/sample_num
        return correlate_coef.mean()
