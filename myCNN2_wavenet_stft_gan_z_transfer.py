"""
Richard Wang
"""

import tensorflow as tf
import sys
from myLib_wavenet_stft_gan_z_transfer import *

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
noise_dims = 64
n_critic = 5
batch_size = 32
# frame_length = 64
# frame_step = 32
LOAD_DIR = './log/'+'myCNN2_wavenet_stft_gan_z_20180416-012834'+'/model_step_240999'
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
    gp = tf.reduce_mean((slopes - 1.)**2)
    return gp

def reshape_result(final_result):
    final_result1 = np.concatenate((np.zeros((final_result.shape[0],final_result.shape[1],1)),final_result[:,:,:depth_/2]),axis=2)
    final_result2 = np.concatenate((np.zeros((final_result.shape[0],final_result.shape[1],1)),final_result[:,:,depth_/2:]),axis=2)
    final_result = np.concatenate((
                                    np.power(10,final_result1[:,:,:,np.newaxis])/10,
                                    final_result2[:,:,:,np.newaxis]),axis=3)
    return final_result


def generator_wave(input_batch, keep_prob, train_phase, reuse=None, name=None):
    current = input_batch
    with tf.variable_scope('generator_transform_layers',reuse = reuse):
        current = conv(current, num_outputs = 64, kernel_size = 16, scope = 'conv_0')
        current = tf.nn.pool(current,[2],'AVG','SAME',strides=[2])
        current = res_block_dis(current, num_outputs=32,kernel_size=4,train_phase=train_phase, scope = 'res1', rate = 1, mode ='generator',resample ='down')
        current = res_block_dis(current, num_outputs=32,kernel_size=4,train_phase=train_phase, scope = 'res2', rate = 1, mode ='generator',resample ='down')
        current = res_block_dis(current, num_outputs=32,kernel_size=4,train_phase=train_phase, scope = 'res3', rate = 1, mode ='generator',resample ='down')
        current = res_block_dis(current, num_outputs=32,kernel_size=4,train_phase=train_phase, scope = 'res4', rate = 1, mode ='generator',resample ='down')
        current = res_block_dis(current, num_outputs=32,kernel_size=4,train_phase=train_phase, scope = 'res5', rate = 1, mode ='generator',resample ='down')
        current = tf.layers.flatten(current)
        current = tf.layers.dense(current, 64, name = 'dense')
        embedding = current
    with tf.variable_scope('generator_common_layers',reuse=reuse):
        current = tf.layers.dense(current, 16*32, name = 'dense1',activation=functools.partial(nonlinearity,mode = 'generator'))
        current = tf.reshape(current,[-1,16,32])
        current = conv(current,num_outputs = 32, kernel_size = 6, scope = 'conv1',activation_fn=None)       
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res1', rate = 1, resample='up', mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res2', rate = 2, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res3', rate = 4, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res4', rate = 8, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res5', rate = 1, resample='up', mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res6', rate = 2, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res7', rate = 4, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res8', rate = 8, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res9', rate = 16, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res10', rate = 1, resample='up', mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res11', rate = 2, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res12', rate = 4, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res13', rate = 8, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res14', rate = 16, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res15', rate = 32, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res16', rate = 1, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res17', rate = 2, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res18', rate = 4, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res19', rate = 8, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res20', rate = 16, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res21', rate = 32, mode ='generator')
        common = current
    with tf.variable_scope('generator_domain_layers_1'+name,reuse=reuse):
        current = common
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res22', rate = 1, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res23', rate = 2, mode ='generator')
        current = res_block_dis(current, num_outputs=32,kernel_size=2,train_phase=train_phase, scope = 'res24', rate = 4, mode ='generator')
        current = norm_fn(current, is_training = train_phase, reuse = reuse, scope = 'norm')
        current = nonlinearity(current, mode ='generator')
        out1 = conv(current, num_outputs=32,kernel_size=1, scope = 'out', activation_fn=None)
    return out1, embedding

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, width, depth]) # [None, 28*28]
y = tf.placeholder(tf.float32, shape=[None, width_, depth_])  # [None, 10]
# z = tf.placeholder(tf.float32, shape=[None, noise_dims])

deno = tf.placeholder(tf.float32)
sub = tf.placeholder(tf.float32)
idx = tf.placeholder(tf.float32, shape=[10])
keep_prob = tf.placeholder(tf.float32)

pred, embedding = generator_wave(x, keep_prob,train_phase=True, reuse = None, name='G1')
# disc_realfak = discriminator(tf.concat([y,pred],axis=0), train_phase = True)
# disc_real = disc_realfak[:batch_size]
# disc_fake = disc_realfak[batch_size:]

# wd = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
# gp = gradient_penalty(y, pred, discriminator)

# D_loss = -wd + gp*10.

l2_loss = tf.add_n([tf.nn.l2_loss(v)
                    for v in tf.trainable_variables()
                    if not('bias' in v.name)])

# G_loss_gan = - tf.reduce_mean(disc_fake)
# G_loss =  G_loss_gan
# D_loss = D_loss
samples_mean,samples_var = tf.nn.moments(embedding,axes=[0])
t1 = (1 + samples_mean**2) / (2 * (samples_var*noise_dims)**2)
t2 = tf.log(samples_var*noise_dims)
KL = tf.reduce_mean(t1 + t2 - 0.5)

losses_sup = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=pred))
losses_test = losses_sup
G_loss = losses_sup + KL
# losses_test = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=pred_test))
# losses_show = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_show_, predictions=pred_show))
# losses = tf.reduce_mean(tf.losses.absolute_difference(labels=y_[:,:,:], predictions=pred[:,:,:])
# theta_D = [v for v in tf.trainable_variables() if 'discriminator' in v.name]
theta_G = [v for v in tf.trainable_variables() if 'generator_transform_layers' in v.name]
# dopt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1 = 0., beta2 = 0.9)#.minimize(D_loss, var_list=theta_D)
gopt = tf.train.AdamOptimizer(learning_rate=0.001)#.minimize(G_loss, var_list=theta_G)

ggrads = gopt.compute_gradients(G_loss,var_list=theta_G)
train_step_G = gopt.apply_gradients(ggrads)
gnorm = tf.global_norm([i[0] for i in ggrads])
# dgrads = dopt.compute_gradients(D_loss,var_list=theta_D)
# train_step_D = dopt.apply_gradients(dgrads)
# dnorm = tf.global_norm([i[0] for i in dgrads])

summ = []
summ.append(tf.summary.scalar("sup_loss", losses_sup))
summ.append(tf.summary.scalar("KL", KL)) 
# summ.append(tf.summary.scalar("gp", gp))
# summ.append(tf.summary.scalar("wd", wd))
# summ.append(tf.summary.scalar("G_loss", G_loss_gan))
# summ.append(tf.summary.scalar("D_loss", D_loss))
validation_summary = tf.summary.scalar("validation_loss", losses_test)
training_summary_merge = tf.summary.merge(summ)
# merged = tf.summary.merge_all()

saver = tf.train.Saver()
saver_load = tf.train.Saver([v for v in tf.trainable_variables() if ('generator_common_layers' in v.name or 'generator_domain_layers_1' in v.name)])

sess = tf.Session()


## Run
sess.run(tf.global_variables_initializer())
saver_load.restore(sess, LOAD_DIR)
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph) 

# Include keep_prob in feed_dict to control dropout rate.
ecog_train, spkr_train, ecog_test, spkr_test, spkrspec_train,spkrspec_test = read_all_data()
for i in range(10000): #820
    noise = np.random.standard_normal([batch_size,noise_dims])
    noise = noise / np.linalg.norm(noise,axis=1)[:,np.newaxis]
    batch = get_batch(ecog_train, spkrspec_train, seg_length=512, batch_size=batch_size, threshold = 0.0) ###########################################################################
    # Logging every ?th iteration in the training process.
    if i%10 == 0:
        # train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
        sup_loss,KL_,l2_loss_,summary_train, gnorm_, embedding_mean_train, embedding_var_train= sess.run([losses_sup, KL,l2_loss,training_summary_merge, gnorm, samples_mean, samples_var], 
                                                                                                    feed_dict={ x: batch[0],
                                                                                                                y: batch[1],
                                                                                                                keep_prob: 1.0})
        testset = get_batch(ecog_test, spkrspec_test, seg_length=512, batch_size=batch_size, threshold = 1.2)
        test_loss,summary_validate,embedding_mean_valid, embedding_var_valid = sess.run([losses_test,validation_summary,samples_mean, samples_var], feed_dict={  x: testset[0],
                                                                                            y: testset[1],
                                                                                            keep_prob: 1.0})
        summary_writer.add_summary(summary_train, i)
        summary_writer.add_summary(summary_validate, i)
        print("Step %d, Training Losses-- sup: %g, KL: %g, l2_loss: %g;    Test Losses-- %g" %(i, sup_loss, KL_,l2_loss_, test_loss ))
        print 'gnorm ', gnorm_, 'embedding_mean_train:', embedding_mean_train.mean(), 'embedding_var_train', embedding_var_train.mean(),'embedding_mean_valid:', embedding_mean_valid.mean(), 'embedding_var_valid', embedding_var_valid.mean()
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
    # for j in range(n_critic):
    #     _ = sess.run([train_step_D],feed_dict={ x: batch[0], 
    #                                     y: batch[1], 
    #                                     keep_prob: 0.5})
    _ =  sess.run([train_step_G],feed_dict={ x: batch[0], 
                                y: batch[1],
                                keep_prob: 0.5})

    if i%500== 499:
        LOG_DIR_RESULT_step = LOG_DIR_RESULT + '/step_' + str(i) +'/'
        if not os.path.exists(LOG_DIR_RESULT_step):
            os.makedirs(LOG_DIR_RESULT_step)
        for j in range(10):
            testset = get_batch(ecog_test, spkrspec_test, seg_length=512, batch_size=batch_size, threshold = 1.2) ###########################################################################
            batch = get_batch(ecog_train, spkrspec_train, seg_length=512, batch_size=batch_size, threshold = 1.2)
            # test_accuracy = accuracy.eval(feed_dict={x: testset[0], y_1: testset[1], deno: batch[2], sub: batch[3], keep_prob: 1.0})
            # test_losses, final_result_1, final_result_2 = sess.run([losses_sup, pred_1, pred_2], 
            #                                     feed_dict={ #x: testset[0], 
            #                                                 y_1: testset[1],
            #                                                 z: noise,
            #                                                 keep_prob: 1.0})
            test_losses, final_result= sess.run([losses_test, pred], 
                                                feed_dict={ x: testset[0], 
                                                            y: testset[1],
                                                            keep_prob: 1.0})
            final_result_train = sess.run(pred, 
                                        feed_dict={ x: batch[0], 
                                                    y: batch[1],
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
            scipy_io.savemat(LOG_DIR_RESULT_step+'pred_STFT_test_'+str(j)+'.mat', dict([('pred_STFT_test',result_to_save)]))
            # result_to_save_2 = np.asarray(final_result_2)
            # scipy_io.savemat(LOG_DIR_RESULT_step+'pred_STFT_test_2_'+str(j)+'.mat', dict([('pred_STFT_test_2',result_to_save_2)]))

            scipy_io.savemat(LOG_DIR_RESULT_step+'GT_STFT_train'+str(j)+'.mat', dict([('GT_STFT_train',gt_save_train)]))
            # # scipy_io.savemat(LOG_DIR_RESULT_step+'input_STFT_train'+str(j)+'.mat', dict([('input_STFT_train',batch[0])]))
            # # scipy_io.savemat(LOG_DIR_RESULT_step+'out_train'+str(j)+'.mat',dict([('out0_train',out_train[0]),('out1_train',out_train[1]),('out2_train',out_train[2]),('out3_train',out_train[3])]))
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