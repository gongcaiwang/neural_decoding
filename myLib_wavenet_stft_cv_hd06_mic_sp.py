import scipy
import scipy.io as scipy_io
from scipy import signal
import h5py
import numpy as np
import random 
import librosa
import math
DATA_DIR = '../../neural_decoding_data/adeen/'
# DATA_DIR = './data/'

depth_ = 32#128#15 
quantization_channels = 2**8

channel_correlation = 1e5*np.array([0.4167,0.4178,0.4430,0.4459,0.4655,0.5357,0.5928,0.5969,0.5993,0.6141,0.6149,0.6150,0.6181,0.6204,0.6345,0.6740,0.7073,0.7271,0.7295,0.7345,0.7522,0.7825,0.9068,0.9343,0.9357,0.9423,0.9444,0.9617,0.9678,0.9730,1.0069,1.0170,1.0408,1.0953,1.0984,1.1271,1.1401,1.1598,1.1627,1.1699,1.1742,1.1789,1.2085,1.2289,1.2646,1.2899,1.3428,1.3486,1.3539,1.3733,1.3874,1.4144,1.4265,1.4547,1.4856,1.4962,1.5420,1.5789,1.5998,1.6046,1.6688,1.7464,1.7740,1.7886])
# channel_correlation_index = np.array([2,64,26,1,25,42,56,34,51,41,43,58,45,44,39,52,50,4,57,63,40,36,49,22,24,27,35,53,28,55,5,60,62,9,33,37,23,10,16,48,13,12,8,54,14,18,47,59,19,31,3,61,29,20,11,17,7,46,15,6,21,32,38,30],dtype = np.int64)
channel_correlation_index = np.array([64,1,2,17,58,26,25,43,39,9,44,40,11,24,35,41,50,42,57,33,34,49,56,10,45,4,16,51,27,28,59,3,8,36,23,52,29,15,22,13,12,63,60,7,21,48,14,53,18,19,6,20,55,61,62,5,47,46,37,38,54,32,31,30],dtype = np.int64)

bad_samples = np.asarray(range(long(2.14e5/8*4),long(2.2e5/8*4))+range(long(5.28e5/8*4),long(5.42e5/8*4)))
# words = range(150)
# del words[95:99]
# del words[37:39]
# # bad_words = np.asarray([37,38,95,96,97,98])
# words = np.asarray(words)
# np.random.shuffle(words)
# rand_words = words
# np.save('rand_words',rand_words)
# rand_words = np.load('rand_words.npy')

def normmat(matrix):
	global denomm, subb
	subb = matrix.mean()
	denomm = np.sqrt(matrix.var())
	out = (matrix-subb)/denomm
	# out = matrix/70
	print 'subb: ', subb, ',denomm: ', denomm 
	return out, denomm, subb

def denormmat(matrix,denomm,subb):
	out = np.arctanh(matrix)+subb
	return out

def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    mu = np.float32(quantization_channels - 1)
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = np.minimum(np.abs(audio), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)

def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    signal = 2 * (output.astype(np.float32) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1. / mu) * ((1 + mu)**np.abs(signal) - 1)
    return np.sign(signal) * magnitude

def spectrogram_log(x,frame_length=256,step=128,bin=16,stop = None):
	x = np.squeeze(x)
	if stop is None:
		stop = frame_length
	eps = 7./3-4./3-1

	# scale = np.ceil(np.logspace(0,np.log2(stop/2+1),bin+1,base=2)).astype(np.int32)
	scale = np.ceil(np.logspace(0,np.log2(512),bin+1,base=2)).astype(np.int32)
	spectrum = np.abs(librosa.core.stft(x.squeeze(),frame_length,step))

	out = np.zeros([bin,spectrum.shape[1]])
	for i in range(scale.shape[0]-1):
		out[i]=spectrum[scale[i]:scale[i+1]].sum(0)
	out = 10*np.log10(out+eps)
	return np.transpose(out[:-1],[1,0])

def spectrogram(x,frame_length=256,step=128,stop = None):
	x = np.squeeze(x)
	if stop is None:
		stop = frame_length
	eps = 7./3-4./3-1

	# scale = np.ceil(np.logspace(0,np.log2(stop/2+1),bin+1,base=2)).astype(np.int32)
	spectrum = np.abs(librosa.core.stft(x.squeeze(),frame_length,step))

	out = 10*np.log10(spectrum[1:]+eps)
	return np.transpose(out,[1,0])


def read_all_data(): 
	# ecogdata_train = h5py.File(DATA_DIR+'gdat_env_train.mat','r') 
	# ecog_train = np.asarray(ecogdata_train['gdat_env_train'])
	# ecog_train = signal.decimate(ecog_train,8,ftype='fir',axis=0)
	# # ecog_train,_,_ = normmat(ecog_train)

	# ecogdata_test = h5py.File(DATA_DIR+'gdat_env_test.mat','r') 
	# ecog_test = np.asarray(ecogdata_test['gdat_env_test'])
	# ecog_test = signal.decimate(ecog_test,8,ftype='fir',axis=0)

	# statics_ecog = ecog_test.mean(), np.sqrt(ecog_test.var()), ecog_test.mean(axis=0,keepdims=True), np.sqrt(ecog_test.var(axis=0, keepdims=True))
	# ecog_train = (ecog_train-statics_ecog[0])/statics_ecog[1]
	# ecog_test = (ecog_test-statics_ecog[0])/statics_ecog[1]
	# ecog_train = ecog_train[channel_correlation_index[-10:]]
	# ecog_test = ecog_test[channel_correlation_index[-10:]]

	ecogdata = h5py.File(DATA_DIR+'gdat_env_HD06.mat','r') 
	ecog = np.asarray(ecogdata['gdat_env'])
	# ecogdata = h5py.File(DATA_DIR+'graph_rep_selected_nodes.mat','r') 
	# ecog = np.asarray(ecogdata['xx']) 
	# ecog = ecog[:,channel_correlation_index[-16:]]
	ecog = np.minimum(ecog,40)
	ecog_test = signal.decimate(ecog[int(2e5):int(10e5)],32,ftype='fir',axis=0)
	statics_ecog = ecog_test.mean(), np.sqrt(ecog_test.var()), ecog_test.mean(axis=0,keepdims=True), np.sqrt(ecog_test.var(axis=0, keepdims=True))
	ecog = (signal.decimate(ecog,32,ftype='fir',axis=0)-statics_ecog[2])/statics_ecog[3]
	sp_ind = ecog.shape[0]

	ecogdata_sp = h5py.File(DATA_DIR+'gdat_env_HD06_sp.mat','r') 
	ecog_sp = np.asarray(ecogdata_sp['gdat_env'])
	# ecogdata_sp = h5py.File(DATA_DIR+'graph_rep_selected_nodes.mat','r') 
	# ecog_sp = np.asarray(ecogdata_sp['xx']) 
	# ecog_sp = ecog_sp[:,channel_correlation_index[-16:]]
	ecog_sp = np.minimum(ecog_sp,6)
	ecog_test = signal.decimate(ecog_sp[int(1e5):int(9e5)],32,ftype='fir',axis=0)
	statics_ecog = ecog_test.mean(), np.sqrt(ecog_test.var()), ecog_test.mean(axis=0,keepdims=True), np.sqrt(ecog_test.var(axis=0, keepdims=True))
	ecog_sp = (signal.decimate(ecog_sp,32,ftype='fir',axis=0)-statics_ecog[2])/statics_ecog[3]

	ecog = ecog_sp
	# ecog = np.concatenate([ecog,ecog_sp],axis=0)


	# spkdata_train = h5py.File(DATA_DIR+'spkr_train.mat','r') 
	# spkr_train = np.asarray(spkdata_train['spkr_train'])
	# spkr_train = signal.decimate(spkr_train,2,ftype='fir',axis=0)
	# # spkr_train = mu_law_encode(spkr_train, quantization_channels)
	# # spkrspec_train = spectrogram_log(spkr_train)
	# spkrspec_train = spectrogram(spkr_train)
	# spkrspec_train = spkrspec_train
	# spkrspec_train,_,_ = normmat(spkrspec_train)

	# spkdata_test = h5py.File(DATA_DIR+'spkr_test.mat','r') 
	# spkr_test = np.asarray(spkdata_test['spkr_test'])
	# spkr_test = signal.decimate(spkr_test,2,ftype='fir',axis=0)
	# # spkr_test = mu_law_encode(spkr_test, quantization_channels)
	# # spkrspec_test = spectrogram_log(spkr_test)
	# spkrspec_test = spectrogram(spkr_test)
	# spkrspec_test = spkrspec_test
	# spkrspec_test,_,_ = normmat(spkrspec_test)

	spkdata = h5py.File(DATA_DIR+'TF32_HD06_mic_sp.mat','r') 
	spkr = np.asarray(spkdata['TF32']['TFlog'])
	spkr,_,_ = normmat(spkr)

	# spkdata_sp = h5py.File(DATA_DIR+'TF32_HD06_sp.mat','r') 
	# spkr_sp = np.asarray(spkdata_sp['TF32']['TFlog'])
	# spkr_sp,_,_ = normmat(spkr_sp)

	spkr = spkr
	# spkr = np.concatenate([spkr,spkr_sp],axis = 0)


	label_mat = scipy.io.loadmat(DATA_DIR+'Events_HD06_sp.mat')['Events']['word'][0]
	labels = []
	unique_labels = []
	for i in range(label_mat.shape[0]):
		labels.append(label_mat[i][0])
		if label_mat[i][0] not in unique_labels:
			unique_labels.append(label_mat[i][0])
	label_ind = np.zeros([label_mat.shape[0]])
	for i in range(label_mat.shape[0]):
		label_ind[i] = unique_labels.index(labels[i])
	# labels = np.asarray(labels,dtype=np.int16)
	label_ind = np.asarray(label_ind,dtype=np.int16)
	label_onehot = np.eye(np.max(label_ind)+1)[label_ind]

	label_train = label_onehot[:100]
	label_test = label_onehot[100:]

	# target_mat = scipy.io.loadmat(DATA_DIR+'Events_HD06_sp.mat')['Events']['istarget'][0]
	# targets = []
	# for i in range(label_mat.shape[0]):
	# 	targets.append(target_mat[i][0,0])
	# targets = np.asarray(targets)
	# target_onehot = np.eye(2)[targets]
	# target_train = target_onehot[:100]
	# target_test = target_onehot[100:]

	peseudo_mat = scipy.io.loadmat(DATA_DIR+'Events_HD06_sp.mat')['Events']['ispseudo'][0]
	peseudos = []
	for i in range(label_mat.shape[0]):
		peseudos.append(peseudo_mat[i][0,0])
	peseudos = np.asarray(peseudos)
	peseudo_onehot = np.eye(2)[peseudos]
	peseudo_train = peseudo_onehot[:100]
	peseudo_test = peseudo_onehot[100:]

	start_ind = scipy.io.loadmat(DATA_DIR+'Events_HD06.mat')['Events']['onset'][0]
	start_ind = np.asarray([start_ind[i][0,0] for i in range(start_ind.shape[0])])#+697850
	start_ind = start_ind[100:]

	end_ind = scipy.io.loadmat(DATA_DIR+'Events_HD06.mat')['Events']['offset'][0]
	end_ind = np.asarray([end_ind[i][0,0] for i in range(end_ind.shape[0])])#+697850
	end_ind = end_ind[100:]

	start_ind_sp = scipy.io.loadmat(DATA_DIR+'Events_HD06_sp.mat')['Events']['onset'][0]
	start_ind_sp = np.asarray([start_ind_sp[i][0,0] for i in range(start_ind_sp.shape[0])])#+sp_ind*256
	start_ind_sp = start_ind_sp[100:]

	end_ind_sp = scipy.io.loadmat(DATA_DIR+'Events_HD06_sp.mat')['Events']['offset'][0]
	end_ind_sp = np.asarray([end_ind_sp[i][0,0] for i in range(end_ind_sp.shape[0])])#+sp_ind*256
	end_ind_sp = end_ind_sp[100:]

	start_ind = start_ind_sp
	end_ind =end_ind_sp
	# start_ind = np.concatenate([start_ind,start_ind_sp],axis=0)
	# end_ind = np.concatenate([end_ind,end_ind_sp],axis=0)

	# word_window_train = np.zeros([spkr_train.shape[0]])
	# word_window_test = np.zeros([spkr_test.shape[0]])
	# for i,j in zip(start_ind_train,end_ind_train):
	# 	word_window_train[i/256:j/256]=1.0
	# for i,j in zip(start_ind_test,end_ind_test):
	# 	word_window_test[i/256:j/256]=1.0


	# net = ecog_train 
	# nst = spkr_train
	# nett = ecog_test
	# nstt = spkr_test

	return ecog,spkr, label_train,label_test,start_ind,peseudo_train,peseudo_test#,word_window_train,word_window_test


# def get_batch(ecog, spkr, start_ind,threshold = 1.0, t_scale = 4, seg_length=160, batch_size=10,mode='train'):
# 	loop_idx = 0
# 	# t_scale = 8 # ratio of t_speech/t_ecog, currently 26701/26701=1
# 	n_delay_1 = 64#28 # samples
# 	n_delay_2 = 128#92#120#92 # samples 

# 	ecog_batch = np.zeros((batch_size,seg_length+n_delay_2-n_delay_1 , ecog.shape[1]))
# 	spkr_batch = np.zeros((batch_size,seg_length/t_scale,depth_)) 

# 	ini_idx = np.zeros((batch_size))

# 	if mode == 'train':
# 		x = np.array([i for i in range(start_ind.shape[0]) if i not in [37,38,95,96]])
# 	ind = np.random.choice(x,batch_size)
# 	if mode == 'train': 
# 		indx = np.maximum(start_ind[ind]/64 - np.random.choice(64*4,batch_size),0)
# 		# indx = start_ind[ind]/64
# 	for i in range(batch_size):
# 		ecog_batch[i] = ecog[indx[i]+n_delay_1:indx[i]+seg_length+n_delay_2]
# 		spkr_batch[i] = spkr[indx[i]/4 : (indx[i]+seg_length)/4]
# 	ecog_batch = ecog_batch[1:]
# 	spkr_batch = spkr_batch[1:]

# 	return ecog_batch, spkr_batch, ini_idx

 
# def get_batch(ecog, spkr, filter_width, threshold = 1.0, t_scale = 1, seg_length=160, batch_size=10):


# 	loop_idx = 0
# 	# t_scale = 8 # ratio of t_speech/t_ecog, currently 26701/26701=1
# 	n_delay_1 = 64/4#28 # samples
# 	n_delay_2 = 128/4#92#120#92 # samples 

# 	ecog_batch = np.zeros((1,seg_length+n_delay_2-n_delay_1 + filter_width-1, ecog.shape[1]))
# 	spkr_batch = np.zeros((1,seg_length/t_scale,depth_)) 

# 	ini_idx = np.zeros((batch_size))

# 	# for i in range(batch_size):
# 	while (loop_idx < batch_size):
# 		# loop_total += 1
# 		seg_index = random.randint(filter_width, ecog.shape[0] - seg_length - 1 - n_delay_2)
# 		# print(seg_index)
# 		if seg_index in bad_samples:
# 			continue
# 		spkspec_seg = spkr[seg_index/t_scale : (seg_index+seg_length)/t_scale]
# 		seg_energy = librosa.feature.rmse(S = np.squeeze(spkspec_seg))
# 		avg_energy = np.mean(seg_energy)
# 		# import pdb; pdb.set_trace()
# 		if avg_energy > threshold:
# 			spkr_batch = np.vstack((spkr_batch, spkspec_seg[np.newaxis,:]))
# 			ecogspec_seg = ecog[seg_index+n_delay_1 -filter_width+1: seg_index+seg_length+n_delay_2]
# 			ecog_batch = np.vstack((ecog_batch, ecogspec_seg[np.newaxis,:]))
# 			ini_idx[loop_idx] = seg_index
# 			loop_idx += 1

# 	# print(ecog_batch.shape)
# 	ecog_batch = ecog_batch[1:]
# 	spkr_batch = spkr_batch[1:]

# 	return ecog_batch, spkr_batch, ini_idx

# def get_batch(ecog, spkr, filter_width, threshold = 1.0, t_scale = 1, seg_length=160, batch_size=10):


# 	loop_idx = 0
# 	# t_scale = 8 # ratio of t_speech/t_ecog, currently 26701/26701=1
# 	n_delay_1 = 64/4#28 # samples
# 	n_delay_2 = 128/4#92#120#92 # samples 

# 	ecog_batch = np.zeros((1,seg_length+n_delay_2-n_delay_1 + filter_width-1, ecog.shape[1]))
# 	spkr_batch = np.zeros((1,seg_length/t_scale,depth_)) 

# 	ini_idx = np.zeros((batch_size))

# 	# for i in range(batch_size):
# 	while (loop_idx < batch_size):
# 		# loop_total += 1
# 		seg_index = random.randint(filter_width/2, ecog.shape[0] - seg_length - 1 - n_delay_2 - filter_width)
# 		# print(seg_index)
# 		if seg_index in bad_samples:
# 			continue
# 		spkspec_seg = spkr[seg_index/t_scale : (seg_index+seg_length)/t_scale]
# 		seg_energy = librosa.feature.rmse(S = np.squeeze(spkspec_seg))
# 		avg_energy = np.mean(seg_energy)
# 		# import pdb; pdb.set_trace()
# 		if avg_energy > threshold:
# 			spkr_batch = np.vstack((spkr_batch, spkspec_seg[np.newaxis,:]))
# 			ecogspec_seg = ecog[seg_index+n_delay_1 -filter_width/2: seg_index+seg_length+n_delay_2+filter_width-filter_width/2-1]
# 			ecog_batch = np.vstack((ecog_batch, ecogspec_seg[np.newaxis,:]))
# 			ini_idx[loop_idx] = seg_index
# 			loop_idx += 1

# 	# print(ecog_batch.shape)
# 	ecog_batch = ecog_batch[1:]
# 	spkr_batch = spkr_batch[1:]

# 	return ecog_batch, spkr_batch, ini_idx

def get_batch(ecog, spkr, cv, rand_words, start_ind, mode, threshold = 1.0, t_scale = 1, seg_length=160, batch_size=10):


	words = range(start_ind.shape[0])
	# del words[95:99]
	# del words[37:39]
	# bad_words = np.asarray([37,38,95,96,97,98])
	rand_words = np.asarray(words)

	words_num = rand_words.size
	test_num = int(math.ceil(words_num*1.0/2)) # 2 fold cross val

	n_delay_1 = 64/4#28 # samples
	n_delay_2 = 128/4#92#120#92 # samples 


	if mode == 'train':
		# x = rand_words 
		x = np.concatenate([rand_words[0:cv*test_num],rand_words[(cv+1)*test_num:]])
	elif mode =='test':
		x = rand_words[cv*test_num:(cv+1)*test_num]

	
	if mode == 'train': 
		ecog_batch = np.zeros((batch_size,seg_length+n_delay_2-n_delay_1 ,64))
		spkr_batch = np.zeros((batch_size, seg_length,32))
		ind = np.random.choice(x,batch_size)
		indx = np.maximum(start_ind[ind]/256 - np.random.choice(64,batch_size),0)
		for i in range(batch_size):
			ecog_batch[i] = ecog[indx[i]+n_delay_1:indx[i]+seg_length+n_delay_2]
			spkr_batch[i] = spkr[indx[i]:indx[i]+seg_length]

		# indx = start_ind[ind]/256

	elif mode=='test':
		indx = np.maximum(start_ind[x]/256 - 32,0)

		ecog_batch = np.zeros((x.size,seg_length+n_delay_2-n_delay_1 ,64))
		spkr_batch = np.zeros((x.size, seg_length,32))

		for i in range(x.size):
			ecog_batch[i] = ecog[indx[i]+n_delay_1:indx[i]+seg_length+n_delay_2]
			spkr_batch[i] = spkr[indx[i]:indx[i]+seg_length]

	return ecog_batch, spkr_batch

def get_batch_show(ecog, spkr,start_ind,threshold = 1.0, t_scale = 64*4, seg_length=160):
	# t_scale = 8 # ratio of t_speech/t_ecog, currently 26701/26701=1
	n_delay_1 = 64/4 # samples
	n_delay_2 = 128/4#120#92 # samples

	ecog_batch = np.zeros((start_ind.shape[0],seg_length+n_delay_2-n_delay_1 ,ecog.shape[-1]))
	spkr_batch = np.zeros((start_ind.shape[0],seg_length,32))
	# spkr_batch = np.zeros((start_ind.shape[0],seg_length/4,32))
	# if mode == 'train':
	# 	x = np.array([i for i in range(start_ind.shape[0]) if i not in [37,38,95,96]])
	# elif mode =='test':
	# 	x = np.array([i for i in range(start_ind.shape[0])])
	# ind = np.random.choice(x,batch_size)
	# if mode == 'train': 
	# 	# indx = np.maximum(start_ind[ind]/t_scale - np.random.choice(64*4,batch_size),0)
	# 	indx = start_ind[ind]/t_scale
	# elif mode=='test':
	indx = start_ind/t_scale
	# label_ = label[ind]
	# target_ = target[ind]
	# peseudo_ = peseudo[ind]
	for i, ind in enumerate(indx):
		ind = np.maximum(ind-160/4,0)
		ecog_batch[i] = ecog[ind+n_delay_1:ind+seg_length+n_delay_2]
		spkr_batch[i] = spkr[ind : (ind+seg_length)]
		# spkr_batch[i] = spkr[ind/4 : (ind+seg_length)/4]
	return ecog_batch,spkr_batch
