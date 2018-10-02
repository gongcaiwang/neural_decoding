import scipy
import scipy.io as scipy_io
from scipy import signal
import h5py
import numpy as np
import random
import librosa
DATA_DIR = '../../neural_decoding_data/adeen/'
# DATA_DIR = './data/'

depth_ = 32#128#15 
quantization_channels = 2**8
channel_correlation = 1e5*np.array([0.4167,0.4178,0.4430,0.4459,0.4655,0.5357,0.5928,0.5969,0.5993,0.6141,0.6149,0.6150,0.6181,0.6204,0.6345,0.6740,0.7073,0.7271,0.7295,0.7345,0.7522,0.7825,0.9068,0.9343,0.9357,0.9423,0.9444,0.9617,0.9678,0.9730,1.0069,1.0170,1.0408,1.0953,1.0984,1.1271,1.1401,1.1598,1.1627,1.1699,1.1742,1.1789,1.2085,1.2289,1.2646,1.2899,1.3428,1.3486,1.3539,1.3733,1.3874,1.4144,1.4265,1.4547,1.4856,1.4962,1.5420,1.5789,1.5998,1.6046,1.6688,1.7464,1.7740,1.7886])
# channel_correlation_index = np.array([2,64,26,1,25,42,56,34,51,41,43,58,45,44,39,52,50,4,57,63,40,36,49,22,24,27,35,53,28,55,5,60,62,9,33,37,23,10,16,48,13,12,8,54,14,18,47,59,19,31,3,61,29,20,11,17,7,46,15,6,21,32,38,30],dtype = np.int64)
channel_correlation_index = np.array([64,1,2,17,58,26,25,43,39,9,44,40,11,24,35,41,50,42,57,33,34,49,56,10,45,4,16,51,27,28,59,3,8,36,23,52,29,15,22,13,12,63,60,7,21,48,14,53,18,19,6,20,55,61,62,5,47,46,37,38,54,32,31,30],dtype = np.int64)

bad_samples = np.asarray(range(long(2.14e5/8*4),long(2.2e5/8*4))+range(long(5.28e5/8*4),long(5.42e5/8*4)))

def normmat(matrix):
	global denomm, subb
	subb = matrix.mean()
	denomm = np.sqrt(matrix.var())
	out = (matrix-subb)/denomm
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
	# ecog_train = (ecog_train-statics_ecog[2])/statics_ecog[3]
	# ecog_test = (ecog_test-statics_ecog[2])/statics_ecog[3]


	# ecogdata = h5py.File(DATA_DIR+'graph_rep.mat','r') 
	# ecog = np.asarray(ecogdata['xx']) 
	ecogdata = h5py.File(DATA_DIR+'gdat_env2.mat','r') 
	ecog = np.asarray(ecogdata['gdat_env'])
	#ecog = ecog[:,channel_correlation_index[-10:]]
	ecog = np.minimum(ecog,30)

	ecog_train = ecog[:557250]
	ecog_test = ecog[557250:]
	ecog_train = signal.decimate(ecog_train,32,ftype='fir',axis=0)
	ecog_test = signal.decimate(ecog_test,32,ftype='fir',axis=0)
	# ecog_train = signal.decimate(ecog_train,8,ftype='fir',axis=0)
	# ecog_test = signal.decimate(ecog_test,8,ftype='fir',axis=0)
	statics_ecog = ecog_test.mean(), np.sqrt(ecog_test.var()), ecog_test.mean(axis=0,keepdims=True), np.sqrt(ecog_test.var(axis=0, keepdims=True))
	ecog_train = (ecog_train-statics_ecog[2])/statics_ecog[3]
	ecog_test = (ecog_test-statics_ecog[2])/statics_ecog[3]


	ecogdata2 = h5py.File(DATA_DIR+'graph_rep_HD06.mat','r') 
	ecog2 = np.asarray(ecogdata2['xx']) 
	# ecogdata2 = h5py.File(DATA_DIR+'gdat_env_HD06.mat','r') 
	# ecog2 = np.asarray(ecogdata2['gdat_env']) 
	ecog_train2 = ecog2[697600/32:]
	# ecog_test = ecog[557250:]
	ecog_train2 = signal.decimate(ecog_train2,32,ftype='fir',axis=0)
	statics_ecog2 = ecog_train2.mean(), np.sqrt(ecog_train2.var()), ecog_train2.mean(axis=0,keepdims=True), np.sqrt(ecog_train2.var(axis=0, keepdims=True))
	ecog_train2 = (ecog_train2-statics_ecog2[2])/statics_ecog2[3]

	# ecog_train = np.concatenate((ecog_train,ecog_train2),axis=0)
	# ecog_test,_,_ = normmat(ecog_test)


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

	spkdata_train = h5py.File(DATA_DIR+'TF32_train.mat','r') 
	spkr_train = np.asarray(spkdata_train['TF32']['TFlog'])
	# spkr_train = np.transpose(spkr_train,[1,0]) 
	spkrspec_train,_,_ = normmat(spkr_train)

	spkdata_train2 = h5py.File(DATA_DIR+'TF32_HD06.mat','r') 
	spkr_train2 = np.asarray(spkdata_train2['TF32']['TFlog'])
	# spkr_train = np.transpose(spkr_train,[1,0]) 
	spkrspec_train2,_,_ = normmat(spkr_train2[697600/256:])
	# spkrspec_train = np.concatenate((spkrspec_train,spkrspec_train2),axis=0)

	spkdata_test = h5py.File(DATA_DIR+'TF32_test.mat','r') 
	spkr_test = np.asarray(spkdata_test['TF32']['TFlog'])
	# spkr_test = np.transpose(spkr_test,[1,0])
	spkrspec_test,_,_ = normmat(spkr_test)

	label_mat = scipy.io.loadmat(DATA_DIR+'Events.mat')['Events']['word'][0]
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

	target_mat = scipy.io.loadmat(DATA_DIR+'Events.mat')['Events']['istarget'][0]
	targets = []
	for i in range(label_mat.shape[0]):
		targets.append(target_mat[i][0,0])
	targets = np.asarray(targets)
	target_onehot = np.eye(2)[targets]
	target_train = target_onehot[:100]
	target_test = target_onehot[100:]

	peseudo_mat = scipy.io.loadmat(DATA_DIR+'Events.mat')['Events']['ispseudo'][0]
	peseudos = []
	for i in range(label_mat.shape[0]):
		peseudos.append(peseudo_mat[i][0,0])
	peseudos = np.asarray(peseudos)
	peseudo_onehot = np.eye(2)[peseudos]
	peseudo_train = peseudo_onehot[:100]
	peseudo_test = peseudo_onehot[100:]

	start_ind = scipy.io.loadmat(DATA_DIR+'Events.mat')['Events']['onset'][0]
	start_ind = np.asarray([start_ind[i][0,0] for i in range(start_ind.shape[0])])
	start_ind_train = start_ind[:100]
	start_ind_test = start_ind[100:]-4458000


	net = ecog_train 
	nst = spkr_train
	nett = ecog_test
	nstt = spkr_test

	return net, nst, nett, nstt, spkrspec_train,spkrspec_test, label_train,label_test,start_ind_train,start_ind_test,target_train,target_test,peseudo_train,peseudo_test


def get_batch(ecog, spkr, threshold = 1.0, t_scale = 1, seg_length=160, batch_size=10):


	loop_idx = 0
	# t_scale = 8 # ratio of t_speech/t_ecog, currently 26701/26701=1
	n_delay_1 = 64/4#28 # samples
	n_delay_2 = 128/4#92#120#92 # samples 

	ecog_batch = np.zeros((1,seg_length+n_delay_2-n_delay_1, ecog.shape[1]))
	spkr_batch = np.zeros((1,seg_length/t_scale,depth_)) 

	ini_idx = np.zeros((batch_size))

	# for i in range(batch_size):
	while (loop_idx < batch_size):
		# loop_total += 1
		seg_index = random.randint(0, ecog.shape[0] - seg_length - 1 - n_delay_2)
		# print(seg_index)
		if seg_index in bad_samples:
			continue
		spkspec_seg = spkr[seg_index/t_scale : (seg_index+seg_length)/t_scale]
		seg_energy = librosa.feature.rmse(S = np.squeeze(spkspec_seg))
		avg_energy = np.mean(seg_energy)
		# import pdb; pdb.set_trace()
		if avg_energy > threshold:
			spkr_batch = np.vstack((spkr_batch, spkspec_seg[np.newaxis,:]))
			ecogspec_seg = ecog[seg_index+n_delay_1 : seg_index+seg_length+n_delay_2]
			ecog_batch = np.vstack((ecog_batch, ecogspec_seg[np.newaxis,:]))
			ini_idx[loop_idx] = seg_index
			loop_idx += 1

	# print(ecog_batch.shape)
	ecog_batch = ecog_batch[1:]
	spkr_batch = spkr_batch[1:]

	return ecog_batch, spkr_batch, ini_idx

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


