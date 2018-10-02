# import scipy.io as scipy_io
# from scipy import signal
# import h5py
# import numpy as np
# import random
# import librosa
# import os

# DATA_DIR = '../../neural_decoding_data/adeen/'
# DATA_DIR2 = '../../neural_decoding_data/shtooka/'
# # DATA_DIR = './data/'

# depth_ = 32#128#15 
# quantization_channels = 2**8

# def normmat(matrix):
# 	global denomm, subb
# 	subb = matrix.mean()
# 	denomm = np.sqrt(matrix.var())
# 	out = (matrix-subb)/denomm
# 	print 'subb: ', subb, ',denomm: ', denomm 
# 	return out, denomm, subb

# def denormmat(matrix,denomm,subb):
# 	out = np.arctanh(matrix)+subb
# 	return out

# def mu_law_encode(audio, quantization_channels):
#     '''Quantizes waveform amplitudes.'''
#     mu = np.float32(quantization_channels - 1)
#     # Perform mu-law companding transformation (ITU-T, 1988).
#     # Minimum operation is here to deal with rare large amplitudes caused
#     # by resampling.
#     safe_audio_abs = np.minimum(np.abs(audio), 1.0)
#     magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
#     signal = np.sign(audio) * magnitude
#     # Quantize signal to the specified number of levels.
#     return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)

# def mu_law_decode(output, quantization_channels):
#     '''Recovers waveform from quantized values.'''
#     mu = quantization_channels - 1
#     # Map values back to [-1, 1].
#     signal = 2 * (output.astype(np.float32) / mu) - 1
#     # Perform inverse of mu-law transformation.
#     magnitude = (1. / mu) * ((1 + mu)**np.abs(signal) - 1)
#     return np.sign(signal) * magnitude

# def spectrogram_log(x,frame_length=256,step=128,bin=16,stop = None):
# 	x = np.squeeze(x)
# 	if stop is None:
# 		stop = frame_length
# 	eps = 7./3-4./3-1

# 	# scale = np.ceil(np.logspace(0,np.log2(stop/2+1),bin+1,base=2)).astype(np.int32)
# 	scale = np.ceil(np.logspace(0,np.log2(512),bin+1,base=2)).astype(np.int32)
# 	spectrum = np.abs(librosa.core.stft(x.squeeze(),frame_length,step))

# 	out = np.zeros([bin,spectrum.shape[1]])
# 	for i in range(scale.shape[0]-1):
# 		out[i]=spectrum[scale[i]:scale[i+1]].sum(0)
# 	out = 10*np.log10(out+eps)
# 	return np.transpose(out[:-1],[1,0])

# def spectrogram(x,frame_length=256,step=128,stop = None):
# 	x = np.squeeze(x)
# 	if stop is None:
# 		stop = frame_length
# 	eps = 7./3-4./3-1

# 	# scale = np.ceil(np.logspace(0,np.log2(stop/2+1),bin+1,base=2)).astype(np.int32)
# 	spectrum = np.abs(librosa.core.stft(x.squeeze(),frame_length,step))

# 	out = 10*np.log10(spectrum[1:]+eps)
# 	return np.transpose(out,[1,0])

# def find_files(directory, pattern='*.flac'):
#     '''Recursively finds all files matching the pattern.'''
#     files = []
#     for root, dirnames, filenames in os.walk(directory):
#         for filename in fnmatch.filter(filenames, pattern):
#             files.append(os.path.join(root, filename))
#     return files

# def read_all_data(): 
# 	ecogdata_train = h5py.File(DATA_DIR+'gdat_env_train.mat','r') 
# 	ecog_train = np.asarray(ecogdata_train['gdat_env_train'])
# 	ecog_train = signal.decimate(ecog_train,8,ftype='fir',axis=0)
# 	ecog_train,_,_ = normmat(ecog_train)

# 	ecogdata_test = h5py.File(DATA_DIR+'gdat_env_test.mat','r') 
# 	ecog_test = np.asarray(ecogdata_test['gdat_env_test'])
# 	ecog_test = signal.decimate(ecog_test,8,ftype='fir',axis=0)
# 	ecog_test,_,_ = normmat(ecog_test)


# 	# spkdata_train = h5py.File(DATA_DIR+'spkr_train.mat','r') 
# 	# spkr_train = np.asarray(spkdata_train['spkr_train'])
# 	# spkr_train = signal.decimate(spkr_train,2,ftype='fir',axis=0)
# 	# # spkr_train = mu_law_encode(spkr_train, quantization_channels)
# 	# # spkrspec_train = spectrogram_log(spkr_train)
# 	# spkrspec_train = spectrogram(spkr_train)
# 	# spkrspec_train = spkrspec_train
# 	# spkrspec_train,_,_ = normmat(spkrspec_train)

# 	# spkdata_test = h5py.File(DATA_DIR+'spkr_test.mat','r') 
# 	# spkr_test = np.asarray(spkdata_test['spkr_test'])
# 	# spkr_test = signal.decimate(spkr_test,2,ftype='fir',axis=0)
# 	# # spkr_test = mu_law_encode(spkr_test, quantization_channels)
# 	# # spkrspec_test = spectrogram_log(spkr_test)
# 	# spkrspec_test = spectrogram(spkr_test)
# 	# spkrspec_test = spkrspec_test
# 	# spkrspec_test,_,_ = normmat(spkrspec_test)

# 	spkdata_train = h5py.File(DATA_DIR+'TF32_train.mat','r') 
# 	spkr_train = np.asarray(spkdata_train['TF32']['TFlog'])
# 	# spkr_train = np.transpose(spkr_train,[1,0]) 
# 	spkrspec_train,_,_ = normmat(spkr_train)

# 	spkdata_train2 = h5py.File(DATA_DIR2+'TF32_b_3.mat','r') 
# 	spkr_train2 = np.asarray(spkdata_train2['TF32_b']['TFlog'])
# 	# spkr_train = np.transpose(spkr_train,[1,0]) 
# 	spkrspec_train2,_,_ = normmat(spkr_train2)

# 	spkdata_test = h5py.File(DATA_DIR+'TF32_test.mat','r') 
# 	spkr_test = np.asarray(spkdata_test['TF32']['TFlog'])
# 	# spkr_test = np.transpose(spkr_test,[1,0])
# 	spkrspec_test,_,_ = normmat(spkr_test)

# 	good_idxs_train = []
# 	indication = np.sqrt((spkrspec_train**2).mean(axis=1))
# 	for i in xrange(spkrspec_train.shape[0]-150):
# 		if indication[i:i+20].mean()<0.8 and indication[i+108:i+128].mean()<0.8 and indication[i+20:i+108].mean()>1.0:
# 			good_idxs_train.append(i)

# 	good_idxs_test = []
# 	indication = np.sqrt((spkrspec_test**2).mean(axis=1))
# 	for i in xrange(spkrspec_test.shape[0]-150):
# 		if indication[i:i+20].mean()<0.8 and indication[i+108:i+128].mean()<0.8 and indication[i+20:i+108].mean()>1.0:
# 			good_idxs_test.append(i)

# 	net = ecog_train 
# 	nst = spkr_train
# 	nett = ecog_test
# 	nstt = spkr_test
 
# 	return net, nst, nett, nstt, spkrspec_train, spkrspec_train2, spkrspec_test, np.asarray(good_idxs_train), np.asarray(good_idxs_test)

# def rmse(input):
# 	return np.sqrt((input**2).mean())

# def get_batch(ecog, spkr, spkr2, good_idxs, threshold = 1.0, threshold2 = 1.0, t_scale = 4, seg_length=160, batch_size=10):


	
# 	# t_scale = 8 # ratio of t_speech/t_ecog, currently 26701/26701=1
# 	n_delay_1 = 28 # samples
# 	n_delay_2 = 92#120#92 # samples

# 	ecog_batch = np.zeros((1,seg_length+n_delay_2-n_delay_1 ,64))
# 	spkr_batch = np.zeros((1,seg_length/t_scale,depth_)) 
# 	spkr_batch2 = np.zeros((1,seg_length/t_scale,depth_)) 

# 	ini_idx2 = np.asarray(random.sample(good_idxs, batch_size))
# 	ini_idx = ini_idx2*t_scale
# 	for i in range(batch_size):
# 		spkspec_seg = spkr2[ini_idx2[i] : ini_idx2[i]+seg_length/t_scale]
# 		spkr_batch = np.vstack((spkr_batch, spkspec_seg[np.newaxis,:]))
# 		ecogspec_seg = ecog[ini_idx[i]+n_delay_1 : ini_idx[i]+seg_length+n_delay_2]
# 		ecog_batch = np.vstack((ecog_batch, ecogspec_seg[np.newaxis,:]))
# 	# ini_idx = np.zeros((batch_size))
# 	# ini_idx2 = np.zeros((batch_size))	
# 	# for i in range(batch_size):
# 	# loop_idx = 0
# 	# while (loop_idx < batch_size):
# 	# 	# loop_total += 1
# 	# 	seg_index = random.randint(0, ecog.shape[0] - seg_length - 1 - n_delay_2)
# 	# 	seg_index2 = random.randint(0, spkr2.shape[0] - seg_length/t_scale-1)
# 	# 	# print(seg_index)
# 	# 	spkspec_seg = spkr[seg_index/t_scale : (seg_index+seg_length)/t_scale]
# 	# 	spkspec_seg2 = spkr2[seg_index2 : seg_index2+seg_length/t_scale]
# 	# 	# seg_energy = max( rmse(spkspec_seg[20:30])
# 	# 	# 				, rmse(spkspec_seg[30:40])
# 	# 	# 				, rmse(spkspec_seg[40:50])
# 	# 	# 				, rmse(spkspec_seg[50:60])
# 	# 	# 				, rmse(spkspec_seg[60:70])
# 	# 	# 				, rmse(spkspec_seg[70:80])
# 	# 	# 				, rmse(spkspec_seg[80:90])
# 	# 	# 				, rmse(spkspec_seg[90:100]))
# 	# 	seg_energy = rmse(spkspec_seg[20:-20])
# 	# 	avg_energy = np.mean(seg_energy)
# 	# 	avg_energy_start = rmse(spkspec_seg[:20])
# 	# 	avg_energy_end = rmse(spkspec_seg[-20:])
# 	# 	seg_energy2 = librosa.feature.rmse(S = np.squeeze(spkspec_seg2))
# 	# 	avg_energy2 = np.mean(seg_energy2)
# 	# 	# import pdb; pdb.set_trace()
# 	# 	if avg_energy > threshold and avg_energy_start<1.0 and avg_energy_end<1.0:
# 	# 		spkr_batch = np.vstack((spkr_batch, spkspec_seg[np.newaxis,:]))
# 	# 		ecogspec_seg = ecog[seg_index+n_delay_1 : seg_index+seg_length+n_delay_2]
# 	# 		ecog_batch = np.vstack((ecog_batch, ecogspec_seg[np.newaxis,:]))
# 	# 		ini_idx[loop_idx] = seg_index
# 	# 		loop_idx += 1

# 	loop_idx2 = 0
# 	while (loop_idx2 < batch_size):
# 		# loop_total += 1
# 		seg_index2 = random.randint(0, spkr2.shape[0] - seg_length/t_scale-1)
# 		# print(seg_index)
# 		spkspec_seg2 = spkr2[seg_index2 : seg_index2+seg_length/t_scale]
# 		seg_energy2 = librosa.feature.rmse(S = np.squeeze(spkspec_seg2))
# 		avg_energy2 = np.mean(seg_energy2)
# 		# import pdb; pdb.set_trace()
# 		if avg_energy2 > threshold2:
# 			spkr_batch2 = np.vstack((spkr_batch2, spkspec_seg[np.newaxis,:]))
# 			ini_idx2[loop_idx2] = seg_index2
# 			loop_idx2 += 1

# 	# print(ecog_batch.shape)
# 	ecog_batch = ecog_batch[1:]
# 	spkr_batch = spkr_batch[1:]
# 	spkr_batch2 = spkr_batch2[1:]

# 	return ecog_batch, spkr_batch,spkr_batch2, ini_idx,ini_idx2




















import scipy.io as scipy_io
from scipy import signal
import h5py
import numpy as np
import random
import librosa
import os

DATA_DIR = '../../neural_decoding_data/adeen/'
DATA_DIR2 = '../../neural_decoding_data/shtooka/'
# DATA_DIR = './data/'

depth_ = 32#128#15 
quantization_channels = 2**8

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

def find_files(directory, pattern='*.flac'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def read_all_data(): 
	ecogdata_train = h5py.File(DATA_DIR+'gdat_env_train.mat','r') 
	ecog_train = np.asarray(ecogdata_train['gdat_env_train'])
	ecog_train = signal.decimate(ecog_train,8,ftype='fir',axis=0)
	ecog_train,_,_ = normmat(ecog_train)

	ecogdata_test = h5py.File(DATA_DIR+'gdat_env_test.mat','r') 
	ecog_test = np.asarray(ecogdata_test['gdat_env_test'])
	ecog_test = signal.decimate(ecog_test,8,ftype='fir',axis=0)
	ecog_test,_,_ = normmat(ecog_test)


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

	spkdata_train2 = h5py.File(DATA_DIR2+'TF32_b_3.mat','r') 
	spkr_train2 = np.asarray(spkdata_train2['TF32_b']['TFlog'])
	# spkr_train = np.transpose(spkr_train,[1,0]) 
	spkrspec_train2,_,_ = normmat(spkr_train2)

	spkdata_test = h5py.File(DATA_DIR+'TF32_test.mat','r') 
	spkr_test = np.asarray(spkdata_test['TF32']['TFlog'])
	# spkr_test = np.transpose(spkr_test,[1,0])
	spkrspec_test,_,_ = normmat(spkr_test)

	net = ecog_train 
	nst = spkr_train
	nett = ecog_test
	nstt = spkr_test

	return net, nst, nett, nstt, spkrspec_train, spkrspec_train2, spkrspec_test


def get_batch(ecog, spkr, spkr2, threshold = 1.0, threshold2 = 1.0, t_scale = 4, seg_length=160, batch_size=10):


	
	# t_scale = 8 # ratio of t_speech/t_ecog, currently 26701/26701=1
	n_delay_1 = 28 # samples
	n_delay_2 = 92#120#92 # samples

	ecog_batch = np.zeros((1,seg_length+n_delay_2-n_delay_1 ,64))
	spkr_batch = np.zeros((1,seg_length/t_scale,depth_)) 
	spkr_batch2 = np.zeros((1,seg_length/t_scale,depth_)) 

	ini_idx = np.zeros((batch_size))
	ini_idx2 = np.zeros((batch_size))	
	# for i in range(batch_size):
	loop_idx = 0
	while (loop_idx < batch_size):
		# loop_total += 1
		seg_index = random.randint(0, ecog.shape[0] - seg_length - 1 - n_delay_2)
		seg_index2 = random.randint(0, spkr2.shape[0] - seg_length/t_scale-1)
		# print(seg_index)
		spkspec_seg = spkr[seg_index/t_scale : (seg_index+seg_length)/t_scale]
		spkspec_seg2 = spkr2[seg_index2 : seg_index2+seg_length/t_scale]
		seg_energy = librosa.feature.rmse(S = np.squeeze(spkspec_seg))
		avg_energy = np.mean(seg_energy)
		seg_energy2 = librosa.feature.rmse(S = np.squeeze(spkspec_seg2))
		avg_energy2 = np.mean(seg_energy2)
		# import pdb; pdb.set_trace()
		if avg_energy > threshold:
			spkr_batch = np.vstack((spkr_batch, spkspec_seg[np.newaxis,:]))
			ecogspec_seg = ecog[seg_index+n_delay_1 : seg_index+seg_length+n_delay_2]
			ecog_batch = np.vstack((ecog_batch, ecogspec_seg[np.newaxis,:]))
			ini_idx[loop_idx] = seg_index
			loop_idx += 1

	loop_idx2 = 0
	while (loop_idx2 < batch_size):
		# loop_total += 1
		seg_index2 = random.randint(0, spkr2.shape[0] - seg_length/t_scale-1)
		# print(seg_index)
		spkspec_seg2 = spkr2[seg_index2 : seg_index2+seg_length/t_scale]
		seg_energy2 = librosa.feature.rmse(S = np.squeeze(spkspec_seg2))
		avg_energy2 = np.mean(seg_energy2)
		# import pdb; pdb.set_trace()
		if avg_energy2 > threshold2:
			spkr_batch2 = np.vstack((spkr_batch2, spkspec_seg[np.newaxis,:]))
			ini_idx2[loop_idx2] = seg_index2
			loop_idx2 += 1

	# print(ecog_batch.shape)
	ecog_batch = ecog_batch[1:]
	spkr_batch = spkr_batch[1:]
	spkr_batch2 = spkr_batch2[1:]

	return ecog_batch, spkr_batch,spkr_batch2, ini_idx,ini_idx2


