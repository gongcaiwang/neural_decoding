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

	spkdata_test = h5py.File(DATA_DIR+'TF32_test.mat','r') 
	spkr_test = np.asarray(spkdata_test['TF32']['TFlog'])
	# spkr_test = np.transpose(spkr_test,[1,0])
	spkrspec_test,_,_ = normmat(spkr_test)

	net = ecog_train 
	nst = spkr_train
	nett = ecog_test
	nstt = spkr_test

	return net, nst, nett, nstt, spkrspec_train,spkrspec_test


def get_batch(ecog, spkr, threshold = 1.0, t_scale = 4, seg_length=160, batch_size=10):


	loop_idx = 0
	# t_scale = 8 # ratio of t_speech/t_ecog, currently 26701/26701=1
	n_delay_1 = 28 # samples
	n_delay_2 = 92#120#92 # samples

	ecog_batch = np.zeros((1,seg_length+n_delay_2-n_delay_1 ,64))
	spkr_batch = np.zeros((1,seg_length/t_scale,depth_)) 

	ini_idx = np.zeros((batch_size))

	# for i in range(batch_size):
	while (loop_idx < batch_size):
		# loop_total += 1
		seg_index = random.randint(0, ecog.shape[0] - seg_length - 1 - n_delay_2)
		# print(seg_index)
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


