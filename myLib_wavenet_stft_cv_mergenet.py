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

# depth = 24
# num_dataset = 2
# num_words = 32
test_num_perset = 50
depth_ = 32#128#15 
quantization_channels = 2**8

channel_correlation = 1e5*np.array([0.4167,0.4178,0.4430,0.4459,0.4655,0.5357,0.5928,0.5969,0.5993,0.6141,0.6149,0.6150,0.6181,0.6204,0.6345,0.6740,0.7073,0.7271,0.7295,0.7345,0.7522,0.7825,0.9068,0.9343,0.9357,0.9423,0.9444,0.9617,0.9678,0.9730,1.0069,1.0170,1.0408,1.0953,1.0984,1.1271,1.1401,1.1598,1.1627,1.1699,1.1742,1.1789,1.2085,1.2289,1.2646,1.2899,1.3428,1.3486,1.3539,1.3733,1.3874,1.4144,1.4265,1.4547,1.4856,1.4962,1.5420,1.5789,1.5998,1.6046,1.6688,1.7464,1.7740,1.7886])
# channel_correlation_index = np.array([2,64,26,1,25,42,56,34,51,41,43,58,45,44,39,52,50,4,57,63,40,36,49,22,24,27,35,53,28,55,5,60,62,9,33,37,23,10,16,48,13,12,8,54,14,18,47,59,19,31,3,61,29,20,11,17,7,46,15,6,21,32,38,30],dtype = np.int64)
channel_correlation_index = np.array([64,1,2,17,58,26,25,43,39,9,44,40,11,24,35,41,50,42,57,33,34,49,56,10,45,4,16,51,27,28,59,3,8,36,23,52,29,15,22,13,12,63,60,7,21,48,14,53,18,19,6,20,55,61,62,5,47,46,37,38,54,32,31,30],dtype = np.int64)



dataset_filename_ecog=[['gdat_env3.mat'],
				       ['gdat_env_HD06.mat','gdat_env_HD06_sp.mat']]

dataset_filename_spkr=[['TF32_train.mat','TF32_test.mat'],
						['TF32_HD06.mat','TF32_HD06_sp.mat']]
event_filename = [['Events.mat'],
				  ['Events_HD06.mat','Events_HD06_sp.mat']]
start_ind_bias = [[0],[2734,0]]
data_range_max = [[30],[40,40]]
# bad_channels = [[None],[None,None]]
bad_channels = [[-2],[None,None]]
event_range = [[None],[-100,-100]]
statics_samples_ecog = [[17415,None],[6250,31250,3125,28125]] #samples for compute statics, note it should be the sample NO. after downsample
statics_samples_spkr = [[0,None,0,None],[0,None,0,None]]
bad_samples = [[37,38,95,96,97],[]]
# bad_samples = [np.asarray(range(long(2.14e5/32),long(2.2e5/32))+range(long(5.28e5/32),long(5.42e5/32))),np.asarray([])]


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
	ecog_alldataset = []
	spkr_alldataset = []
	start_ind_alldataset = []
	word_alldataset = []
	ecog_len = []

	for ds in range(len(dataset_filename_ecog)):
		ecog_ = []
		ecog_len_=[0]
		for file in range(len(dataset_filename_ecog[ds])):
			ecogdata = h5py.File(DATA_DIR+dataset_filename_ecog[ds][file],'r')
			ecog = np.asarray(ecogdata['gdat_env'])
			ecog = np.minimum(ecog,data_range_max[ds][file])
			if bad_channels[ds][file] is not None:
				ecog = np.delete(ecog,bad_channels[ds][file],1)
			ecog = signal.decimate(ecog,32,ftype='fir',axis=0)
			samples_for_statics = ecog[statics_samples_ecog[ds][file*2]:statics_samples_ecog[ds][file*2+1]]
			statics_ecog = samples_for_statics.mean(axis=0,keepdims=True), np.sqrt(samples_for_statics.var(axis=0, keepdims=True))
			ecog = (ecog - statics_ecog[0])/statics_ecog[1]
			ecog_len_+= [ecog.shape[0]]
			ecog_+=[ecog]
		ecog_alldataset+= [np.concatenate(ecog_,axis=0)]
		ecog_len+=[ecog_len_]


	for ds in range(len(dataset_filename_spkr)):
		spkr_=[]
		for file in range(len(dataset_filename_spkr[ds])):
			spkrdata = h5py.File(DATA_DIR+dataset_filename_spkr[ds][file],'r')
			spkr = np.asarray(spkrdata['TF32']['TFlog'])
			if ds ==0 and file ==0:
				samples_for_statics = spkr[statics_samples_spkr[0][0*2]:statics_samples_spkr[0][0*2+1]]
			# samples_for_statics = spkr[statics_samples_spkr[ds][file*2]:statics_samples_spkr[ds][file*2+1]]
			statics_spkr = samples_for_statics.mean(axis=0,keepdims=True), np.sqrt(samples_for_statics.var(axis=0, keepdims=True))
			spkr = (spkr - statics_spkr[0])/statics_spkr[1]
			spkr_+=[spkr]
		spkr_alldataset +=[np.concatenate(spkr_,axis=0)]

	for ds in range(len(event_filename)):
		start_ind_=[]
		bias_cum = 0
		for file in range(len(event_filename[ds])):
			start_ind = scipy.io.loadmat(DATA_DIR+event_filename[ds][file])['Events']['onset'][0]
			start_ind = np.asarray([start_ind[i][0,0] for i in range(start_ind.shape[0])])[:event_range[ds][file]]
			start_ind = start_ind//256 + start_ind_bias[ds][file] + np.cumsum(ecog_len[ds])[file]
			start_ind_ += [start_ind]
		start_ind_alldataset += [np.concatenate(start_ind_,axis=0)]

	unique_labels = []
	for ds in range(len(event_filename)):
		word_=[]
		for file in range(len(event_filename[ds])):
			label_mat = scipy.io.loadmat(DATA_DIR+event_filename[ds][file])['Events']['word'][0][:event_range[ds][file]]
			labels = []
			for i in range(label_mat.shape[0]):
				labels.append(label_mat[i][0])
				if label_mat[i][0] not in unique_labels:
					unique_labels.append(label_mat[i][0])
			label_ind = np.zeros([label_mat.shape[0]])
			for i in range(label_mat.shape[0]):
				label_ind[i] = unique_labels.index(labels[i])
			label_ind = np.asarray(label_ind,dtype=np.int16)
			word_+=[label_ind]
		word_alldataset += [np.concatenate(word_,axis=0)]

	return 	ecog_alldataset, spkr_alldataset, start_ind_alldataset, word_alldataset

def get_batch(ecog_alldataset, spkr_alldataset,start_ind_alldataset,word_alldataset, mode, num_words, threshold = 1.0, t_scale = 1, seg_length=160, ):

	n_delay_1 = 64/4#28 # samples
	n_delay_2 = 128/4#92#120#92 # samples 
	num_dataset = len(ecog_alldataset)
	if mode =='train':
		random_words = np.random.choice(np.unique(word_alldataset[0]),num_words,replace=False)
		dataset_selected_words = np.zeros([num_dataset,num_words],dtype = np.int32)

		word_id = np.tile(random_words,num_dataset)

		for i in range(num_dataset):
			for j,w in enumerate(random_words):
				good_sample_mask = np.zeros(len(word_alldataset[i][:-test_num_perset]))==0 # last 50 words for testing
				good_sample_mask[bad_samples[i]]=False
				dataset_selected_words[i,j] = np.random.choice(np.where(word_alldataset[i][:-test_num_perset]==w * good_sample_mask)[0],1) #the last 50 words of each dataset is for testing

		ecog_batch_all = []
		spkr_batch_all = []
		for i in range(num_dataset):
			ecog_batch = np.zeros((num_words,seg_length+n_delay_2-n_delay_1 ,ecog_alldataset[i].shape[-1]))
			spkr_batch = np.zeros((num_words, seg_length,spkr_alldataset[i].shape[-1]))
			# indx = np.maximum(start_ind[ind]/256 - np.random.choice(64,batch_size),0)
			indx = np.maximum(start_ind_alldataset[i][dataset_selected_words[i]] - np.random.choice(64,num_words),0)
			for j in range(num_words):
				ecog_batch[j] = ecog_alldataset[i][indx[j]+n_delay_1:indx[j]+seg_length+n_delay_2]
				spkr_batch[j] = spkr_alldataset[i][indx[j]:indx[j]+seg_length]
			ecog_batch_all += [ecog_batch]
			spkr_batch_all += [spkr_batch]

	if mode == 'test':
		# word_id=[]
		# for i in range(num_dataset):
		# 	word_id += [word_alldataset[i][-50:]]
		# word_id = np.concatenate(word_id,axis=0)
		word_id = np.tile(np.arange(test_num_perset),num_dataset)
		ecog_batch_all = []
		spkr_batch_all = []
		for i in range(num_dataset):
			ecog_batch = np.zeros((test_num_perset, seg_length+n_delay_2-n_delay_1 ,ecog_alldataset[i].shape[1]))
			spkr_batch = np.zeros((test_num_perset, seg_length,32))
			order = np.argsort(word_alldataset[i][-test_num_perset:])
			for j in range(test_num_perset):
				indx = start_ind_alldataset[i][-test_num_perset:][order][j]-32
				ecog_batch[j] = ecog_alldataset[i][indx + n_delay_1: indx+seg_length+n_delay_2]
				spkr_batch[j] = spkr_alldataset[i][indx : indx+seg_length]
			ecog_batch_all += [ecog_batch]
			spkr_batch_all += [spkr_batch]


	spkr_batch_all = np.concatenate(spkr_batch_all,axis=0)

	return ecog_batch_all, spkr_batch_all, word_id
