import scipy.io as scipy_io
import h5py
import numpy as np
import random
import librosa
DATA_DIR = '../../neural_decoding_data/adeen/'
# DATA_DIR = './data/'

ECOG_DEPTH = 64
SPKR_DEPTH =1

def normmat(matrix):
	global denomm, subb
	subb = matrix.mean()
	denomm = np.sqrt(matrix.var())
	out = (matrix-subb)/denomm
	return out, denomm, subb

def denormmat(matrix,denomm,subb):
	out = np.arctanh(matrix)+subb
	return out


def read_all_data():
	eps = 7./3-4./3-1
	ecogdata_train = h5py.File(DATA_DIR+'ECoGSpec_Train.mat','r')
	ecogesult_train = np.asarray(ecogdata_train['ECoGSpec_Train'])
	ecogspec_train = np.transpose(ecogesult_train,(2,1,0))
	ecogspec_train = 10*np.log10(np.sqrt(ecogspec_train[:,:,:64]**2+ecogspec_train[:,:,64:]**2)+eps)

	spkdata_train = h5py.File(DATA_DIR+'SpkrSpec_Train.mat','r') 
	spkresult_train = np.asarray(spkdata_train['SpkrSpec_Train'])
	spkspec_train = np.transpose(spkresult_train,(2,1,0))
	spkspec_train = 10*np.log10(np.sqrt(spkspec_train[:,:,0]**2+spkspec_train[:,:,1]**2)+eps)
	spkspec_train = spkspec_train[:,:,np.newaxis]

	ecogdata_test = h5py.File(DATA_DIR+'ECoGSpec_Test.mat','r') 
	ecogesult_test = np.asarray(ecogdata_test['ECoGSpec_Test'])
	ecogspec_test = np.transpose(ecogesult_test,(2,1,0))
	ecogspec_test = 10*np.log10(np.sqrt(ecogspec_test[:,:,:64]**2+ecogspec_test[:,:,64:]**2)+eps)

	spkdata_test = h5py.File(DATA_DIR+'SpkrSpec_Test.mat','r') 
	spkresult_test = np.asarray(spkdata_test['SpkrSpec_Test'])
	spkspec_test = np.transpose(spkresult_test,(2,1,0))
	spkspec_test = 10*np.log10(np.sqrt(spkspec_test[:,:,0]**2+spkspec_test[:,:,1]**2)+eps)
	spkspec_test = spkspec_test[:,:,np.newaxis]

	norm_ecog_train = normmat(ecogspec_train)
	net = norm_ecog_train[0]
	norm_spkr_train = normmat(spkspec_train)
	nst = norm_spkr_train[0]

	norm_ecog_test = normmat(ecogspec_test)
	nett = norm_ecog_test[0]
	norm_spkr_test = normmat(spkspec_test)
	nstt = norm_spkr_test[0]
	return norm_ecog_train,norm_spkr_train,norm_ecog_test,norm_spkr_test

# def normmat(matrix):
# 	global denomm, subb
# 	denomm = matrix.max() - matrix.min()
# 	subb = matrix.min()
# 	mid = (matrix - matrix.min()) / denomm
# 	out = 2 * mid - 1
# 	return out, denomm, subb

# def denormmat(matrix, denomm, subb):
# 	ini = (matrix + 1) / 2
# 	out2 = ini * denomm + subb
# 	return out2






def get_data_train(seg_length, batch_size, net, nst, threshold = 0.003):

	# global seg_index, t_scale, n_delay_1, n_delay_2
	# ecog_batch = []
	# spkr_batch = []
	loop_idx = 0
	t_scale = 1 # ratio of t_speech/t_ecog, currently 26701/26701=1
	n_delay_1 = 10 # samples
	n_delay_2 = 30 # samples

	ecog_batch = np.zeros((1,16, seg_length+n_delay_2-n_delay_1 ,ECOG_DEPTH))
	spkr_batch = np.zeros((1,128,seg_length ,SPKR_DEPTH))

	ini_idx = np.zeros((10))

	while (loop_idx < 10):
		seg_index = random.randint(0, net[0].shape[1] - seg_length - 1 - n_delay_2)
		# print(seg_index)
		spkspec_seg = nst[0][:, t_scale*seg_index : t_scale*(seg_index+seg_length), :]
		seg_energy = librosa.feature.rmse(S = spkspec_seg)
		avg_energy = np.mean(seg_energy)
		if avg_energy > threshold:
			spkr_batch = np.vstack((spkr_batch, spkspec_seg[np.newaxis,:]))
			ecogspec_seg = net[0][:, seg_index+n_delay_1 : seg_index+seg_length+n_delay_2, :]
			ecog_batch = np.vstack((ecog_batch, ecogspec_seg[np.newaxis,:]))
			ini_idx[loop_idx] = seg_index
			loop_idx += 1

	# print(ecog_batch.shape)
	ecog_batch = ecog_batch[1:, :, :, :]
	spkr_batch = spkr_batch[1:, :, :, :]

	deno = net[1]
	sub = net[2]

	return ecog_batch, spkr_batch, deno, sub, ini_idx



def get_data_test(sess, seg_length, batch_size, nett, nstt, threshold = 0.003):

	# global seg_index2, seg_length, t_scale, n_delay_1, n_delay_2
	# ecog_batch = []
	# spkr_batch = []

	loop_idx2 = 0
	t_scale = 1 # ratio of t_speech/t_ecog, currently 26701/26701=1
	n_delay_1 = 10 # samples
	n_delay_2 = 30 # samples

	ecog_batch = np.zeros((1,16, seg_length+n_delay_2-n_delay_1 ,ECOG_DEPTH))
	spkr_batch = np.zeros((1,128,seg_length,SPKR_DEPTH))

	ini_idx2 = np.zeros((10))

	while (loop_idx2 < 10):
		seg_index2 = random.randint(0, nett[0].shape[1] - seg_length - 1 - n_delay_2)
		# print(seg_index2)
		spkspec_seg = nstt[0][:, t_scale*seg_index2 : t_scale*(seg_index2+seg_length), :]
		seg_energy = librosa.feature.rmse(S = spkspec_seg)
		avg_energy = np.mean(seg_energy)
		if avg_energy > threshold:
			spkr_batch = np.vstack((spkr_batch, spkspec_seg[np.newaxis,:]))
			ecogspec_seg = nett[0][:, seg_index2+n_delay_1 : seg_index2+seg_length+n_delay_2, :]
			ecog_batch = np.vstack((ecog_batch, ecogspec_seg[np.newaxis,:]))
			ini_idx2[loop_idx2] = seg_index2
			loop_idx2 += 1

	ecog_batch = ecog_batch[1:, :, :, :]
	spkr_batch = spkr_batch[1:, :, :, :]

	deno = nett[1]
	sub = nett[2]


	return ecog_batch, spkr_batch, deno, sub, ini_idx2

