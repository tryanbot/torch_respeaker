import torch
from torch import nn
from model import KeyWordSpotter
import os
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav

import time

path_to_weights_file = "/home/respeaker/inference_wav/weights/epoch_999_0.176075_0.803333.weights"
path_to_wav_file = "../demo.wav"
path_to_mfcc_file = "../demo.mfcc.npy"


def turn_waves_to_mfcc(path_to_waves, numcep=20):
	print()
	wave_path = path_to_waves
	mfcc_path = path_to_mfcc_file
	(rate, sig) = wav.read(wave_path)
	if len(sig) > 0:
		mfcc_feat = mfcc(sig, rate, numcep=numcep)
		np.save(mfcc_path, mfcc_feat)
	return mfcc_path

def load_model(model_path):
	model = KeyWordSpotter(20)
	model.load_state_dict(torch.load(model_path,map_location={'cuda:0': 'cpu'}), strict=False)
	model.eval()
	return model

model = load_model(path_to_weights_file)


def load_mfcc(input_mfcc_path):
	x_np_raw = np.load(input_mfcc_path)
	x_np = padd_concat_mfccs(x_np_raw)
	if(x_np.shape[1]<=250):
		x_np_zeropadded = np.zeros((1,250,20))
		x_np_zeropadded[:,:x_np.shape[1],:] = x_np
	else:
		x_np_zeropadded = x_np
	x = torch.from_numpy(x_np_zeropadded).float()
	
	return x

def padd_concat_mfccs(mfcc):
	max_length = mfcc.shape[0]
	mfccs_array = np.zeros((1, max_length, mfcc.shape[-1]), dtype=np.float32)
	mfcc = np.expand_dims(mfcc, 0)
	mfccs_array[0, :mfcc.shape[1], :mfcc.shape[2]] = mfcc

	return mfccs_array


def test_inference(model,wave_path):
	turn_waves_to_mfcc(wave_path,20)
	flag_keyword = inference(model,path_to_mfcc_file)

def Average(lst): 
    return sum(lst) / len(lst) 

def inference_berulang(kali_percobaan,mfcc_path):
	
	times = []
	for i in range(0,kali_percobaan):
		tic = time.clock()
		mfcc = load_mfcc(mfcc_path)
		out = model(mfcc)
		toc = time.clock()
		times.append(toc-tic)
	return Average(times)


def is_keyword(mfcc_path):
	tic = time.clock()
	mfcc = load_mfcc(mfcc_path)
	out = model(mfcc)
	key_word = False
	if out.item() > 0.5:
		key_word = True
	toc = time.clock()
	print("Durasi Deteksi")
	print(toc-tic)
	return key_word
	