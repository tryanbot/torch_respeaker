import torch
from torch import nn
from model import KeyWordSpotter
#import glob
import os
import numpy as np
from python_speech_features import mfcc
#import progressbar
import scipy.io.wavfile as wav 

import time

path_to_weights_file = "./weights/epoch_999_0.176075_0.803333.weights"
path_to_wav_file = "../demo2_true.wav"
path_to_mfcc_file = "demo.mfcc.npy"
call_func = 0
in_func = 0
def turn_waves_to_mfcc(path_to_waves, numcep=20):
	#path_to_mfcc = "./demo.mfcc"
	wave_path = path_to_waves
	mfcc_path = path_to_mfcc_file
	(rate, sig) = wav.read(wave_path)
	if len(sig) > 0:
		mfcc_feat = mfcc(sig, rate, numcep=numcep)
		return mfcc_feat
		
def load_model(model_path):
	model = KeyWordSpotter(20)
	model.load_state_dict(torch.load(model_path,map_location={'cuda:0': 'cpu'}), strict=False)
	model.eval()
	return model

def inference(model,mfcc_feat):
  in_func = time.clock()
  print('length to call function : ',str(in_func - call_func))
  with torch.no_grad():
    x_np = padd_concat_mfccs(mfcc_feat)
    if x_np.shape[1]<250:
      x_np_zeropadded = np.zeros((1,250,20))
      x_np_zeropadded[:,:x_np.shape[1],:] = x_np
      x = torch.from_numpy(x_np_zeropadded).float()
    else:
      x = torch.from_numpy(x_np).float()
    tic = time.clock()
    out = model(x)
    toc = time.clock()
    print('length to run model : ',str(toc-tic))
    #print(out.numpy()[0])
    key_word = False
    if out.numpy()[0] > 0.5:
      key_word = True
    return key_word


def padd_concat_mfccs(mfcc):
	max_length = mfcc.shape[0]
	mfccs_array = np.zeros((1, max_length, mfcc.shape[-1]), dtype=np.float32)
	mfcc = np.expand_dims(mfcc, 0)
	mfccs_array[0, :mfcc.shape[1], :mfcc.shape[2]] = mfcc

	return mfccs_array


def test_inference(model,wave_path):
	mfcc_features = turn_waves_to_mfcc(wave_path,20)
	flag_keyword = inference(model,mfcc_features)
	print (flag_keyword)


def Average(lst): 
    return sum(lst) / len(lst) 

model = load_model(path_to_weights_file)

# kali_percobaan = 10
# times = []
# for i in range(0,kali_percobaan):
  # tic = time.clock()
  # call_func = time.clock()
  # test_inference(model,path_to_wav_file)
  # toc = time.clock()
  # times.append(toc-tic)

kali_percobaan = 10
times = []
for i in range(0,kali_percobaan):
  mfcc_features = turn_waves_to_mfcc(path_to_wav_file,20)
  x_np = padd_concat_mfccs(mfcc_features)
  if x_np.shape[1]<250:
    x_np_zeropadded = np.zeros((1,250,20))
    x_np_zeropadded[:,:x_np.shape[1],:] = x_np
    x = torch.from_numpy(x_np_zeropadded).float()
  else:
    x = torch.from_numpy(x_np).float()
  tic = time.clock()
  out = model(x)
  toc = time.clock()
  times.append(toc-tic)


wakturata= Average(times)

print (times)
print (wakturata)

 
