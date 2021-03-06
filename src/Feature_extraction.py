# from tqdm import tqdm
import os,sys
import numpy as np
import librosa

'''
Created by Shammur
To run the feature extraction for pretrained model on MGB-3 corpus (5 class Arabic DI task by Shon, AAli)
'''


def cmvn_slide(feat ,winlen=300 ,cmvn=False):  # feat : (length, dim) 2d matrix
    # function for Cepstral Mean Variance Nomalization

    maxlen = np.shape(feat)[0]
    new_feat = np.empty_like(feat)
    cur = 1
    leftwin = 0
    rightwin = int(winlen /2)  # modified by shammur

    # middle
    for cur in range(maxlen):
        cur_slide = feat[cur -leftwin:cur +rightwin ,:]
        # cur_slide = feat[cur-winlen/2:cur+winlen/2,:]
        mean =np.mean(cur_slide, axis=0)
        std = np.std(cur_slide, axis=0)
        if cmvn == 'mv':
            new_feat[cur, :] = (feat[cur, :] - mean) / std  # for cmvn
        elif cmvn == 'm':
            new_feat[cur, :] = (feat[cur, :] - mean)  # for cmn
        if leftwin < winlen / 2:
            leftwin += 1
        elif maxlen - cur < winlen / 2:
            rightwin -= 1
    return new_feat


def feat_extract(filelist, feat_type, n_fft_length=512, hop=160, vad=True, cmvn=False, exclude_short=500):
    # function for feature extracting

    feat = []
    utt_shape = []
    new_utt_label = []
    for index, wavname in enumerate(filelist):
        # read audio input
        # print(wavname)
        y, sr = librosa.core.load(wavname, sr=16000, mono=True, dtype='float')

        # extract feature
        if feat_type == 'melspec':
            Y = librosa.feature.melspectrogram(y, sr, n_fft=n_fft_length, hop_length=hop, n_mels=40, fmin=133,
                                               fmax=6955)
        elif feat_type == 'mfcc':
            Y = librosa.feature.mfcc(y, sr, n_fft=n_fft_length, hop_length=hop, n_mfcc=40, fmin=133, fmax=6955)
        elif feat_type == 'spec':
            Y = np.abs(librosa.core.stft(y, n_fft=n_fft_length, hop_length=hop, win_length=400))
        elif feat_type == 'logspec':
            Y = np.log(np.abs(librosa.core.stft(y, n_fft=n_fft_length, hop_length=hop, win_length=400)))
        elif feat_type == 'logmel':
            Y = np.log(librosa.feature.melspectrogram(y, sr, n_fft=n_fft_length, hop_length=hop, n_mels=40, fmin=133,
                                                      fmax=6955))
            # print(Y.shape)

        Y = Y.transpose()
        # print(Y.shape)

        # Simple VAD based on the energy
        if vad:
            E = librosa.feature.rms(y, frame_length=n_fft_length, hop_length=hop, )
            '''librosa change log for .7.0 >> (Root mean square error (rmse) has been renamed to rms)'''
            # E = librosa.feature.rmse(y, frame_length=n_fft_length,hop_length=hop,)
            threshold = np.mean(E) / 2 * 1.04
            vad_segments = np.nonzero(E > threshold)
            if vad_segments[1].size != 0:
                Y = Y[vad_segments[1], :]

        # exclude short utterance under "exclude_short" value
        print(Y.shape[0], Y.shape)
        if exclude_short == 0 or (Y.shape[0] > exclude_short):
            if cmvn:
                Y = cmvn_slide(Y, 300, cmvn)
            # print('Y',Y.shape)
            feat.append(Y)
            utt_shape.append(np.array(Y.shape))
            # print('ut_shape',len(utt_shape))
            #             new_utt_label.append(utt_label[index])
            sys.stdout.write('%s\r' % index)
            sys.stdout.flush()

        # if index == 0:
        #     break

    tffilename = feat_type + '_fft' + str(n_fft_length) + '_hop' + str(hop)
    if vad:
        tffilename += '_vad'
    if cmvn == 'm':
        tffilename += '_cmn'
    elif cmvn == 'mv':
        tffilename += '_cmvn'
    if exclude_short > 0:
        tffilename += '_exshort' + str(exclude_short)

    return feat, new_utt_label, utt_shape, tffilename  # feat : (length, dim) 2d matrix

