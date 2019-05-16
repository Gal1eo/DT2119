import argparse
import json
import os
import random
import torch
import torch.nn.functional as F
import torch.utils.data
import sys
import utils

class OneHot(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_length, mu_quantization,
                 filter_length, sampling_rate):

        self.audio_files = utils.file_to_list(training_files)
        random.seed(123)
        random.shuffle(self.audio_files)
        self.segment_length = segment_length
        self.mu_quantization = mu_quantization
        self.sampling_rate = sampling_rate

    def __getitem__(self, index):
        filename = self.audio_files[index]
        audio, sampling_rate = utils.load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("Sampling rate doesn't math")

        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = F.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        audio = utils.mu_law_encode(audio / utils.MAX_WAV_VALUE, self.mu_quantization)
        return audio

    def __len__(self):
        return len(self.audio_files)














