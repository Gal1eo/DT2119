import argparse
import json
import os
import time
import torch
import numpy as np
import torch.nn as nn

from scipy.io.wavfile import write
from torch.utils.data import DataLoader
from WaveNet import WaveNetModel
from Onehot import OneHot
from utils import *



class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = wavenet_config["output_channels"]

    def forward(self, inputs, targets):
        """
        inputs are batch by num_classes by sample
        targets are batch by sample
        torch CrossEntropyLoss needs
            input = batch * samples by num_classes
            targets = batch * samples
        """
        targets = targets.view(-1)
        inputs = inputs.transpose(1, 2)
        inputs = inputs.contiguous()
        inputs = inputs.view(-1, self.num_classes)
        return nn.CrossEntropyLoss()(inputs, targets)


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    model_for_saving = WaveNetModel(**wavenet_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def train(output_directory, epochs, learning_rate,
          iters_per_checkpoint, batch_size, seed, checkpoint_path):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    criterion = CrossEntropyLoss()
    model = WaveNetModel(**wavenet_config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model,
                                                      optimizer)
        iteration += 1  # next iteration is iteration + 1

    trainset = OneHot(**data_config)
    train_loader = DataLoader(trainset,
                              shuffle=False,
                              num_workers=1,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=False)

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    #epoch_offset = 0
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            model.zero_grad()

            x = batch
            x = x.transpose(1,2)
            x = to_gpu(x).long()
            #y = to_gpu(y)
            #x = (x, y)
            y_pred = model(x)
            loss = criterion(y_pred, x)
            reduced_loss = loss.data
            loss.backward()
            optimizer.step()
            '''
            music = y_pred.argmax(dim=1)
            music = music.transpose(0,1)
            music = mu_law_decode(music, 256) * MAX_WAV_VALUE
            music = music.cpu()
            music = music.numpy()
            scaled = np.int16(music)
            write("a.wav", 44100, scaled)
            '''
            print("{}:\t{:.9f}".format(iteration, reduced_loss))

            if (iteration % iters_per_checkpoint == 0):
                checkpoint_path = "{}/wavenet_{}".format(output_directory, iteration)
                #if not os.path.exists(checkpoint_path):
                #    os.mkdir(checkpoint_path)
                save_checkpoint(model, optimizer, learning_rate, iteration,
                                checkpoint_path)

            iteration += 1
'''
def genetrate(eval_files, checkpoint_path):
    model = WaveNetModel(**wavenet_config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model, optimizer, iteration = load_checkpoint(checkpoint_path, model,
                                                      optimizer)
    model.eval()
    eval_files = os.path.join('dataset', eval_files)
    filename = file_to_list(eval_files)
    filepath = os.path.join('dataset', filename[0])
    audio, sampling_rate = load_wav_to_torch(filepath)

    audio = audio[0:44100]
    input = mu_law_encode(audio / MAX_WAV_VALUE, 256)
    input = to_gpu(input)
    a = input.unsqueeze(0)
    output = model(a)
    generation = output.argmax(dim = 1)
    music = mu_law_decode(generation, 256)

    return music
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    # parser.add_argument('-r', '--rank', type=int, default=0,
    #                   help='rank of process for distributed')
    # parser.add_argument('-g', '--group_name', type=str, default='',
    #                    help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global wavenet_config
    wavenet_config = config["wavenet_config"]
    global eval_config
    eval_config = config["eval_config"]

    #num_gpus = torch.cuda.device_count()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(**train_config)
    #music = genetrate(**eval_config)