import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


class Conv(nn.Module):
    """
    A convolution with the option to be causal and use xavier initialization
    """
    def __init__(self, input_channels, output_channels, kernel_size=1, stride=1,
                 dilation=1, bias=True, init_gain='linear', is_causal=False):
        super(Conv, self).__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(input_channels, output_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    dilation=dilation, bias=bias)

        nn.init.xavier_uniform(
            self.conv.weight, gain=torch.nn.init.calculate_gain(init_gain))

    def forward(self, signal):
        if self.is_causal:
            padding = (int((self.kernel_size - 1) * self.dilation), 0)
            signal = F.pad(signal, padding)
        return  self.conv(signal)


class WaveNetModel(nn.Module):
    """
    Args:
        layers (Int):               Number of layers in each block
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model
    """
    def __init__(self, input_channels, layers, max_dilation,
                 residual_channels, skip_channels, output_channels,
                 cond_channels, upsample_window, upsample_stride):


        super(WaveNetModel, self).__init__()
        self.upsampel = nn.ConvTranspose1d(cond_channels,
                                           cond_channels,
                                           upsample_window,
                                           upsample_stride)
        self.layers = layers
        self.max_dilation = max_dilation
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.output_channels = output_channels
        self.cond_layers = Conv(cond_channels, 2*residual_channels*layers,
                                init_gain='tanh')
        self.dilate_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.embed = nn.Embedding(input_channels,residual_channels)
        self.conv_out = Conv(skip_channels, output_channels,
                             bias=False, init_gain='relu')
        self.conv_end = Conv(output_channels, output_channels,
                             bias=False, init_gain='linear')
        dilation_factor = math.floor(math.log2(max_dilation)) + 1
        for layer in range(layers):
            dilation = 2 ** dilation_factor

            self.dilate_layers.append(Conv(residual_channels, 2*residual_channels,
                             kernel_size=2, dilation= dilation,
                             init_gain='tanh', is_causal=True))

            if layer < layers - 1:
                self.residual_layers.append(Conv(residual_channels, residual_channels,
                                                 init_gain='linear'))

            self.skip_layers.append(Conv(residual_channels, skip_channels,
                                         init_gain='relu'))

    def forward(self, forward_input):
        a = forward_input[0]
        # forward_input = a.transpose(0,1)
        forward_input = self.embed(a.long())
        forward_input = forward_input.transpose(1,2)

        for layer in range(self.layers):
            in_act = self.dilate_layers[layer](forward_input)
            t_act = F.tanh(in_act[:,:self.residual_channels,:])
            s_act = F.sigmoid(in_act[:,:self.residual_channels,:])
            act = t_act * s_act
            if layer < len(self.residual_layers):
                res_acts = self.residual_layers[layer](act)
            forward_input = res_acts + forward_input

            if layer == 0:
                output = self.skip_layers[layer](act)
            else:
                output = output + self.skip_layers[layer](act)

        output = F.relu(output, True)
        output = self.conv_out(output)
        output = F.relu(output, True)
        output = self.conv_end(output)
        output = F.softmax(output,dim=1)
        b = output.argmax(dim=1)
        # remove last probability
        #last = output[:, :, -1]
        #last = last.unsqueeze(2)
        #output = output[:, :, :-1]

        # repalce first probablity with equal probabilities
        #first = last * 0.0 + 1/self.output_channels
        #output = torch.cat((first, output), dim=2)

        return output


    def export_weights(self):
        """
        Returns a dictionary with tensors ready for load
        """
        model = {}
        # We're not using a convolution to start to this does nothing
        model["embedding_prev"] = torch.cuda.FloatTensor(self.output_channels,
                                                         self.residual_channels).fill_(0.0)

        model["embedding_curr"] = self.embed.weight.data
        model["conv_out_weight"] = self.conv_out.conv.weight.data
        model["conv_end_weight"] = self.conv_end.conv.weight.data

        dilate_weights = []
        dilate_biases = []
        for layer in self.dilate_layers:
            dilate_weights.append(layer.conv.weight.data)
            dilate_biases.append(layer.conv.bias.data)
        model["dilate_weights"] = dilate_weights
        model["dilate_biases"] = dilate_biases

        model["max_dilation"] = self.max_dilation

        residual_weights = []
        residual_biases = []
        for layer in self.res_layers:
            residual_weights.append(layer.conv.weight.data)
            residual_biases.append(layer.conv.bias.data)
        model["res_weights"] = residual_weights
        model["res_biases"] = residual_biases

        skip_weights = []
        skip_biases = []
        for layer in self.skip_layers:
            skip_weights.append(layer.conv.weight.data)
            skip_biases.append(layer.conv.bias.data)
        model["skip_weights"] = skip_weights
        model["skip_biases"] = skip_biases

        model["use_embed_tanh"] = False

        return model

