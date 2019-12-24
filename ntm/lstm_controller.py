# LSTM Controller for NTM

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional
import numpy as np


class LSTMController(nn.Module):

    def __init__(self, input_size, output_size, controller_size, read_data_size, num_outputs, outp_layer_size):
        super(LSTMController, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = controller_size
        self.num_outputs = output_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=output_size)
        self.output_layer = nn.Linear(outp_layer_size, num_outputs)
        self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        self.h_state = torch.zeros([1, output_size])

    def forward(self, inputs, hidden=None):
        inputs = inputs.unsqueeze(0)
        outp, state = self.lstm(inputs, hidden)
        self.h_state = outp.squeeze(0)
        return self.h_state, state

    def output(self, read_data):
        end_state = read_data
        output = torch.nn.functional.sigmoid(self.output_layer(end_state))
        return output

    def size(self):
        return self.input_size, self.output_size

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

