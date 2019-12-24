import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def _split_cols(mat, lengths):
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results

class NTMHead(nn.Module):
    def __init__(self, memory, controller_size):
        super(NTMHead, self).__init__()
        self.memory = memory
        self.controller_size = controller_size
        self.N, self.M = memory.size()

    def _address_memory(self, k, beta, g, s, gamma, w_prev):
        k = k.clone()
        beta = F.softplus(beta)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        gamma = 1 + F.softplus(gamma)

        w = self.memory.address(k, beta, g, s, gamma, w_prev)

        return w

class NTMReadHead(NTMHead):
    def __init__(self, memory, controller_size):
        super(NTMReadHead,self).__init__(memory, controller_size)

        self.read_length = [self.M, 1, 1, 3, 1]
        self.fc_read = nn.Linear(controller_size, sum(self.read_length))
        self.reset_parameters()

    def is_read_head(self):
        return True

    def create_new_state(self, batch_size):
        # The state holds the previous time step address weightings
        return torch.zeros(batch_size, self.N)

    def forward(self, embeddings, w_prev):
        o = self.fc_read(embeddings)
        k, beta, g, s, gamma = _split_cols(o, self.read_length)

        w = self._address_memory(k, beta, g, s, gamma, w_prev)
        r = self.memory.read(w)

        return r, w

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

class NTMWriteHead(NTMHead):
    def __init__(self, memory, controller_size):
        super(NTMWriteHead,self).__init__(memory, controller_size)

        self.write_length = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.fc_write = nn.Linear(controller_size, sum(self.write_length))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    def create_new_state(self, batch_size):
        return torch.zeros(batch_size, self.N)

    def forward(self, embeddings, w_prev):
        o = self.fc_write(embeddings)
        k, beta, g, s, gamma, e, a = _split_cols(o, self.write_length)
        e = F.sigmoid(e)
        w = self._address_memory(k, beta, g, s, gamma, w_prev)
        self.memory.write(w,e,a)

        return w




