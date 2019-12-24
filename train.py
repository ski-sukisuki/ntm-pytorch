import random
import torch
from torch import nn
from torch import optim
import numpy as np
import json
from attr import attrs, attrib, Factory

from ntm.wrapper import EncapsulatedNTM

seed = 10
report_interval = 200
checkpoint_interval = 1000

# parameters
name = "copy-task"
controller_size = 100
controller_layers = 1
num_heads = 1
sequence_width = 8
sequence_min_len = 1
sequence_max_len = 20
memory_n = 128
memory_m = 20
num_batches = 50000
batch_size = 1
rmsprop_lr = 1e-4
rmsprop_momentum = 0.9
rmsprop_alpha = 0.95
checkpoint_path = './copy'

def progress_bar(batch_num, report_interval, last_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num-1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print("\r[{}{}]: {} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), batch_num, last_loss), end='')


def generate_seq(num_batches, batch_size, seq_width, min_len, max_len):
    for batch_num in range(num_batches):
        seq_len = random.randint(min_len, max_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = torch.from_numpy(seq)

        inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0  # delimiter in our control channel
        outp = seq.clone()

        yield batch_num + 1, inp.float(), outp.float()


@attrs
class CopyTaskModelTraining(object):
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        net = EncapsulatedNTM(sequence_width + 1, sequence_width,
                              controller_size, controller_layers,
                              num_heads,
                              memory_n, memory_m)
        return net

    @dataloader.default
    def default_dataloader(self):
        return generate_seq(num_batches, batch_size,
                           sequence_width,
                           sequence_min_len, sequence_max_len)

    @criterion.default
    def default_criterion(self):
        return nn.BCELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=rmsprop_momentum,
                             alpha=rmsprop_alpha,
                             lr=rmsprop_lr)

def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)

def train_batch(net, criterion, optimizer, X, Y):
    optimizer.zero_grad()
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    for i in range(inp_seq_len):
        net(X[i])

    # Read the output (no input given)
    y_out = torch.zeros(Y.size())
    for i in range(outp_seq_len):
        y_out[i], _ = net()

    loss = criterion(y_out, Y)
    loss.backward()
    clip_grads(net)
    optimizer.step()

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    return loss.item(), cost.item() / batch_size


def save_checkpoint(net, batch_num, losses, costs, seq_lengths):

    basename = "{}/{}-{}-batch-{}".format(checkpoint_path, name, seed, batch_num)
    model_fname = basename + ".model"
    torch.save(net.state_dict(), model_fname)

    # Save the training history
    train_fname = basename + ".json"
    content = {
        "loss": losses,
        "cost": costs,
        "seq_lengths": seq_lengths
    }
    open(train_fname, 'wt').write(json.dumps(content))

# def generate_seq(num_batches, batch_size, seq_width, min_len, max_len):
#     for batch_num in range(num_batches):
#         seq_len = random.randint(min_len, max_len)
#         seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
#         seq = torch.from_numpy(seq)
#
#         inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)
#         inp[:seq_len, :, :seq_width] = seq
#         inp[seq_len, :, seq_width] = 1.0  # delimiter in our control channel
#         outp = seq.clone()
#
#         yield batch_num + 1, inp.float(), outp.float()




def train_model(model, num_batches, batch_size):
    losses = []
    costs = []
    seq_lengths = []


    for batch_num, x, y in model.dataloader:
        loss, cost = train_batch(model.net, model.criterion, model.optimizer, x, y)
        losses += [loss]
        costs += [cost]
        seq_lengths += [y.size(0)]
        progress_bar(batch_num, report_interval, loss)

        # # Update the progress bar
        # progress_bar(batch_num, args.report_interval, loss)

        # Report
        # if batch_num % args.report_interval == 0:
        #     mean_loss = np.array(losses[-args.report_interval:]).mean()
        #     mean_cost = np.array(costs[-args.report_interval:]).mean()
        #     mean_time = int(((get_ms() - start_ms) / args.report_interval) / batch_size)
        #     progress_clean()
        #     LOGGER.info("Batch %d Loss: %.6f Cost: %.2f Time: %d ms/sequence",
        #                 batch_num, mean_loss, mean_cost, mean_time)
        #     start_ms = get_ms()

        # Checkpoint
        if (checkpoint_interval != 0) and (batch_num % checkpoint_interval == 0):
            save_checkpoint(model.net,
                            batch_num, losses, costs, seq_lengths)


# def evaluate(net, criterion, X, Y):
#     """Evaluate a single batch (without training)."""
#     inp_seq_len = X.size(0)
#     outp_seq_len, batch_size, _ = Y.size()
#
#     # New sequence
#     net.init_sequence(batch_size)
#
#     # Feed the sequence + delimiter
#     states = []
#     for i in range(inp_seq_len):
#         o, state = net(X[i])
#         states += [state]
#
#     # Read the output (no input given)
#     y_out = torch.zeros(Y.size())
#     for i in range(outp_seq_len):
#         y_out[i], state = net()
#         states += [state]
#
#     loss = criterion(y_out, Y)
#
#     y_out_binarized = y_out.clone().data
#     y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)
#
#     # The cost is the number of error bits per sequence
#     cost = torch.sum(torch.abs(y_out_binarized - Y.data))
#
#     result = {
#         'loss': loss.data[0],
#         'cost': cost / batch_size,
#         'y_out': y_out,
#         'y_out_binarized': y_out_binarized,
#         'states': states
#     }
#
#     return result



def main():
    model = CopyTaskModelTraining()
    train_model(model, num_batches, batch_size)


if __name__ == '__main__':
    main()