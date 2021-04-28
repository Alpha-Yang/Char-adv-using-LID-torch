import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import get_data
from scipy.spatial.distance import pdist, cdist, squareform
from model import AutoEncoderUnet
import argparse

print("--------------PyTorch VERSION:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("..............device", device)

parser = argparse.ArgumentParser(description="CHARLID")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=100, help='batch size for training')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--h', type=int, default=28, help='height of input images')
parser.add_argument('--w', type=int, default=28, help='width of input images')
parser.add_argument('--c', type=int, default=1, help='channel of input images')
parser.add_argument('--version', type=int, default=0, help='experiment version')
args = parser.parse_args()

torch.manual_seed(2021)

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
torch.set_printoptions(precision=8)
dataset = "mnist"

print('Data set: %s' % dataset)
trainDataset, testDataset = get_data()
train_loader = torch.utils.data.DataLoader(
    dataset=trainDataset,
    batch_size=args.batch_size, 
    shuffle=False,
)
test_loader = torch.utils.data.DataLoader(
    dataset=testDataset,
    batch_size=args.batch_size, 
    shuffle=False,
)

model = AutoEncoderUnet(args.c)
model = model.to(device)

# Report the training process
log_dir = os.path.join(args.exp_dir, 'version_%d' % args.version)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log_gpus.txt'),'w')
sys.stdout= f

for arg in vars(args):
    print(arg, getattr(args, arg))
print("This is a version extracting LID which doesnt need transform GPU to CPU.")
print("-" * 50)

import time
start = time.time()

# lid of a batch of query points X
def mle_batch(data, batch, k):
    k = min(k, len(data)-1)
    f = lambda v: - k / torch.sum(torch.log(v/v[-1]))
    a = torch.cdist(batch, data)
    a, _ = torch.sort(a, dim=1)
    a = a[:,1:k+1]
    a = torch.stack([f(x) for x in torch.unbind(a, dim=0)])
    return a

def estimate(i_batch, lid_dim, n_feed, funcs):
    lid_batch = torch.zeros(n_feed, lid_dim, dtype=torch.float64)
    for i, func in enumerate(funcs):
        X_act = func.reshape(n_feed, -1)
        # print("X_act: ", X_act.shape)

        # random clean samples
        # Maximum likelihood estimation of local intrinsic dimensionality (LID)
        lid_batch[:, i] = mle_batch(X_act, X_act, k=50)
        # print("lid_batch: ", lid_batch.shape)
    return lid_batch

for batch_idx, (x, y) in enumerate(test_loader):
    print("LID of batch %d in testing: " % (batch_idx+1))
    x = x.to(device)
    y_funcs = model(x)
    n_feed = x.shape[0]
    lid_dim = len(y_funcs)
    lid_batch = estimate(batch_idx, lid_dim, n_feed, y_funcs)
    print(lid_batch)
    print("The shape of batch %d is: " % (batch_idx+1), lid_batch.shape)
    print("-" * 50)

end = time.time()
print('Time cost of faster version = %fs' % (end - start))

sys.stdout = orig_stdout
f.close()