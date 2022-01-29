######### Setting Seeds for Reproducibility #########

# Set a seed value
seed_value = 789
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ["PYTHONHASHSEED"] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)
# 4. Set Torch seed at a fixed value
import torch

torch.manual_seed(seed_value)

# 5. Set TF seed at a fixed value
import tensorflow as tf

tf.random.set_seed(seed_value)

# 6. CuDNN settings
import config, classes

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#####################################################

# Libraries

import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import wide_resnet

device = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE -> {0}".format(device))

print("==> Preparing data..")

import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', help='training batch size')
parser.add_argument('--epochs', help='epochs to train for')
args = parser.parse_args()

BATCH_SIZE = int(args.batch_size)
print("Using Batch Size : {0}".format(BATCH_SIZE))

EPOCHS = int(args.epochs)
print("Using Epochs : {0}".format(EPOCHS))

TRAIN_DATASET = 'cifar100'

WRITE_FOLDER = os.path.join("{0}_{1}_BS{2}".format(seed_value, TRAIN_DATASET, BATCH_SIZE))

# Folder to collect epoch snapshots
if not os.path.exists(WRITE_FOLDER):
    os.makedirs(name=WRITE_FOLDER)

trainset = classes.CIFAR100(apply_augmentation=False, apply_transform=True)
print(trainset)

trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    worker_init_fn=np.random.seed(seed_value),
)

testset = classes.CIFAR100_TEST()
testloader = DataLoader(
    testset,
    batch_size=256,
    shuffle=False,
    num_workers=2,
    worker_init_fn=np.random.seed(seed_value),
)

# net = resnet.ResNet18()
net = wide_resnet.Wide_ResNet(
    depth=28, widen_factor=10, dropout_rate=0, num_classes=100
)
net = net.to(device)

if device == "cuda":
    net = torch.nn.DataParallel(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True
)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[40, 50, 55], gamma=0.2
)

# Epoch Gradient Matrix
epoch_gradient_snapshot = np.zeros(
    shape=(len(trainset), 32 * 32)
)
collect_gradient_snapshots = np.zeros(
    shape=(len(trainset), 32 * 32, 3)
)

epoch_predictions = np.zeros(shape=(len(trainset)))

test_epoch_gradient_snapshot = np.zeros(
    shape=(len(testset), 32 * 32)
)
test_epoch_predictions = np.zeros(shape=(len(testset)))

# Softmax for Predictions
softmax = torch.nn.Softmax(dim=-1)


def train(epoch):
    # net.train()
    train_loss = 0
    correct = 0
    total = 0

    # Zero Out Epoch Matrix at Epoch Start
    epoch_gradient_snapshot.fill(0)
    epoch_predictions.fill(0)

    for ixs, inputs, targets in trainloader:

        net.train()
        train_inputs, train_targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Temporary Enable Grad for Input
        train_inputs.requires_grad = True

        outputs = net(train_inputs)
        loss = criterion(outputs, train_targets)
        loss.backward()

        # Collect Grads for Batch & Write to Epoch Matx
        curr_batch_grads = train_inputs.grad.clone().cpu().detach()
        curr_batch_grads = torch.mean(curr_batch_grads, dim=1).reshape(
            curr_batch_grads.shape[0], 32 * 32
        )
        # Capture gradients of images
        epoch_gradient_snapshot[ixs, :] = curr_batch_grads

        # Zero out Input Grads before Stepping
        train_inputs.grad.zero_()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += train_targets.size(0)
        correct += predicted.eq(train_targets).sum().item()

        # Write Predictions
        target_softmax_output = softmax(outputs.clone().cpu().detach())[np.arange(len(targets)), targets]
        epoch_predictions[ixs[ixs < len(trainset)]] = target_softmax_output[ixs < len(trainset)]

    scheduler.step()
    loss = train_loss / len(trainloader)
    acc = 100.0 * correct / total
    return acc, loss


def test(epoch):
    test_loss = 0
    correct = 0
    total = 0
    class_correct = list(0.0 for i in range(100))
    class_total = list(0.0 for i in range(100))
    # Zero Out Epoch Matrix at Epoch Start
    test_epoch_gradient_snapshot.fill(0)
    # Zero out Preds at Epoch Start
    test_epoch_predictions.fill(0)

    for ixs, inputs, targets in testloader:
        net.eval()
        test_inputs, test_targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # Temporary Enable Grad for Input
        test_inputs.requires_grad = True

        outputs = net(test_inputs)
        loss = criterion(outputs, test_targets)
        loss.backward()

        # Collect Grads for Batch & Write to Epoch Matx
        curr_batch_grads = test_inputs.grad.clone().cpu().detach()
        curr_batch_grads = torch.mean(curr_batch_grads, dim=1).reshape(
            curr_batch_grads.shape[0], 32 * 32
        )
        test_epoch_gradient_snapshot[ixs, :] = curr_batch_grads

        # # Write Predictions
        target_softmax_output = softmax(outputs.clone().cpu().detach())[np.arange(len(targets)), targets]
        test_epoch_predictions[ixs[ixs < len(testset)]] = target_softmax_output[ixs < len(testset)]

        # Zero out Input Grads before Stepping
        test_inputs.grad.zero_()

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += test_targets.size(0)
        correct += predicted.eq(test_targets).sum().item()
        c = predicted.eq(test_targets).squeeze()
        for bb in range(test_targets.shape[0]):
            label = test_targets[bb]
            class_correct[label] += c[bb].item()
            class_total[label] += 1

    loss = test_loss / len(testloader)
    acc = 100.0 * correct / total
    return acc, loss


# Main Training Loop
collect_mtrx_data = []  # For ze Metrics
collect_predprob_train_data = {}
collect_predprob_test_data = {}
collect_rawvog_data = {}


_track_lr = optimizer.param_groups[0]["lr"]
print("Learning Rate --> {1}".format(_track_lr, optimizer.param_groups[0]["lr"]))
for epoch in tqdm(range(EPOCHS)):
    # Check for LR Changes
    if _track_lr != optimizer.param_groups[0]["lr"]:
        print(
            "Learning Rate updated from {0} --> {1}".format(
                _track_lr, optimizer.param_groups[0]["lr"]
            )
        )
        _track_lr = optimizer.param_groups[0]["lr"]

    train_acc, train_loss = train(epoch)
    test_acc, test_loss = test(epoch)
    # Write Metrics
    collect_mtrx_data.append((train_acc, train_loss, test_acc, test_loss, epoch))
    # Write Predictons
    collect_predprob_train_data['EPOCH_{0}'.format(str(epoch))] = epoch_predictions.tolist()
    collect_predprob_test_data['EPOCH_{0}'.format(str(epoch))] = test_epoch_predictions.tolist()

    print(
        "Epoch: {0} | Train_Acc: {1}\tTrain_Loss: {2} | Test_Acc: {3}\tTest_Loss: {4}".format(
            epoch, train_acc, train_loss, test_acc, test_loss
        )
    )
    # Rolling Buffer to Collect Gradient Snapshots
    collect_gradient_snapshots = np.concatenate(
        [
            collect_gradient_snapshots[:, :, 1:],
            np.expand_dims(epoch_gradient_snapshot, axis=-1),
        ],
        axis=-1,
    )

    if epoch >= 3:
        print("************ Calculating VOG at EPOCH {0} ************".format(epoch))
    # Calculate VOG
        VOG = np.mean(
            np.sqrt(
                np.mean(
                    np.square(
                        collect_gradient_snapshots
                        - np.mean(collect_gradient_snapshots, axis=-1, keepdims=True)
                    ),
                    axis=-1,
                )
            ),
            axis=-1,
        )

        collect_rawvog_data['EPOCH_{0}'.format(str(epoch))] = VOG.tolist()

        # Z-Norm by Class
        VOG_Z = np.zeros_like(VOG)
        for c in range(100):
            curr_class_ixs = trainset.class_ixs[c]

            curr_vog_scores = VOG[curr_class_ixs]
            VOG_Z[curr_class_ixs] = (
                                            curr_vog_scores - curr_vog_scores.mean()
                                    ) / curr_vog_scores.std()
        print(VOG_Z.shape)


# Write Metric Files
mtrx_df = pd.DataFrame(
    data=collect_mtrx_data,
    columns=[
        "train_accuracy",
        "train_loss",
        "test_accuracy",
        "test_loss",
        "recorded_at_epoch",
    ],
)


collect_predprob_train_data_df = pd.DataFrame.from_dict(collect_predprob_train_data)
collect_predprob_test_data_df = pd.DataFrame.from_dict(collect_predprob_test_data)
collect_rawvog_df = pd.DataFrame.from_dict(collect_rawvog_data)

# Write Files
mtrx_df.to_csv(os.path.join(WRITE_FOLDER, "mtrx_df.csv"), index=False)
collect_rawvog_df.to_csv(os.path.join(WRITE_FOLDER, "collect_rawvog_df.csv"), index=False)
collect_predprob_train_data_df.to_csv(os.path.join(WRITE_FOLDER, "collect_predprob_train_df.csv"), index=False)
collect_predprob_test_data_df.to_csv(os.path.join(WRITE_FOLDER, "collect_predprob_test_df.csv"), index=False)
