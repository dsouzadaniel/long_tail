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
# import config, loaders, classes
import config, classes

if config.REPRODUCIBLE:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

#####################################################
# Libraries
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from models import wide_resnet
device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE -> {0}".format(device))

#####################################################
# Settings
# TRAIN_DATASET = 'cifar100'
# TRAIN_DATASET = 'N20_A20_T60'
TRAIN_DATASET = 'N20_A20_TX2'

MSP_AUG_PCT = 0.2
#####################################################

ADD_AUG_COPIES = 0
TGT_AUG_EPOCH_AFTER = 4

assert  0<=MSP_AUG_PCT<=1, "MSP_AUG_PCT must be between 0 and 1"

_using_longtail_dataset = False if TRAIN_DATASET == 'cifar100' else True

EXP_NAME = 'aug_msp_{0}'.format(MSP_AUG_PCT)
WRITE_FOLDER = os.path.join("C100_{0}_{1}".format(seed_value, TRAIN_DATASET), EXP_NAME)

# Folder to collect epoch snapshots
if not os.path.exists(WRITE_FOLDER):
    os.makedirs(name=WRITE_FOLDER)

if not _using_longtail_dataset:
    print("{0}_Using Original({1}) Dataset_{0}".format("*" * 50, TRAIN_DATASET))
    orig_trainset = classes.CIFAR100(apply_augmentation=False)
else:
    print("{0}_Using LongTail({1}) Dataset_{0}".format("*" * 50, TRAIN_DATASET))
    _train_npz = os.path.join(config.DATASET_FOLDER, 'LONGTAIL_CIFAR100', TRAIN_DATASET + '.npz')
    orig_trainset = classes.LONGTAIL_CIFAR100(dataset_npz=_train_npz, apply_augmentation=False)

print(orig_trainset)

#  Initialize to all 1s to augment the entire dataset
to_augment_next_epoch = np.ones(shape=(len(orig_trainset)))

# For No Augmentation, set below variables accordingly
if MSP_AUG_PCT==0:
    to_augment_next_epoch = np.zeros(shape=(len(orig_trainset)))

print("\n","*"*100)
print("Augmenting the Bottom {0}% MSP with {1} Additional Copies starting after Epoch {2}".format(int(MSP_AUG_PCT*100), ADD_AUG_COPIES, TGT_AUG_EPOCH_AFTER))

print("*"*100,"\n")

orig_trainloader = DataLoader(
    orig_trainset,
    batch_size=config.TRAIN_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    worker_init_fn=np.random.seed(seed_value),
)

testset = classes.CIFAR100_TEST()
testloader = DataLoader(
    testset,
    batch_size=config.TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    worker_init_fn=np.random.seed(seed_value),
)

net = wide_resnet.Wide_ResNet(
    depth=28, widen_factor=10, dropout_rate=0, num_classes=len(config.CLASSES_C100)
)
net = net.to(device)

if device == "cuda":
    net = torch.nn.DataParallel(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(), lr=config.LR, momentum=0.9, weight_decay=5e-4, nesterov=True
)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[40, 50, 55], gamma=0.2
)


# Initialize Prediction Arrays
train_epoch_predictions = np.zeros(shape=(len(orig_trainset)))
test_epoch_predictions = np.zeros(shape=(len(testset)))

# Softmax for Predictions
softmax = torch.nn.Softmax(dim=-1)


def train(epoch):
    train_loss = 0
    correct = 0
    total = 0

    print("EPOCH {0}: Augment 1-hot Sum : {1}".format(epoch, np.sum(to_augment_next_epoch)))

    if not _using_longtail_dataset:
        curr_trainset = classes.CIFAR100_DYNAMIC(augment_indicator=to_augment_next_epoch,
                                                num_additional_copies=0 if epoch <= TGT_AUG_EPOCH_AFTER else ADD_AUG_COPIES)
    else:
        curr_trainset = classes.LONGTAIL_CIFAR100_DYNAMIC(dataset_npz=_train_npz,
                                                         augment_indicator=to_augment_next_epoch,
                                                         num_additional_copies=0 if epoch <= TGT_AUG_EPOCH_AFTER else ADD_AUG_COPIES)

    curr_trainloader = DataLoader(
        curr_trainset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        worker_init_fn=np.random.seed(seed_value),
    )

    # Zero Out Epoch Matrix at Epoch Start
    train_epoch_predictions.fill(0)

    for ixs, inputs, targets in curr_trainloader:
        net.train()
        train_inputs, train_targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(train_inputs)
        loss = criterion(outputs, train_targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += train_targets.size(0)
        correct += predicted.eq(train_targets).sum().item()

        # Write Predictions
        target_softmax_output = softmax(outputs.clone().cpu().detach())[np.arange(len(targets)), targets]
        train_epoch_predictions[ixs[ixs < len(orig_trainset)]] = target_softmax_output[ixs < len(orig_trainset)]

    scheduler.step()
    loss = train_loss / len(orig_trainloader)
    acc = 100.0 * correct / total
    return acc, loss


def test(epoch):
    test_loss = 0
    correct = 0
    total = 0
    class_correct = list(0.0 for i in range(len(config.CLASSES_C100)))
    class_total = list(0.0 for i in range(len(config.CLASSES_C100)))

    # Zero out Preds at Epoch Start
    test_epoch_predictions.fill(0)

    for ixs, inputs, targets in testloader:
        net.eval()
        test_inputs, test_targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()


        outputs = net(test_inputs)
        loss = criterion(outputs, test_targets)
        loss.backward()

        # # Write Predictions
        target_softmax_output = softmax(outputs.clone().cpu().detach())[np.arange(len(targets)), targets]
        test_epoch_predictions[ixs[ixs < len(testset)]] = target_softmax_output[ixs < len(testset)]

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
collect_mtrx_data = []
collect_aupr_data = []

collect_predprob_train_data = {}
collect_predprob_test_data = {}


_track_lr = optimizer.param_groups[0]["lr"]
print("Learning Rate --> {1}".format(_track_lr, optimizer.param_groups[0]["lr"]))
for epoch in tqdm(range(config.EPOCHS)):

    AUGMENT_SCHEDULE = (epoch >= TGT_AUG_EPOCH_AFTER)

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
    collect_mtrx_data.append((train_acc, train_loss, test_acc, test_loss, epoch))

    print(
        "Epoch: {0} | Train_Acc: {1}\tTrain_Loss: {2} | Test_Acc: {3}\tTest_Loss: {4}".format(
            epoch, train_acc, train_loss, test_acc, test_loss
        )
    )

    # Write Predictons
    collect_predprob_train_data['EPOCH_{0}'.format(str(epoch))] = train_epoch_predictions.tolist()
    collect_predprob_test_data['EPOCH_{0}'.format(str(epoch))] = test_epoch_predictions.tolist()

    if AUGMENT_SCHEDULE:
        # Reset the Augment 1-Hot at every epoch
        to_augment_next_epoch.fill(0)

        print("Clearing the Augment 1-hot Sum: {1} ".format(epoch, np.sum(to_augment_next_epoch)))
        # ##################### Choosing using SFMX over the entire dataset #####################

        curr_sfmx_scores = train_epoch_predictions

        _, min_sfmx_ix = torch.topk(
            torch.tensor(curr_sfmx_scores), k=int(len(orig_trainset) * MSP_AUG_PCT), largest=False
        )
        # Prep for AUGMENT in the next epoch
        to_augment_next_epoch[min_sfmx_ix] = 1

        # Additional Information Available if using LongTail Datasets
        if _using_longtail_dataset:

            # AUPR Calculation
            noisy_1hot, atypical_1hot = np.zeros(len(orig_trainset)), np.zeros(len(orig_trainset))
            np.put(a=noisy_1hot, ind=orig_trainset.selected_ixs_for_noisy, v=1)
            np.put(a=atypical_1hot, ind=orig_trainset.selected_ixs_for_atypical, v=1)

            assert len(orig_trainset.selected_ixs_for_noisy) == sum(
                noisy_1hot), "Noisy 1 Hot is not equal to num of noisy"
            assert len(orig_trainset.selected_ixs_for_atypical) == sum(
                atypical_1hot), "Atypical 1 Hot is not equal to num of atypical"

            # AUPR Data
            noisy_aupr_random = average_precision_score(y_true=noisy_1hot, y_score=np.random.rand(len(orig_trainset)))
            atypical_aupr_random = average_precision_score(y_true=atypical_1hot,
                                                           y_score=np.random.rand(len(orig_trainset)))

            noisy_aupr_sfmx = average_precision_score(y_true=noisy_1hot, y_score=-train_epoch_predictions)
            atypical_aupr_sfmx = average_precision_score(y_true=atypical_1hot, y_score=-train_epoch_predictions)

            collect_aupr_data.append((noisy_aupr_random, atypical_aupr_random, noisy_aupr_sfmx, atypical_aupr_sfmx,epoch))

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

# Write Files
mtrx_df.to_csv(os.path.join(WRITE_FOLDER, "metrics.csv"), index=False)
collect_predprob_train_data_df.to_csv(os.path.join(WRITE_FOLDER, "train_predprob.csv"), index=False)
collect_predprob_test_data_df.to_csv(os.path.join(WRITE_FOLDER, "test_predprob.csv"), index=False)


# Write Additional Files( if using LongTail dataset)
if _using_longtail_dataset:

    aupr_df = pd.DataFrame(
        data=collect_aupr_data,
        columns=[
            "noisy_aupr_random",
            "atypical_aupr_random",
            "noisy_aupr_sfmx",
            "atypical_aupr_sfmx",
            "recorded_at_epoch",
        ],
    )

    # Write Files
    aupr_df.to_csv(os.path.join(WRITE_FOLDER, "aupr.csv"), index=False)
