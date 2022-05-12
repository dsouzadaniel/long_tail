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
import argparse
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from models import wide_resnet

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE -> {0}".format(device))

parser = argparse.ArgumentParser(description='LongTail Training Recipes')
parser.add_argument('--MSP_AUG_PCT',
                    type=float,
                    required=False,
                    default=1.0,
                    help='How much of the bottom MSP % to Augment [Default: 1.0]')
parser.add_argument('--DOWNWEIGHT_PCT',
                    type=float,
                    required=False,
                    default=0.0,
                    help='How much of the bottom MSP % to Downweight')
parser.add_argument('--DOWNWEIGHT_TO',
                    type=float,
                    required=False,
                    default=1.0,
                    help='Downweight the selected data to this value [Default: 1.0]')
parser.add_argument('--RELABEL_PCT',
                    type=float,
                    required=False,
                    default=0.0,
                    help='How much of the bottom MSP % to Relabel [Default: 0.0]')
parser.add_argument('--COPY_PCT',
                    type=float,
                    required=False,
                    default=0.0,
                    help='How much of the bottom MSP % to use for copies [Default: 0.0]')
parser.add_argument('--NUM_COPIES',
                    type=int,
                    required=False,
                    default=0,
                    help='How many copies to create of the COPY_PCT %  [Default: 0.0]')
parser.add_argument('--EPOCHS',
                    type=int,
                    required=False,
                    default=60,
                    help='How many epochs to run for [Default: 60]')

args = parser.parse_args()
print(f"MSP Augment Pct :{args.MSP_AUG_PCT}")
print(f"Downweight {args.DOWNWEIGHT_PCT} To :{args.DOWNWEIGHT_TO}")
print(f"Relabel Pct :{args.RELABEL_PCT}")
print(f"Adding {args.NUM_COPIES} copies of the Bottom Pct :{args.COPY_PCT}")
print(f"Epochs :{args.EPOCHS}")

# Clever XNOR to check critical settings
assert not ((args.DOWNWEIGHT_PCT==0.0)!=(args.DOWNWEIGHT_TO==1.0)), "Your Downweighting Settings don't make sense. Check settings"
assert not ((args.NUM_COPIES==0.0)!=(args.COPY_PCT==0.0)), "Your Copy Settings don't make sense. Check settings"

#####################################################
# Settings
TRAIN_DATASET = 'cifar10'
# TRAIN_DATASET = 'N20_A20_T60'
# TRAIN_DATASET = 'N20_A20_TX2'

#####################################################
# Targeted Augmentation

# At which epochs Targeted Augmentation will be applied
TGT_AUG_EPOCH_START = 1
TGT_AUG_EPOCH_STOP = 3
# Targeted Augmentation
MSP_AUG_PCT = args.MSP_AUG_PCT
COPY_PCT = args.COPY_PCT
NUM_COPIES = args.NUM_COPIES

#####################################################
# Targeted Intervention Settings

# At which epoch Intervention Data will be Recorded & when will it be enacted
INTERVENTION_RECORD_EPOCH = TGT_AUG_EPOCH_STOP
INTERVENTION_ACT_EPOCH = INTERVENTION_RECORD_EPOCH + 1  # ( i.e acting on the very next epoch)
# Specific Interventions
RELABEL_PCT = args.RELABEL_PCT  # Default : 0

DOWNWEIGHT_PCT = args.DOWNWEIGHT_PCT  # Default is amount of Noisy
DOWNWEIGHT_TO = args.DOWNWEIGHT_TO  # Default : 1.0

if DOWNWEIGHT_TO == 1.0 and RELABEL_PCT == 0.0:
    INTERVENTION_STR = f"STANDARD"
elif DOWNWEIGHT_TO == 1.0:
    INTERVENTION_STR = f"RELABEL_{RELABEL_PCT}_AT_{INTERVENTION_RECORD_EPOCH}_EFFECTIVE_FROM_{INTERVENTION_ACT_EPOCH}"
elif RELABEL_PCT == 0.0:
    INTERVENTION_STR = f"DOWNWEIGHT_{DOWNWEIGHT_PCT}_TO_{DOWNWEIGHT_TO}_EFFECTIVE_FROM_{INTERVENTION_ACT_EPOCH}"
else:
    INTERVENTION_STR = f"RELABEL_{RELABEL_PCT}_AT_{INTERVENTION_RECORD_EPOCH}_DOWNWEIGHT_{DOWNWEIGHT_PCT}_TO_{DOWNWEIGHT_TO}_EFFECTIVE_FROM_{INTERVENTION_ACT_EPOCH}"
#####################################################

assert TGT_AUG_EPOCH_STOP >= TGT_AUG_EPOCH_START, "The Target Stop Epoch is smaller than the Start Epoch!"
assert INTERVENTION_RECORD_EPOCH >= TGT_AUG_EPOCH_STOP, "The Intervention Epoch is smaller than the Stop Epoch!"

assert 0 <= MSP_AUG_PCT <= 1, "MSP_AUG_PCT must be between 0 and 1"
assert 0 <= COPY_PCT <= 1, "COPY_PCT must be between 0 and 1"
assert 0 <= RELABEL_PCT <= 1, "RELABEL_PCT must be between 0 and 1"

_using_longtail_dataset = False if TRAIN_DATASET == 'cifar10' else True

print("Relabel PCT : {0}".format(INTERVENTION_STR))

# REWIND_INDICATOR = "aug_1.0_rewind_3_drop_0.2.npy"
REWIND_INDICATOR = "vanilla_aug_1.0_rewind_3_drop_0.2.npy"
REWIND_STR = 'REWIND_3_STD'
REWIND_ACTION = True


EXP_NAME = 'aug_msp_{0}_from_{1}_to_{2}'.format(MSP_AUG_PCT, TGT_AUG_EPOCH_START, TGT_AUG_EPOCH_STOP)
if COPY_PCT!=0.0 or NUM_COPIES!=0.0:
    EXP_NAME = EXP_NAME + f"_with_{NUM_COPIES}_copies_of_{COPY_PCT}"

if REWIND_ACTION:
    EXP_NAME = REWIND_STR+"_"+EXP_NAME

WRITE_FOLDER = os.path.join("RRR_C10_{0}_{1}_{2}".format(seed_value, INTERVENTION_STR, TRAIN_DATASET), EXP_NAME)

# Folder to collect epoch snapshots
if not os.path.exists(WRITE_FOLDER):
    os.makedirs(name=WRITE_FOLDER)

TARGET_PROBS_FOLDER = os.path.join(WRITE_FOLDER, 'target_probs_npz_files')
if not os.path.exists(TARGET_PROBS_FOLDER):
    os.makedirs(name=TARGET_PROBS_FOLDER)

MODEL_PREDS_FOLDER = os.path.join(WRITE_FOLDER, 'model_preds_npz_files')
if not os.path.exists(MODEL_PREDS_FOLDER):
    os.makedirs(name=MODEL_PREDS_FOLDER)

if not _using_longtail_dataset:
    print("{0}_Using Original({1}) Dataset_{0}".format("*" * 50, TRAIN_DATASET))
    orig_trainset = classes.CIFAR10(apply_augmentation=False)
else:
    print("{0}_Using LongTail({1}) Dataset_{0}".format("*" * 50, TRAIN_DATASET))
    _train_npz = os.path.join(config.DATASET_FOLDER, 'LONGTAIL_CIFAR10', TRAIN_DATASET + '.npz')
    orig_trainset = classes.LONGTAIL_CIFAR10(dataset_npz=_train_npz, apply_augmentation=False)

if REWIND_ACTION:
    orig_trainset.filter_dataset(ixs_to_keep=torch.where(torch.tensor(np.load(REWIND_INDICATOR))==1)[0])

print(orig_trainset)

# Initialize for Relabel
# print("Reading Default Labels")
curr_labels = [d[1] for d in orig_trainset.dataset]
# print("Writing Default Labels")
# with open(os.path.join(WRITE_FOLDER, 'LATEST_RELABELS_FOR_DATASET.npy'), 'wb') as f:
#     np.save(f, np.array([c for c in curr_labels], dtype=int))


#  Initialize to all 1s to augment the entire dataset
to_augment_next_epoch = np.ones(shape=(len(orig_trainset)))
to_copy_next_epoch = np.zeros(shape=(len(orig_trainset)))
curr_epoch_image_weight = np.ones(shape=(len(orig_trainset)))

# For No Augmentation, set below variables accordingly
if MSP_AUG_PCT == 0:
    to_augment_next_epoch = np.zeros(shape=(len(orig_trainset)))

print("\n", "*" * 100)
print(f"Augmenting the Bottom {int(MSP_AUG_PCT * 100)}% MSP [Copying the Bottom {int(COPY_PCT * 100)}% MSP with {NUM_COPIES} Additional Copies] starting from Epoch {TGT_AUG_EPOCH_START} to Epoch {TGT_AUG_EPOCH_STOP}")

print("*" * 100, "\n")

orig_trainloader = DataLoader(
    orig_trainset,
    batch_size=config.TRAIN_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    worker_init_fn=np.random.seed(seed_value),
)

testset = classes.CIFAR10_TEST()
testloader = DataLoader(
    testset,
    batch_size=config.TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    worker_init_fn=np.random.seed(seed_value),
)

net = wide_resnet.Wide_ResNet(
    depth=28, widen_factor=10, dropout_rate=0, num_classes=len(config.CLASSES_C10)
)
net = net.to(device)

if device == "cuda":
    net = torch.nn.DataParallel(net)

criterion = nn.CrossEntropyLoss()
unpacked_criterion = nn.CrossEntropyLoss(reduction='none')

optimizer = optim.SGD(
    net.parameters(), lr=config.LR, momentum=0.9, weight_decay=5e-4, nesterov=True
)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[10, 20, 30], gamma=0.2
)

# Initialize Prediction Arrays
train_target_probs = -1 * np.ones(shape=(len(orig_trainset)))
train_argmax_predictions = -1 * np.ones(shape=(len(orig_trainset)))
# print("#"*10,"Pre-Epoch {0} Predictions Written : {1}".format(loop_type, np.sum(target_pred_probs > -1)))
print("#" * 10, "Pre-Epoch Predictions Written : {0}".format(np.sum(train_argmax_predictions > -1)))

test_epoch_predictions = np.zeros(shape=(len(testset)))

# Softmax for Predictions
softmax = torch.nn.Softmax(dim=-1)


def train(epoch):
    train_loss = 0
    correct = 0
    total = 0

    print("EPOCH {0}: Augment 1-hot Sum : {1}".format(epoch, np.sum(to_augment_next_epoch)))
    print("EPOCH {0}: Copy 1-hot Sum : {1}".format(epoch, np.sum(to_copy_next_epoch)))

    if not _using_longtail_dataset:
        curr_trainset = classes.CIFAR10_DYNAMIC(augment_indicator=to_augment_next_epoch,
                                                # num_additional_copies=0 if epoch < TGT_AUG_EPOCH_START else NUM_COPIES,
                                                num_additional_copies=NUM_COPIES if TGT_AUG_EPOCH_START <= epoch <= TGT_AUG_EPOCH_STOP else 0,
                                                )
    else:
        # curr_trainset = classes.LONGTAIL_CIFAR10_DYNAMIC(dataset_npz=_train_npz,
        #                                                  augment_indicator=to_augment_next_epoch,
        #                                                  num_additional_copies=0 if epoch <= TGT_AUG_EPOCH_START else NUM_COPIES)

        curr_trainset = classes.LONGTAIL_CIFAR10_DYNAMIC_TEMP(dataset_npz=_train_npz,
                                                         augment_indicator=to_augment_next_epoch,
                                                         copy_indicator=to_copy_next_epoch,
                                                         num_additional_copies=NUM_COPIES if TGT_AUG_EPOCH_START<=epoch<=TGT_AUG_EPOCH_STOP else 0,
                                                         # num_additional_copies=0 if epoch < TGT_AUG_EPOCH_START else NUM_COPIES,
        )

    if REWIND_ACTION:
        curr_trainset.filter_dataset(ixs_to_keep=torch.where(torch.tensor(np.load(REWIND_INDICATOR))==1)[0])
    print(f"Length of Dataset for this epoch is : {len(curr_trainset)}")

    if (epoch >= INTERVENTION_ACT_EPOCH) and (RELABEL_PCT != 0.0):
        # If Previous Epoch involved Relabelling, then load new labels!
        curr_labels = np.load(os.path.join(WRITE_FOLDER, 'LATEST_RELABELS_FOR_DATASET.npy'))
        print("Using New Labels")
        curr_trainset.make_dataset_new_labels(new_labels=curr_labels.tolist())

    curr_trainloader = DataLoader(
        curr_trainset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        worker_init_fn=np.random.seed(seed_value),
    )

    # Zero Out Epoch Matrix at Epoch Start
    train_target_probs.fill(-1)
    train_argmax_predictions.fill(-1)

    for ixs, inputs, targets in curr_trainloader:
        net.train()
        train_inputs, train_targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(train_inputs)

        if TRAIN_DATASET=='cifar10':
            loss = criterion(outputs, train_targets)
        else:
            loss = unpacked_criterion(outputs, train_targets)
            curr_batch_weight = torch.as_tensor(curr_epoch_image_weight[torch.tensor(curr_trainset.map_to_orig_ix[ixs], dtype=int)]).to(device)
            loss = torch.mean(curr_batch_weight * loss)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += train_targets.size(0)
        correct += predicted.eq(train_targets).sum().item()

        # Write Predictions
        target_softmax_output = softmax(outputs.clone().cpu().detach())[np.arange(len(targets)), targets]
        train_target_probs[ixs[ixs < len(orig_trainset)]] = target_softmax_output[ixs < len(orig_trainset)]

        train_argmax_predictions[ixs[ixs < len(orig_trainset)]] = \
        torch.argmax(softmax(outputs.clone().cpu().detach()), dim=-1, keepdim=False)[ixs < len(orig_trainset)]

    scheduler.step()
    loss = train_loss / len(orig_trainloader)
    acc = 100.0 * correct / total
    return acc, loss


def test(epoch):
    test_loss = 0
    correct = 0
    total = 0
    class_correct = list(0.0 for i in range(len(config.CLASSES_C10)))
    class_total = list(0.0 for i in range(len(config.CLASSES_C10)))

    # Zero out Preds at Epoch Start
    test_epoch_predictions.fill(0)

    for ixs, inputs, targets in testloader:
        net.eval()
        test_inputs, test_targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(test_inputs)
        loss = criterion(outputs, test_targets)

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
collect_label_change_data = []

collect_predprob_train_data = {}
collect_predprob_test_data = {}

_track_lr = optimizer.param_groups[0]["lr"]
print("Learning Rate --> {1}".format(_track_lr, optimizer.param_groups[0]["lr"]))
for epoch in tqdm(range(args.EPOCHS)):

    if (epoch == INTERVENTION_ACT_EPOCH) and (DOWNWEIGHT_PCT != 0.0):
        print(f"Curr Epoch Image Weight! : Pre-Sum {np.sum(curr_epoch_image_weight)}")
        curr_epoch_image_weight[ix_for_downweighting] = DOWNWEIGHT_TO
        print(f"Curr Epoch Image Weight! : Post-Sum {np.sum(curr_epoch_image_weight)}")

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
    collect_predprob_train_data['EPOCH_{0}'.format(str(epoch))] = train_target_probs.tolist()
    collect_predprob_test_data['EPOCH_{0}'.format(str(epoch))] = test_epoch_predictions.tolist()

    curr_train_target_probs = np.array([round(n, 5) for n in train_target_probs.tolist()])
    with open(os.path.join(TARGET_PROBS_FOLDER, 'EPOCH_{0}'.format(str(epoch)) + '.npy'), 'wb') as f:
        np.save(f, curr_train_target_probs)

    with open(os.path.join(MODEL_PREDS_FOLDER, 'EPOCH_{0}'.format(str(epoch)) + '.npy'), 'wb') as f:
        np.save(f, train_argmax_predictions)

    to_augment_next_epoch.fill(1)
    to_copy_next_epoch.fill(0)

    TGT_AUGMENT_SCHEDULE = (TGT_AUG_EPOCH_START - 1 <= epoch <= TGT_AUG_EPOCH_STOP - 1)

    if TGT_AUGMENT_SCHEDULE:
        # Reset the Augment 1-Hot at every epoch
        to_augment_next_epoch.fill(0)

        print("Clearing the Augment 1-hot Sum: {1} ".format(epoch, np.sum(to_augment_next_epoch)))
        # ##################### Choosing using SFMX over the entire dataset #####################

        curr_sfmx_scores = train_target_probs

        _, min_sfmx_ix = torch.topk(
            torch.tensor(curr_sfmx_scores), k=int(len(orig_trainset) * MSP_AUG_PCT), largest=False
        )
        # Prep for AUGMENT in the next epoch
        to_augment_next_epoch[min_sfmx_ix] = 1


        to_copy_next_epoch.fill(0)
        print("Clearing the Copy 1-hot Sum: {1} ".format(epoch, np.sum(to_copy_next_epoch)))
        _, min_sfmx_ix_for_copy = torch.topk(
            torch.tensor(curr_sfmx_scores), k=int(len(orig_trainset) * COPY_PCT), largest=False
        )
        # Prep for Copy in the next epoch
        to_copy_next_epoch[min_sfmx_ix_for_copy] = 1
        print("Now the Copy 1-hot Sum: {1} ".format(epoch, np.sum(to_copy_next_epoch)))


    # Additional Information Available if using LongTail Datasets
    if not REWIND_ACTION and _using_longtail_dataset:
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

        noisy_aupr_sfmx = average_precision_score(y_true=noisy_1hot, y_score=-train_target_probs)
        atypical_aupr_sfmx = average_precision_score(y_true=atypical_1hot, y_score=-train_target_probs)

        collect_aupr_data.append((noisy_aupr_random, atypical_aupr_random, noisy_aupr_sfmx, atypical_aupr_sfmx, epoch))

    if (epoch == INTERVENTION_RECORD_EPOCH):
        ####### RELABEL #########

        curr_sfmx_scores = train_target_probs

        _, ix_for_downweighting = torch.topk(
            torch.tensor(curr_sfmx_scores), k=int(len(orig_trainset) * DOWNWEIGHT_PCT), largest=False
        )

        if (RELABEL_PCT != 0.0):
            _, ix_for_relabelling = torch.topk(
                torch.tensor(curr_sfmx_scores), k=int(len(orig_trainset) * RELABEL_PCT), largest=False
            )

            ix_for_relabelling = ix_for_relabelling.numpy()

            # Relabel for next epoch
            new_labels = np.array(curr_labels)
            label_change = len(ix_for_relabelling) - sum(
                new_labels[ix_for_relabelling] == train_argmax_predictions[ix_for_relabelling])
            collect_label_change_data.append((label_change, epoch))
            print("Relabelling {0} Images : {1}/{0} Labels Changed In This Epoch".format(len(ix_for_relabelling),
                                                                                         label_change))
            new_labels[ix_for_relabelling] = train_argmax_predictions[ix_for_relabelling]

            with open(os.path.join(WRITE_FOLDER, 'LATEST_RELABELS_FOR_DATASET.npy'), 'wb') as f:
                np.save(f, new_labels)

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

relabel_df = pd.DataFrame(
    data=collect_label_change_data,
    columns=[
        "labels_changed",
        "epoch",
    ]
)

collect_predprob_train_data_df = pd.DataFrame.from_dict(collect_predprob_train_data)
collect_predprob_test_data_df = pd.DataFrame.from_dict(collect_predprob_test_data)

# Write Files
mtrx_df.to_csv(os.path.join(WRITE_FOLDER, "metrics.csv"), index=False)

if (RELABEL_PCT != 0.0):
    relabel_df.to_csv(os.path.join(WRITE_FOLDER, "relabel.csv"), index=False)

collect_predprob_train_data_df.to_csv(os.path.join(WRITE_FOLDER, "train_predprob.csv"), index=False)
collect_predprob_test_data_df.to_csv(os.path.join(WRITE_FOLDER, "test_predprob.csv"), index=False)

# Write Additional Files( if using LongTail dataset)
if not REWIND_ACTION and _using_longtail_dataset:
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
