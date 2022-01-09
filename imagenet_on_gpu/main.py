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

import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
import pandas as pd

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0

#####################################################
# Settings
# TRAIN_DATASET = 'imagenet'
# TRAIN_DATASET = 'N20_A20_T60'
TRAIN_DATASET = 'N20_A20_TX2'

DATASET_SIZE = 1281167
if TRAIN_DATASET=='N20_A20_TX2':
    DATASET_SIZE = 1281152
DATASET_SIZE = 10000

MSP_AUG_PCT = 0.2
#####################################################
ADD_AUG_COPIES = 0
TGT_AUG_EPOCH_AFTER = 4

assert 0 <= MSP_AUG_PCT <= 1, "MSP_AUG_PCT must be between 0 and 1"

# Softmax for Predictions
softmax = torch.nn.Softmax(dim=-1)

train_epoch_predictions = np.zeros(shape=(DATASET_SIZE))

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    _using_longtail_dataset = False if TRAIN_DATASET == 'imagenet' else True

    EXP_NAME = 'aug_msp_{0}'.format(MSP_AUG_PCT)
    WRITE_FOLDER = os.path.join("{0}_{1}".format(seed_value, TRAIN_DATASET), EXP_NAME)

    # Folder to collect epoch snapshots
    if not os.path.exists(WRITE_FOLDER):
        os.makedirs(name=WRITE_FOLDER)


    dataset_props = {}
    dataset_props['traindir'] = traindir
    dataset_props['_using_longtail_dataset'] = _using_longtail_dataset
    dataset_props['size'] = DATASET_SIZE



    if not _using_longtail_dataset:
        print("{0}_Using Original({1}) Dataset_{0}".format("*" * 50, TRAIN_DATASET))
        orig_trainset = classes.IMAGENET(train_directory=traindir)
    else:
        print("{0}_Using LongTail({1}) Dataset_{0}".format("*" * 50, TRAIN_DATASET))
        _train_npz = os.path.join(config.DATASET_FOLDER, 'LONGTAIL_IMAGENET', TRAIN_DATASET + '.npz')
        dataset_props['_train_npz'] = _train_npz
        orig_trainset = classes.LONGTAIL_IMAGENET(train_directory=traindir, dataset_npz=_train_npz, apply_augmentation=False)

    print(orig_trainset)

    #  Initialize to all 1s to augment the entire dataset
    to_augment_next_epoch = np.ones(shape=(len(orig_trainset)))

    # For No Augmentation, set below variables accordingly
    if MSP_AUG_PCT == 0:
        to_augment_next_epoch = np.zeros(shape=(len(orig_trainset)))

    print("\n", "*" * 100)
    print("Augmenting the Bottom {0}% MSP with {1} Additional Copies starting after Epoch {2}".format(
        int(MSP_AUG_PCT * 100), ADD_AUG_COPIES, TGT_AUG_EPOCH_AFTER))

    print("*" * 100, "\n")

    # train_loader = torch.utils.data.DataLoader(
    #     orig_trainset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        classes.IMAGENET(valdir,apply_transform=True, apply_augmentation=True)
        ,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return



    # Main Training Loop
    collect_mtrx_data = []
    collect_aupr_data = []

    collect_predprob_train_data = {}

    _track_lr = optimizer.param_groups[0]["lr"]
    print("Learning Rate --> {1}".format(_track_lr, optimizer.param_groups[0]["lr"]))

    for epoch in range(args.start_epoch, args.epochs):

        AUGMENT_SCHEDULE = (epoch >= TGT_AUG_EPOCH_AFTER)

        # Check for LR Changes
        if _track_lr != optimizer.param_groups[0]["lr"]:
            print(
                "Learning Rate updated from {0} --> {1}".format(
                    _track_lr, optimizer.param_groups[0]["lr"]
                )
            )
            _track_lr = optimizer.param_groups[0]["lr"]

        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_acc, train_loss = train(model, criterion, optimizer, epoch, dataset_props, to_augment_next_epoch, args)

        # evaluate on validation set
        test_acc, test_loss = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = test_acc > best_acc1
        best_acc1 = max(test_acc, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

        collect_mtrx_data.append((train_acc, train_loss, test_acc, test_loss, epoch))

        print(
            "Epoch: {0} | Train_Acc: {1}\tTrain_Loss: {2} | Test_Acc: {3}\tTest_Loss: {4}".format(
                epoch, train_acc, train_loss, test_acc, test_loss
            )
        )

        # Write Predictons
        collect_predprob_train_data['EPOCH_{0}'.format(str(epoch))] = [round(n,5) for n in train_epoch_predictions.tolist()]

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
                noisy_aupr_random = average_precision_score(y_true=noisy_1hot,
                                                            y_score=np.random.rand(len(orig_trainset)))
                atypical_aupr_random = average_precision_score(y_true=atypical_1hot,
                                                               y_score=np.random.rand(len(orig_trainset)))

                noisy_aupr_sfmx = average_precision_score(y_true=noisy_1hot, y_score=-train_epoch_predictions)
                atypical_aupr_sfmx = average_precision_score(y_true=atypical_1hot, y_score=-train_epoch_predictions)

                collect_aupr_data.append(
                    (noisy_aupr_random, atypical_aupr_random, noisy_aupr_sfmx, atypical_aupr_sfmx, epoch))

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

    # Write Files
    mtrx_df.to_csv(os.path.join(WRITE_FOLDER, "metrics.csv"), index=False)
    collect_predprob_train_data_df.to_csv(os.path.join(WRITE_FOLDER, "train_predprob.csv"), index=False)


    # Write Additional Files( if using LongTail dataset)
    if dataset_props['_using_longtail_dataset']:

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


def train(model, criterion, optimizer, epoch, dataset_props, to_augment_next_epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    print("EPOCH {0}: Augment 1-hot Sum : {1}".format(epoch, np.sum(to_augment_next_epoch)))

    if not dataset_props['_using_longtail_dataset']:
        curr_trainset = classes.IMAGENET_DYNAMIC(augment_indicator=to_augment_next_epoch,
                                                num_additional_copies=0 if epoch <= TGT_AUG_EPOCH_AFTER else ADD_AUG_COPIES)
    else:
        curr_trainset = classes.LONGTAIL_IMAGENET_DYNAMIC(train_directory=os.path.join(args.data, 'train'),
                                                          dataset_npz=dataset_props['_train_npz'],
                                                         augment_indicator=to_augment_next_epoch,
                                                         num_additional_copies=0 if epoch <= TGT_AUG_EPOCH_AFTER else ADD_AUG_COPIES)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(curr_trainset)
    else:
        train_sampler = None

    curr_trainloader = DataLoader(
        curr_trainset,
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler
    )

    progress = ProgressMeter(
        len(curr_trainloader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    # Zero Out Epoch Matrix at Epoch Start
    train_epoch_predictions.fill(0)

    start_batch_time = time.time()

    for i, (ixs, images, target) in enumerate(curr_trainloader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Write Predictions
        target_softmax_output = softmax(output.clone().cpu().detach())[np.arange(len(target)), target]
        train_epoch_predictions[ixs[ixs < dataset_props['size']]] = target_softmax_output[ixs < dataset_props['size']]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            print("Total Predictions Written : {0}".format(np.sum(train_epoch_predictions > 0)))
            print("Step Size Time : {0}".format(pretty_time_delta(time.time() - start_batch_time)))
            start_batch_time = time.time()

    return round(top1.avg.cpu().item(),5), round(losses.avg,5)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return round(top1.avg.cpu().item(),5), round(losses.avg,5)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def pretty_time_delta(seconds):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%dd %dh %dm %ds' % (days, hours, minutes, seconds)
    elif hours > 0:
        return '%dh %dm %ds' % (hours, minutes, seconds)
    elif minutes > 0:
        return '%dm %ds' % (minutes, seconds)
    else:
        return '%ds' % (seconds,)

if __name__ == '__main__':
    main()