
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
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import torch as ch
import numpy as np
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from cox.utils import Parameters

from .tools import helpers
from .tools.helpers import AverageMeter, ckpt_at_epoch, has_attr
from .tools import constants as consts
from .longtail import classes
import dill 
import os
import time
import warnings
from sklearn.metrics import average_precision_score
import pandas as pd

if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

try:
    from apex import amp
except Exception as e:
    warnings.warn('Could not import amp.')


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


def check_required_args(args, eval_only=False):
    """
    Check that the required training arguments are present.

    Args:
        args (argparse object): the arguments to check
        eval_only (bool) : whether to check only the arguments for evaluation
    """
    required_args_eval = ["adv_eval"]
    required_args_train = ["epochs", "out_dir", "adv_train",
        "log_iters", "lr", "momentum", "weight_decay"]
    adv_required_args = ["attack_steps", "eps", "constraint", 
            "use_best", "attack_lr", "random_restarts"]

    # Generic function for checking all arguments in a list
    def check_args(args_list):
        for arg in args_list:
            assert has_attr(args, arg), f"Missing argument {arg}"

    # Different required args based on training or eval:
    if not eval_only: check_args(required_args_train)
    else: check_args(required_args_eval)
    # More required args if we are robustly training or evaling
    is_adv = bool(args.adv_train) or bool(args.adv_eval)
    if is_adv:
        check_args(adv_required_args)
    # More required args if the user provides a custom training loss
    has_custom_train = has_attr(args, 'custom_train_loss')
    has_custom_adv = has_attr(args, 'custom_adv_loss')
    if has_custom_train and is_adv and not has_custom_adv:
        raise ValueError("Cannot use custom train loss \
            without a custom adversarial loss (see docs)")


def make_optimizer_and_schedule(args, model, checkpoint, params):
    """
    *Internal Function* (called directly from train_model)

    Creates an optimizer and a schedule for a given model, restoring from a
    checkpoint if it is non-null.

    Args:
        args (object) : an arguments object, see
            :meth:`~robustness.train.train_model` for details
        model (AttackerModel) : the model to create the optimizer for
        checkpoint (dict) : a loaded checkpoint saved by this library and loaded
            with `ch.load`
        params (list|None) : a list of parameters that should be updatable, all
            other params will not update. If ``None``, update all params 

    Returns:
        An optimizer (ch.nn.optim.Optimizer) and a scheduler
            (ch.nn.optim.lr_schedulers module).
    """
    # Make optimizer
    param_list = model.parameters() if params is None else params
    optimizer = SGD(param_list, args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.mixed_precision:
        model.to('cuda')
        model, optimizer = amp.initialize(model, optimizer, 'O1')

    # Make schedule
    schedule = None
    if args.custom_lr_multiplier == 'cyclic':
        eps = args.epochs
        lr_func = lambda t: np.interp([t], [0, eps*4//15, eps], [0, 1, 0])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.custom_lr_multiplier:
        cs = args.custom_lr_multiplier
        periods = eval(cs) if type(cs) is str else cs
        if args.lr_interpolation == 'linear':
            lr_func = lambda t: np.interp([t], *zip(*periods))[0]
        else:
            def lr_func(ep):
                for (milestone, lr) in reversed(periods):
                    if ep >= milestone: return lr
                return 1.0
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.step_lr:
        schedule = lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=args.step_lr_gamma)

    # Fast-forward the optimizer and the scheduler if resuming
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            schedule.load_state_dict(checkpoint['schedule'])
        except:
            steps_to_take = checkpoint['epoch']
            print('Could not load schedule (was probably LambdaLR).'
                  f' Stepping {steps_to_take} times instead...')
            for i in range(steps_to_take):
                schedule.step()
        
        if 'amp' in checkpoint and checkpoint['amp'] not in [None, 'N/A']:
            amp.load_state_dict(checkpoint['amp'])

        # TODO: see if there's a smarter way to do this
        # TODO: see what's up with loading fp32 weights and then MP training
        if args.mixed_precision:
            model.load_state_dict(checkpoint['model'])

    return optimizer, schedule

def eval_model(args, model, loader, store):
    """
    Evaluate a model for standard (and optionally adversarial) accuracy.

    Args:
        args (object) : A list of arguments---should be a python object 
            implementing ``getattr()`` and ``setattr()``.
        model (AttackerModel) : model to evaluate
        loader (iterable) : a dataloader serving `(input, label)` batches from
            the validation set
        store (cox.Store) : store for saving results in (via tensorboardX)
    """
    check_required_args(args, eval_only=True)
    start_time = time.time()

    if store is not None: 
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)
    writer = store.tensorboard if store else None

    assert not hasattr(model, "module"), "model is already in DataParallel."
    model = ch.nn.DataParallel(model)

    prec1, nat_loss, _, _ = _model_loop(args, 'val', 50000, loader,
                                        model, None, 0, False, writer)

    adv_prec1, adv_loss = float('nan'), float('nan')
    if args.adv_eval: 
        args.eps = eval(str(args.eps)) if has_attr(args, 'eps') else None
        args.attack_lr = eval(str(args.attack_lr)) if has_attr(args, 'attack_lr') else None
        adv_prec1, adv_loss, _, _ = _model_loop(args, 'val', loader,
                                        model, None, 0, True, writer)
    log_info = {
        'epoch':0,
        'nat_prec1':prec1,
        'adv_prec1':adv_prec1,
        'nat_loss':nat_loss,
        'adv_loss':adv_loss,
        'train_prec1':float('nan'),
        'train_loss':float('nan'),
        'time': time.time() - start_time
    }

    # Log info into the logs table
    if store: store[consts.LOGS_TABLE].append_row(log_info)
    return log_info

def train_model(args, model, *, checkpoint=None, dp_device_ids=None,
            store=None, update_params=None, disable_no_grad=False):
    """
    Main function for training a model. 

    Args:
        args (object) : A python object for arguments, implementing
            ``getattr()`` and ``setattr()`` and having the following
            attributes. See :attr:`robustness.defaults.TRAINING_ARGS` for a 
            list of arguments, and you can use
            :meth:`robustness.defaults.check_and_fill_args` to make sure that
            all required arguments are filled and to fill missing args with
            reasonable defaults:

            adv_train (int or bool, *required*)
                if 1/True, adversarially train, otherwise if 0/False do 
                standard training
            epochs (int, *required*)
                number of epochs to train for
            lr (float, *required*)
                learning rate for SGD optimizer
            weight_decay (float, *required*)
                weight decay for SGD optimizer
            momentum (float, *required*)
                momentum parameter for SGD optimizer
            step_lr (int)
                if given, drop learning rate by 10x every `step_lr` steps
            custom_lr_multplier (str)
                If given, use a custom LR schedule, formed by multiplying the
                    original ``lr`` (format: [(epoch, LR_MULTIPLIER),...])
            lr_interpolation (str)
                How to drop the learning rate, either ``step`` or ``linear``,
                    ignored unless ``custom_lr_multiplier`` is provided.
            adv_eval (int or bool)
                If True/1, then also do adversarial evaluation, otherwise skip
                (ignored if adv_train is True)
            log_iters (int, *required*)
                How frequently (in epochs) to save training logs
            save_ckpt_iters (int, *required*)
                How frequently (in epochs) to save checkpoints (if -1, then only
                save latest and best ckpts)
            attack_lr (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack step size
            constraint (str, *required if adv_train or adv_eval*)
                the type of adversary constraint
                (:attr:`robustness.attacker.STEPS`)
            eps (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack budget
            attack_steps (int, *required if adv_train or adv_eval*)
                number of steps to take in adv attack
            custom_eps_multiplier (str, *required if adv_train or adv_eval*)
                If given, then set epsilon according to a schedule by
                multiplying the given eps value by a factor at each epoch. Given
                in the same format as ``custom_lr_multiplier``, ``[(epoch,
                MULTIPLIER)..]``
            use_best (int or bool, *required if adv_train or adv_eval*) :
                If True/1, use the best (in terms of loss) PGD step as the
                attack, if False/0 use the last step
            random_restarts (int, *required if adv_train or adv_eval*)
                Number of random restarts to use for adversarial evaluation
            custom_train_loss (function, optional)
                If given, a custom loss instead of the default CrossEntropyLoss.
                Takes in `(logits, targets)` and returns a scalar.
            custom_adv_loss (function, *required if custom_train_loss*)
                If given, a custom loss function for the adversary. The custom
                loss function takes in `model, input, target` and should return
                a vector representing the loss for each element of the batch, as
                well as the classifier output.
            custom_accuracy (function)
                If given, should be a function that takes in model outputs
                and model targets and outputs a top1 and top5 accuracy, will 
                displayed instead of conventional accuracies
            regularizer (function, optional) 
                If given, this function of `model, input, target` returns a
                (scalar) that is added on to the training loss without being
                subject to adversarial attack
            iteration_hook (function, optional)
                If given, this function is called every training iteration by
                the training loop (useful for custom logging). The function is
                given arguments `model, iteration #, loop_type [train/eval],
                current_batch_ims, current_batch_labels`.
            epoch hook (function, optional)
                Similar to iteration_hook but called every epoch instead, and
                given arguments `model, log_info` where `log_info` is a
                dictionary with keys `epoch, nat_prec1, adv_prec1, nat_loss,
                adv_loss, train_prec1, train_loss`.

        model (AttackerModel) : the model to train.
        loaders (tuple[iterable]) : `tuple` of data loaders of the form
            `(train_loader, val_loader)` 
        checkpoint (dict) : a loaded checkpoint previously saved by this library
            (if resuming from checkpoint)
        dp_device_ids (list|None) : if not ``None``, a list of device ids to
            use for DataParallel.
        store (cox.Store) : a cox store for logging training progress
        update_params (list) : list of parameters to use for training, if None
            then all parameters in the model are used (useful for transfer
            learning)
        disable_no_grad (bool) : if True, then even model evaluation will be
            run with autograd enabled (otherwise it will be wrapped in a ch.no_grad())
    """
    # Logging setup
    writer = store.tensorboard if store else None
    prec1_key = f"{'adv' if args.adv_train else 'nat'}_prec1"
    if store is not None: 
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)
    
    # Reformat and read arguments
    check_required_args(args) # Argument sanity check
    for p in ['eps', 'attack_lr', 'custom_eps_multiplier']:
        setattr(args, p, eval(str(getattr(args, p))) if has_attr(args, p) else None)
    if args.custom_eps_multiplier is not None: 
        eps_periods = args.custom_eps_multiplier
        args.custom_eps_multiplier = lambda t: np.interp([t], *zip(*eps_periods))[0]

    # Initial setup
    MSP_AUG_PCT = args.msp_aug_pct
    RELABEL_PCT = args.relabel_pct
    RELABEL_EPOCH = args.relabel_epoch
    #####################################################
    ADD_AUG_COPIES = 0
    TGT_AUG_EPOCH_AFTER = 4

    start_track_time = time.time()

    assert 0 <= MSP_AUG_PCT <= 1, "MSP_AUG_PCT must be between 0 and 1"
    assert 0 <= RELABEL_PCT <= 1, "RELABEL_PCT must be between 0 and 1"

    data_path = args.data
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'val')

    EXP_NAME = 'aug_msp_{0}'.format(MSP_AUG_PCT)
    if RELABEL_PCT==-1:
        WRITE_FOLDER = os.path.join(
            "TEMP_{0}_RELABEL_{1}_{2}".format(seed_value, RELABEL_PCT, args.longtail_dataset),
            EXP_NAME)
    else:
        WRITE_FOLDER = os.path.join("TEMP_{0}_RELABEL_{1}_at_{2}_{3}".format(seed_value, RELABEL_PCT, RELABEL_EPOCH, args.longtail_dataset), EXP_NAME)
    # Folder to collect epoch snapshots
    if not os.path.exists(WRITE_FOLDER):
        os.makedirs(name=WRITE_FOLDER)

    TRAIN_PREDS_FOLDER = os.path.join(WRITE_FOLDER,'train_pred_npz_files')
    if not os.path.exists(TRAIN_PREDS_FOLDER):
        os.makedirs(name=TRAIN_PREDS_FOLDER)

    _using_longtail_dataset = True if args.longtail_dataset!=None else False

    if not _using_longtail_dataset:
        print("{0}_Using Original({1}) Dataset_{0}".format("*" * 50, "IMAGENET"))
        orig_trainset = classes.IMAGENET(train_directory=train_path, apply_transform=True, apply_augmentation=True)
    else:
        print("{0}_Using LongTail({1}) Dataset_{0}".format("*" * 50, args.longtail_dataset))
        _train_npz = os.path.join(args.longtail_folder, 'LONGTAIL_IMAGENET', args.longtail_dataset + '.npz')
        # dataset_props['_train_npz'] = _train_npz
        orig_trainset = classes.LONGTAIL_IMAGENET(train_directory=train_path, dataset_npz=_train_npz, apply_augmentation=False)

    print(orig_trainset)

    # Initialize for Relabel
    # print("Reading Default Labels")
    curr_labels = [d[1] for d in orig_trainset.dataset]
    # print("Writing Default Labels")
    with open(os.path.join(WRITE_FOLDER,'LATEST_RELABELS_FOR_DATASET.npy'), 'wb') as f:
        np.save(f, np.array([orig_trainset.class_2_ix[c] for c in curr_labels],dtype=float))
    # print("Done!")
    #  Initialize to all 1s to augment the entire dataset
    to_augment_next_epoch = np.ones(shape=(len(orig_trainset)))

    # For No Augmentation, set below variables accordingly
    if MSP_AUG_PCT == 0:
        to_augment_next_epoch = np.zeros(shape=(len(orig_trainset)))

    print("\n", "*" * 100)
    print("Augmenting the Bottom {0}% MSP with {1} Additional Copies starting after Epoch {2}".format(
        int(MSP_AUG_PCT * 100), ADD_AUG_COPIES, TGT_AUG_EPOCH_AFTER))

    print("*" * 100, "\n")

    val_set = classes.IMAGENET_TEST(test_directory=test_path, class_2_ix=orig_trainset.class_2_ix, apply_transform=True)

    val_loader = DataLoader(val_set, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = helpers.DataPrefetcher(val_loader)

    opt, schedule = make_optimizer_and_schedule(args, model, checkpoint, update_params)

    # Put the model into parallel mode
    assert not hasattr(model, "module"), "model is already in DataParallel."
    model = ch.nn.DataParallel(model, device_ids=dp_device_ids).cuda()

    best_prec1, start_epoch = (0, 0)
    if checkpoint:
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint[prec1_key] if prec1_key in checkpoint \
            else _model_loop(args, 'val', val_loader, model, None, start_epoch-1, args.adv_train, writer=None)[0]


    # Main Training Loop
    collect_mtrx_data = []
    collect_aupr_data = []
    collect_label_change_data = []

    # collect_predprob_train_data = {}

    # Timestamp for training start time
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):

        AUGMENT_SCHEDULE = (epoch >= TGT_AUG_EPOCH_AFTER)
        RELABEL_SCHEDULE = AUGMENT_SCHEDULE if RELABEL_EPOCH==-1 else (epoch == RELABEL_EPOCH)

        print("EPOCH {0}: Augment 1-hot Sum : {1}".format(epoch, np.sum(to_augment_next_epoch)))

        if not _using_longtail_dataset:
            curr_trainset = classes.IMAGENET_DYNAMIC(augment_indicator=to_augment_next_epoch,
                                                     num_additional_copies=0 if epoch <= TGT_AUG_EPOCH_AFTER else ADD_AUG_COPIES)
        else:
            curr_trainset = classes.LONGTAIL_IMAGENET_DYNAMIC(train_directory=train_path,
                                                              dataset_npz=_train_npz,
                                                              augment_indicator=to_augment_next_epoch,
                                                              num_additional_copies=0 if epoch <= TGT_AUG_EPOCH_AFTER else ADD_AUG_COPIES)


        if AUGMENT_SCHEDULE:
            curr_labels = np.load(os.path.join(WRITE_FOLDER,'LATEST_RELABELS_FOR_DATASET.npy'))
            print("Using New Labels")
            curr_trainset.make_dataset_new_labels(new_labels=[orig_trainset.ix_2_class[i] for i in curr_labels.tolist()])


        curr_train_loader = DataLoader(curr_trainset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers, pin_memory=True)
        curr_train_loader = helpers.DataPrefetcher(curr_train_loader)

        # train for one epoch
        train_prec1, train_loss, train_target_probs, train_argmax_predictions = _model_loop(args, 'train', len(orig_trainset), curr_train_loader,
                model, opt, epoch, args.adv_train, writer)
        last_epoch = (epoch == (args.epochs - 1))

        # ctx = ch.enable_grad() if disable_no_grad else ch.no_grad()
        # with ctx:
        #     prec1, nat_loss, _, _ = _model_loop(args, 'val', len(val_set), val_loader, model,
        #                                      None, epoch, False, writer)

        # round(top1.avg.cpu().item(), 5), round(losses.avg, 5)
        # evaluate on validation set
        sd_info = {
            'model':model.state_dict(),
            'optimizer':opt.state_dict(),
            'schedule':(schedule and schedule.state_dict()),
            'epoch': epoch+1,
            'amp': amp.state_dict() if args.mixed_precision else None,
        }


        def save_checkpoint(filename):
            ckpt_save_path = os.path.join(args.out_dir if not store else \
                                          store.path, filename)
            ch.save(sd_info, ckpt_save_path, pickle_module=dill)

        save_its = args.save_ckpt_iters
        should_save_ckpt = (epoch % save_its == 0) and (save_its > 0)
        should_log = (epoch % args.log_iters == 0)

        if should_log or last_epoch or should_save_ckpt:
            # log + get best
            ctx = ch.enable_grad() if disable_no_grad else ch.no_grad() 
            with ctx:
                prec1, nat_loss, _, _  = _model_loop(args, 'val', len(val_set), val_loader, model,
                        None, epoch, False, writer)

            collect_mtrx_data.append((round(train_prec1.cpu().item(), 5), round(train_loss, 5),
                                      round(prec1.cpu().item(), 5), round(nat_loss, 5), epoch))

            # loader, model, epoch, input_adv_exs
            should_adv_eval = args.adv_eval or args.adv_train
            adv_val = should_adv_eval and _model_loop(args, 'val', val_loader,
                    model, None, epoch, True, writer)
            adv_prec1, adv_loss = adv_val or (-1.0, -1.0)

            # remember best prec@1 and save checkpoint
            our_prec1 = adv_prec1 if args.adv_train else prec1
            is_best = our_prec1 > best_prec1
            best_prec1 = max(our_prec1, best_prec1)
            sd_info[prec1_key] = our_prec1

            # log every checkpoint
            log_info = {
                'epoch':epoch + 1,
                'nat_prec1':prec1,
                'adv_prec1':adv_prec1,
                'nat_loss':nat_loss,
                'adv_loss':adv_loss,
                'train_prec1':train_prec1,
                'train_loss':train_loss,
                'time': time.time() - start_time
            }

            # Log info into the logs table
            if store: store[consts.LOGS_TABLE].append_row(log_info)
            # If we are at a saving epoch (or the last epoch), save a checkpoint
            if should_save_ckpt or last_epoch: save_checkpoint(ckpt_at_epoch(epoch))

            # Update the latest and best checkpoints (overrides old one)
            save_checkpoint(consts.CKPT_NAME_LATEST)
            if is_best: save_checkpoint(consts.CKPT_NAME_BEST)

        if schedule: schedule.step()
        if has_attr(args, 'epoch_hook'): args.epoch_hook(model, log_info)

        # Write Predictons
        # collect_predprob_train_data['EPOCH_{0}'.format(str(epoch))] = [round(n,5) for n in train_target_probs.tolist()]

        curr_train_target_probs = np.array([round(n,5) for n in train_target_probs.tolist()])
        with open(os.path.join(TRAIN_PREDS_FOLDER,'EPOCH_{0}'.format(str(epoch))+'.npy'), 'wb') as f:
            np.save(f, curr_train_target_probs)

        if AUGMENT_SCHEDULE:
            # Reset the Augment 1-Hot at every epoch
            to_augment_next_epoch.fill(0)

            # print("Clearing the Augment 1-hot Sum: {1} ".format(epoch, np.sum(to_augment_next_epoch)))
            # ##################### Choosing using SFMX over the entire dataset #####################

            curr_sfmx_scores = train_target_probs

            _, min_sfmx_ix = ch.topk(
                ch.tensor(curr_sfmx_scores), k=int(len(orig_trainset) * MSP_AUG_PCT), largest=False
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
                noisy_aupr_random = round(average_precision_score(y_true=noisy_1hot,
                                                            y_score=np.random.rand(len(orig_trainset))), 5)
                atypical_aupr_random = round(average_precision_score(y_true=atypical_1hot,
                                                               y_score=np.random.rand(len(orig_trainset))), 5)

                noisy_aupr_sfmx = round(average_precision_score(y_true=noisy_1hot, y_score=-train_target_probs), 5)
                atypical_aupr_sfmx = round(average_precision_score(y_true=atypical_1hot, y_score=-train_target_probs), 5)

                collect_aupr_data.append(
                    (noisy_aupr_random, atypical_aupr_random, noisy_aupr_sfmx, atypical_aupr_sfmx, epoch))

        if RELABEL_SCHEDULE:
            ####### RELABEL #########
            new_labels = curr_labels

            _, ix_for_relabelling = ch.topk(
                ch.tensor(curr_sfmx_scores), k=int(len(orig_trainset) * RELABEL_PCT), largest=False
            )

            # Relabel for next epoch
            # print("IXS : {0}".format(ix_for_relabelling[:10]))
            # print(new_labels[ix_for_relabelling][:10])
            # print(train_argmax_predictions[ix_for_relabelling][:10])

            label_change = len(ix_for_relabelling) - sum(new_labels[ix_for_relabelling]==train_argmax_predictions[ix_for_relabelling])
            collect_label_change_data.append((label_change, epoch))
            print("Relabelling {0} Images : {1}/{0} Labels Changed In This Epoch".format(len(ix_for_relabelling),label_change))
            new_labels[ix_for_relabelling]  = train_argmax_predictions[ix_for_relabelling]

            with open(os.path.join(WRITE_FOLDER,'LATEST_RELABELS_FOR_DATASET.npy'), 'wb') as f:
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
        data = collect_label_change_data,
        columns=[
            "labels_changed",
            "epoch",
        ]
    )

    # collect_predprob_train_data_df = pd.DataFrame.from_dict(collect_predprob_train_data)

    # Write Files
    mtrx_df.to_csv(os.path.join(WRITE_FOLDER, "metrics.csv"), index=False)
    relabel_df.to_csv(os.path.join(WRITE_FOLDER, "relabel.csv"), index=False)

    # collect_predprob_train_data_df.to_csv(os.path.join(WRITE_FOLDER, "train_predprob.csv"), index=False)


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

    print("*"*10,"Total Training Time : {0}".format(pretty_time_delta(time.time() - start_track_time)),"*"*10)

    return model

def _model_loop(args, loop_type, dataset_size, loader, model, opt, epoch, adv, writer):
    """
    *Internal function* (refer to the train_model and eval_model functions for
    how to train and evaluate models).

    Runs a single epoch of either training or evaluating.

    Args:
        args (object) : an arguments object (see
            :meth:`~robustness.train.train_model` for list of arguments
        loop_type ('train' or 'val') : whether we are training or evaluating
        dataset_size : length of the dataset
        loader (iterable) : an iterable loader of the form 
            `(image_batch, label_batch)`
        model (AttackerModel) : model to train/evaluate
        opt (ch.optim.Optimizer) : optimizer to use (ignored for evaluation)
        epoch (int) : which epoch we are currently on
        adv (bool) : whether to evaluate adversarially (otherwise standard)
        writer : tensorboardX writer (optional)

    Returns:
        The average top1 accuracy and the average loss across the epoch.
    """
    if not loop_type in ['train', 'val']:
        err_msg = "loop_type ({0}) must be 'train' or 'val'".format(loop_type)
        raise ValueError(err_msg)
    is_train = (loop_type == 'train')

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    prec = 'NatPrec' if not adv else 'AdvPrec'
    loop_msg = 'Train' if loop_type == 'train' else 'Val'

    # switch to train/eval mode depending
    model = model.train() if is_train else model.eval()

    # If adv training (or evaling), set eps and random_restarts appropriately
    if adv:
        eps = args.custom_eps_multiplier(epoch) * args.eps \
                if (is_train and args.custom_eps_multiplier) else args.eps
        random_restarts = 0 if is_train else args.random_restarts

    # Custom training criterion
    has_custom_train_loss = has_attr(args, 'custom_train_loss')
    train_criterion = args.custom_train_loss if has_custom_train_loss \
            else ch.nn.CrossEntropyLoss()
    
    has_custom_adv_loss = has_attr(args, 'custom_adv_loss')
    adv_criterion = args.custom_adv_loss if has_custom_adv_loss else None

    attack_kwargs = {}
    if adv:
        attack_kwargs = {
            'constraint': args.constraint,
            'eps': eps,
            'step_size': args.attack_lr,
            'iterations': args.attack_steps,
            'random_start': args.random_start,
            'custom_loss': adv_criterion,
            'random_restarts': random_restarts,
            'use_best': bool(args.use_best)
        }

    iterator = tqdm(enumerate(loader), total=len(loader))

    # target_pred_probs = np.zeros(shape=(dataset_size))
    target_pred_probs = -1 * np.ones(shape=(dataset_size))
    model_argmax_preds = -1 * np.ones(shape=(dataset_size))
    # print("#"*10,"Pre-Epoch {0} Predictions Written : {1}".format(loop_type, np.sum(target_pred_probs > -1)))
    print("#"*10,"Pre-Epoch {0} Predictions Written : {1}".format(loop_type, np.sum(model_argmax_preds > -1)))

    # Softmax for Predictions
    softmax = torch.nn.Softmax(dim=-1)

    for i, (idx, inp, target) in iterator:
       # measure data loading time
        target = target.cuda(non_blocking=True)
        output, final_inp = model(inp, target=target, make_adv=adv,
                                  **attack_kwargs)
        loss = train_criterion(output, target)

        if len(loss.shape) > 0: loss = loss.mean()

        model_logits = output[0] if (type(output) is tuple) else output
        idx = idx.cpu().numpy()
        # logits = model_logits.detach().cpu().numpy()
        target_softmax_output = softmax(model_logits.clone().cpu().detach())[np.arange(len(target)), target]
        target_pred_probs[idx[idx < dataset_size]] = target_softmax_output[idx < dataset_size]

        model_softmax_output = softmax(model_logits.clone().cpu().detach())
        model_argmax_preds[idx[idx < dataset_size]] = torch.argmax(model_softmax_output, dim=-1, keepdim=False)[idx < dataset_size]
        # print("Total Predictions Written : {0}".format(np.sum(target_pred_probs > 0)))

    # measure accuracy and record loss
        top1_acc = float('nan')
        top5_acc = float('nan')
        try:
            maxk = min(5, model_logits.shape[-1])
            if has_attr(args, "custom_accuracy"):
                prec1, prec5 = args.custom_accuracy(model_logits, target)
            else:
                prec1, prec5 = helpers.accuracy(model_logits, target, topk=(1, maxk))
                prec1, prec5 = prec1[0], prec5[0]

            losses.update(loss.item(), inp.size(0))
            top1.update(prec1, inp.size(0))
            top5.update(prec5, inp.size(0))

            top1_acc = top1.avg
            top5_acc = top5.avg
        except:
            warnings.warn('Failed to calculate the accuracy.')

        reg_term = 0.0
        if has_attr(args, "regularizer"):
            reg_term =  args.regularizer(model, inp, target)
        loss = loss + reg_term

        # compute gradient and do SGD step
        if is_train:
            opt.zero_grad()
            if args.mixed_precision:
                with amp.scale_loss(loss, opt) as sl:
                    sl.backward()
            else:
                loss.backward()
            opt.step()
        elif adv and i == 0 and writer:
            # add some examples to the tensorboard
            nat_grid = make_grid(inp[:15, ...])
            adv_grid = make_grid(final_inp[:15, ...])
            writer.add_image('Nat input', nat_grid, epoch)
            writer.add_image('Adv input', adv_grid, epoch)

        # ITERATOR
        desc = ('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
                '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | '
                'Reg term: {reg} ||'.format( epoch, prec, loop_msg, 
                loss=losses, top1_acc=top1_acc, top5_acc=top5_acc, reg=reg_term))

        # USER-DEFINED HOOK
        if has_attr(args, 'iteration_hook'):
            args.iteration_hook(model, i, loop_type, inp, target)

        iterator.set_description(desc)
        iterator.refresh()

    # print("#"*10,"Post-Epoch {0} Predictions Written : {1}".format(loop_type, np.sum(target_pred_probs > -1)))
    print("#" * 10, "Post-Epoch {0} Predictions Written : {1}".format(loop_type, np.sum(model_argmax_preds > -1)))

    if writer is not None:
        prec_type = 'adv' if adv else 'nat'
        descs = ['loss', 'top1', 'top5']
        vals = [losses, top1, top5]
        for d, v in zip(descs, vals):
            writer.add_scalar('_'.join([prec_type, loop_type, d]), v.avg,
                              epoch)

    return top1.avg, losses.avg, target_pred_probs, model_argmax_preds

