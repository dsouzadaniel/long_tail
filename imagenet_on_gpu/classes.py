# Libraries
from collections import Counter
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import torch

from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from typing import Dict, List
import os
from random import shuffle

# Imports
class IMAGENET(DatasetFolder):

    def __init__(self, train_directory, apply_transform=True, apply_augmentation=False):
        augmentation = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ]
            )
            if apply_augmentation == True
            else transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ]
            )
        )
        self.transforms = augmentation if apply_transform == True else None

        classes, self.class_2_ix = self.find_classes(train_directory)

        self.ix_2_class = {v: k for k, v in self.class_2_ix.items()}

        self.dataset = self.make_dataset(train_directory, self.class_2_ix)
        self.dataset_name = "ImageNet"

    def __getitem__(self, ix):
        path, target = self.dataset[ix]
        sample = self.loader(path)
        if self.transforms:
            data = self.transforms(sample)
        else:
            data = sample
        return ix, data, target

    def __len__(self):
        return len(self.dataset)

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        print("Classes Found : {0}".format(len(classes)))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def make_dataset(
            self,
            directory: str,
            class_2_ix: Dict[str, int]):
        """Generates a list of samples of a form (path_to_sample, class).
        Note: The class_2_ix parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)

        if class_2_ix is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_2_ix:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        instances = []
        available_classes = set()
        for target_class in sorted(class_2_ix.keys()):
            class_index = class_2_ix[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

        empty_classes = set(class_2_ix.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            raise FileNotFoundError(msg)

        shuffle(instances)
        # print("###"*10, len(list(set([i[1] for i in instances[:50000]]))))
        return instances[:50000]

        # return instances


    def __repr__(self):
        repr_str = f'*** {self.__class__.__name__}({self.dataset_name})'
        return repr_str

class IMAGENET_DYNAMIC(DatasetFolder):

    def __init__(self, train_directory, augment_indicator, num_additional_copies=0):

        self.augment_indicator = augment_indicator
        self.num_additional_copies = num_additional_copies

        # Get Original Dataset without any transform/augmentation
        _orig_dataset = IMAGENET(apply_augmentation=False).dataset

        self.augment_and_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )

        classes, self.class_2_ix = self.find_classes(train_directory)

        # Collect the Ixs for __getitem__
        self.ixs_to_augment = np.where(self.augment_indicator == 1)[0]

        expanded_dataset = []
        for ix in self.ixs_to_augment:
            expanded_dataset.extend([_orig_dataset[ix] for _ in range(self.num_additional_copies)])

        assert len(expanded_dataset) == np.sum(self.augment_indicator) * self.num_additional_copies, len(
            expanded_dataset
        )

        # Add the newly added ixs
        self.ixs_to_augment = np.concatenate((self.ixs_to_augment,np.arange(len(_orig_dataset),(len(_orig_dataset)+len(self.ixs_to_augment)))))

        self.dataset = _orig_dataset + expanded_dataset

        # shuffle so added images aren't all at the end
        self.shuffled_ix_mapping = np.random.permutation(len(self.dataset))

        self.dataset_name = "ImageNet Dynamic"

    def __getitem__(self, ix):
        path, target = self.dataset[ix]
        sample = self.loader(path)
        if self.transforms:
            data = self.transforms(sample)
        else:
            data = sample
        return ix, data, target

    def __len__(self):
        return len(self.dataset)

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        print("Classes Found : {0}".format(len(classes)))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def make_dataset(
            self,
            directory: str,
            class_2_ix: Dict[str, int]):
        """Generates a list of samples of a form (path_to_sample, class).
        Note: The class_2_ix parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)

        if class_2_ix is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_2_ix:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        instances = []
        available_classes = set()
        for target_class in sorted(class_2_ix.keys()):
            class_index = class_2_ix[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

        empty_classes = set(class_2_ix.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            raise FileNotFoundError(msg)

        shuffle(instances)
        return instances[:50000]

        # return instances

    def __repr__(self):
        repr_str = f'*** {self.__class__.__name__}({self.dataset_name})'
        return repr_str

class LONGTAIL_IMAGENET(DatasetFolder):

    def __init__(self, train_directory, dataset_npz, apply_transform=True, apply_augmentation=False):
        augmentation = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ]
            )
            if apply_augmentation == True
            else transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ]
            )
        )
        self.transforms = augmentation if apply_transform == True else None
        self._dataset_npz = np.load(dataset_npz, allow_pickle=True)

        self.orig_labels = [f[1] for f in self._dataset_npz['filenames']]
        classes, self.class_2_ix = self.find_classes(train_directory)

        self.ix_2_class = {v: k for k, v in self.class_2_ix.items()}
        self.class_2_text = self._dataset_npz['class_mapping'].item()

        self.labels = np.array([self.class_2_ix[l] for l in self.orig_labels])

        self.dataset = self.make_dataset(train_directory, self.class_2_ix)
        self.dataset_name = self._dataset_npz['repr_data']

        _indicator = self._dataset_npz['indicator_data']
        self.selected_ixs_for_noisy = np.where(_indicator == -3)[0]
        self.selected_ixs_for_atypical = np.where(_indicator == -2)[0]
        self.selected_ixs_for_typical = np.where(_indicator == -1)[0]

        self.selected_ixs_for_noisy = self.selected_ixs_for_noisy[self.selected_ixs_for_noisy<50000]
        self.selected_ixs_for_atypical = self.selected_ixs_for_atypical[self.selected_ixs_for_atypical<50000]
        self.selected_ixs_for_typical = self.selected_ixs_for_typical[self.selected_ixs_for_typical<50000]

        self.class_ixs = [np.where(self.labels == c)[0] for c in range(len(classes))]
        self.num_of_dupes = self._dataset_npz['num_dupes_data'].item()

    def __getitem__(self, ix):
        path, target = self.dataset[ix]
        sample = self.loader(path)
        if self.transforms:
            data = self.transforms(sample)
        else:
            data = sample
        return ix, data, self.class_2_ix[target]

    def __len__(self):
        return len(self.dataset)

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        print("Classes Found : {0}".format(len(classes)))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def make_dataset(self, directory: str, class_2_ix: Dict[str, int]):
        dataset = [(os.path.join(directory, str(d[0]).split('_')[0], str(d[0])), str(d[1])) for d in
                   self._dataset_npz['filenames']][:50000]
        return dataset

    def __repr__(self):
        _num_of_noisy_ixs = len(self.selected_ixs_for_noisy)
        _num_of_atypical_ixs = len(self.selected_ixs_for_atypical)
        _num_of_typical_ixs = len(self.selected_ixs_for_typical) // self.num_of_dupes
        _num_of_typical_dupes = self.num_of_dupes

        _noisy_pct = (len(self.selected_ixs_for_noisy) / len(self.dataset)) * 100
        _atypical_pct = (len(self.selected_ixs_for_atypical) / len(self.dataset)) * 100
        _typical_pct = (len(self.selected_ixs_for_typical) / len(self.dataset)) * 100

        if self.num_of_dupes == 1:
            # C-Score Dataset
            repr_str = f'*** {self.__class__.__name__}({self.dataset_name}): {len(self.dataset)} ***\t\nNoisy({_noisy_pct:.2f}%)-> {_num_of_noisy_ixs} Unique Images with Shuffled Labels \nAtypical({_atypical_pct:.2f}%)-> {_num_of_atypical_ixs} Unique Images with Lowest C-Scores\nTypical({_typical_pct:.2f}%)-> {_num_of_typical_ixs} Remaining Unique Images'
        else:
            # Frequency Based Dataset
            repr_str = f'*** {self.__class__.__name__}({self.dataset_name}): {len(self.dataset)} ***\t\nNoisy({_noisy_pct:.2f}%)-> {_num_of_noisy_ixs} Unique Images with Shuffled Labels \nAtypical({_atypical_pct:.2f}%)-> {_num_of_atypical_ixs} Unique Images with Original Labels\nTypical({_typical_pct:.2f}%)-> {_num_of_typical_ixs} Unique Images with Original Labels X {_num_of_typical_dupes} Copies'

        return repr_str

class LONGTAIL_IMAGENET_DYNAMIC(DatasetFolder):

    def __init__(self, train_directory, dataset_npz, augment_indicator, num_additional_copies=0):

        self.augment_indicator = augment_indicator
        self.num_additional_copies = num_additional_copies
        self._dataset_npz = np.load(dataset_npz, allow_pickle=True)
        classes, self.class_2_ix = self.find_classes(train_directory)
        # Get Original Dataset without any transform/augmentation
        _orig_dataset = LONGTAIL_IMAGENET(train_directory=train_directory, dataset_npz=dataset_npz, apply_transform=False, apply_augmentation=False).dataset

        self.augment_and_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )

        # Collect the Ixs for __getitem__
        self.ixs_to_augment = np.where(self.augment_indicator == 1)[0]

        expanded_dataset = []
        for ix in self.ixs_to_augment:
            expanded_dataset.extend([_orig_dataset[ix] for _ in range(self.num_additional_copies)])

        assert len(expanded_dataset) == np.sum(self.augment_indicator) * self.num_additional_copies, len(
            expanded_dataset
        )

        # Add the newly added ixs
        self.ixs_to_augment = np.concatenate((self.ixs_to_augment,np.arange(len(_orig_dataset),(len(_orig_dataset)+len(self.ixs_to_augment)))))

        self.dataset = _orig_dataset + expanded_dataset

        # shuffle so added images aren't all at the end
        self.shuffled_ix_mapping = np.random.permutation(len(self.dataset))

    def __getitem__(self, ix):
        added_ix = self.shuffled_ix_mapping[ix]
        path, target = self.dataset[added_ix]

        sample = self.loader(path)
        # Check if the ix is of interest to augment
        if added_ix in self.ixs_to_augment:
            data = self.augment_and_transform(sample)
        else:
            data = self.transform(sample)

        return added_ix, data, self.class_2_ix[target]

    def __len__(self):
        return len(self.dataset)

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        print("Classes Found : {0}".format(len(classes)))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def make_dataset(self, directory: str, class_2_ix: Dict[str, int]):
        dataset = [(os.path.join(directory, str(d[0]).split('_')[0], str(d[0])), str(d[1])) for d in
                   self._dataset_npz['filenames']][:50000]
        return dataset