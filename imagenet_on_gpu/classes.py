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

# Imports
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
        classes, class_2_ix = self.find_classes(train_directory)

        self.ix_2_class = {v: k for k, v in class_2_ix.items()}
        self.class_2_text = self._dataset_npz['class_mapping'].item()

        self.labels = np.array([class_2_ix[l] for l in self.orig_labels])

        self.dataset = self.make_dataset(train_directory, class_2_ix)
        self.dataset_name = self._dataset_npz['repr_data']

        _indicator = self._dataset_npz['indicator_data']
        self.selected_ixs_for_noisy = np.where(_indicator == -3)[0]
        self.selected_ixs_for_atypical = np.where(_indicator == -2)[0]
        self.selected_ixs_for_typical = np.where(_indicator == -1)[0]

        self.class_ixs = [np.where(self.labels == c)[0] for c in range(len(classes))]
        self.num_of_dupes = self._dataset_npz['num_dupes_data'].item()

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

    def make_dataset(self, directory: str, class_2_ix: Dict[str, int]):
        dataset = [(os.path.join(directory, str(d[0]).split('_')[0], str(d[0])), str(d[1])) for d in
                   self._dataset_npz['filenames']]
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