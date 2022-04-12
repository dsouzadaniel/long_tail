# Libraries
import os
import numpy as np
from typing import List, Tuple
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
# Class Definitions

# LONGTAIL Dataset Class
import torchvision
from torchvision import transforms

#######################################################################################
class CIFAR10(Dataset):
    def __init__(self, apply_transform=True, apply_augmentation=False):
        augmentation = (
            transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            if apply_augmentation == True
            else transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        )

        self.transforms = augmentation if apply_transform == True else None

        self.dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=None)
        self.dataset_name = repr(self.dataset)

        _labels = np.array(self.dataset.targets)

        self.classes = self.dataset.classes

        self.class_ixs = np.stack(
            [np.where(_labels == c)[0] for c in range(len(self.classes))]
        )

    def __getitem__(self, ix):
        data, target = self.dataset[ix]
        if self.transforms:
            data = self.transforms(data)
        return ix, data, target

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        repr_str = f'*** {self.__class__.__name__}({self.dataset_name})'
        return repr_str

class CIFAR10_DYNAMIC(Dataset):

    def __init__(self, augment_indicator, num_additional_copies=0):
        """A dataset class that performs Selective Augmentation on CIFAR10

        Args:
                augment_indicator: A 1-hot numpy array indicating which images to augment
                num_additional_copies: Number of additional augmented copies to add back to the dynamic dataset
        """
        self.augment_indicator = augment_indicator
        self.num_additional_copies = num_additional_copies

        self.augment_and_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        # Collect the Ixs for __getitem__
        self.ixs_to_augment = np.where(self.augment_indicator == 1)[0]

        # Get Original Dataset without any transform/augmentation
        _orig_dataset = CIFAR10(apply_augmentation=False).dataset
        self.dataset = self.make_dataset(base_dataset=_orig_dataset)

        # shuffle so added images aren't all at the end
        self.shuffled_ix_mapping = np.random.permutation(len(self.dataset))

    def make_dataset(self, base_dataset: List[Tuple]):

        expanded_dataset = []
        for ix in self.ixs_to_augment:
            expanded_dataset.extend([base_dataset[ix] for _ in range(self.num_additional_copies)])

        assert len(expanded_dataset) == np.sum(self.augment_indicator) * self.num_additional_copies, len(expanded_dataset)

         # Add the newly added ixs
        self.ixs_to_augment = np.concatenate(
            (self.ixs_to_augment, np.arange(len(base_dataset), (len(base_dataset) + len(self.ixs_to_augment)))))

        dataset = base_dataset + expanded_dataset
        return dataset

    def make_dataset_new_labels(self, new_labels:List[str]):
        assert len(self.dataset) == len(new_labels), "New Labels aren't the same size as the dataset!"
        self.dataset = [(old_path, new_label) for (old_path, old_label),new_label in zip(self.dataset, new_labels)
                        ]
        print("New Labels Loaded!")
        return

    def __getitem__(self, ix):
        added_ix = self.shuffled_ix_mapping[ix]
        image, target = self.dataset[added_ix]

        # Check if the ix is of interest to augment
        if added_ix in self.ixs_to_augment:
            data = self.augment_and_transform(image)
        else:
            data = self.transform(image)

        return added_ix, data, target

    def __len__(self):
        return len(self.dataset)

class LONGTAIL_CIFAR10(Dataset):
    def __init__(self, dataset_npz, apply_transform=True, apply_augmentation=False):
        augmentation = (
            transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            if apply_augmentation == True
            else transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        )

        self.transforms = augmentation if apply_transform == True else None

        _dataset_npz = np.load(dataset_npz)

        self.dataset = self.make_dataset(_dataset_npz)

        _labels = np.array(_dataset_npz['label_data'])
        self.dataset_name = _dataset_npz['repr_data']

        self.num_of_dupes = _dataset_npz['num_dupes_data']

        _indicator = _dataset_npz['indicator_data']
        self.selected_ixs_for_noisy = np.where(_indicator == -3)[0]
        self.selected_ixs_for_atypical = np.where(_indicator == -2)[0]
        self.selected_ixs_for_typical = np.where(_indicator == -1)[0]

        self.classes = ['airplane',
                        'automobile',
                        'bird',
                        'cat',
                        'deer',
                        'dog',
                        'frog',
                        'horse',
                        'ship',
                        'truck']

        self.class_ixs = np.stack(
            [np.where(_labels == c)[0] for c in range(len(self.classes))]
        )

    def make_dataset(self, _dataset_npz):
        dataset = [(Image.fromarray(image_array), label) for image_array, label in
         zip(list(_dataset_npz['image_data']), _dataset_npz['label_data'])]
        return dataset

    def __getitem__(self, ix):
        data, label = self.dataset[ix]
        if self.transforms:
            data = self.transforms(data)
        return ix, data, label

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        _num_of_noisy_ixs = len(self.selected_ixs_for_noisy)
        _num_of_atypical_ixs = len(self.selected_ixs_for_atypical)
        _num_of_typical_ixs = len(self.selected_ixs_for_typical) // self.num_of_dupes
        _num_of_typical_dupes = self.num_of_dupes

        _noisy_pct = (len(self.selected_ixs_for_noisy) / len(self.dataset)) * 100
        _atypical_pct = (len(self.selected_ixs_for_atypical) / len(self.dataset)) * 100
        _typical_pct = (len(self.selected_ixs_for_typical) / len(self.dataset)) * 100

        if self.num_of_dupes==1:
            # C-Score Dataset
            repr_str = f'*** {self.__class__.__name__}({self.dataset_name}): {len(self.dataset)} ***\t\nNoisy({_noisy_pct}%)-> {_num_of_noisy_ixs} Unique Images with Shuffled Labels \nAtypical({_atypical_pct}%)-> {_num_of_atypical_ixs} Unique Images with Lowest C-Scores\nTypical({_typical_pct}%)-> {_num_of_typical_ixs} Remaining Unique Images'
        else:
            # Frequency Based Dataset
            repr_str = f'*** {self.__class__.__name__}({self.dataset_name}): {len(self.dataset)} ***\t\nNoisy({_noisy_pct}%)-> {_num_of_noisy_ixs} Unique Images with Shuffled Labels \nAtypical({_atypical_pct}%)-> {_num_of_atypical_ixs} Unique Images with Original Labels\nTypical({_typical_pct}%)-> {_num_of_typical_ixs} Unique Images with Original Labels X {_num_of_typical_dupes} Copies'

        return repr_str

class LONGTAIL_CIFAR10_DYNAMIC(Dataset):

    def __init__(self, dataset_npz, augment_indicator, num_additional_copies=0):
        """A dataset class that performs Selective Augmentation on LONGTAIL_CIFAR10

        Args:
                augment_indicator: A 1-hot numpy array indicating which images to augment
                num_additional_copies: Number of additional augmented copies to add back to the dynamic dataset
        """
        self.augment_indicator = augment_indicator
        self.num_additional_copies = num_additional_copies

        self.augment_and_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        # Collect the Ixs for __getitem__
        self.ixs_to_augment = np.where(self.augment_indicator==1)[0]

        # Get Original Dataset without any transform/augmentation
        _orig_dataset = LONGTAIL_CIFAR10(dataset_npz=dataset_npz, apply_transform=False, apply_augmentation=False).dataset
        self.dataset = self.make_dataset(base_dataset = _orig_dataset)


        # shuffle so added images aren't all at the end
        self.shuffled_ix_mapping = np.random.permutation(len(self.dataset))

    def make_dataset(self, base_dataset: List[Tuple]):

        expanded_dataset = []
        for ix in self.ixs_to_augment:
            expanded_dataset.extend([base_dataset[ix] for _ in range(self.num_additional_copies)])

        assert len(expanded_dataset) == np.sum(self.augment_indicator) * self.num_additional_copies, len(expanded_dataset)

         # Add the newly added ixs
        self.ixs_to_augment = np.concatenate(
            (self.ixs_to_augment, np.arange(len(base_dataset), (len(base_dataset) + len(self.ixs_to_augment)))))

        dataset = base_dataset + expanded_dataset
        return dataset

    def make_dataset_new_labels(self, new_labels:List[str]):
        assert len(self.dataset) == len(new_labels), "New Labels aren't the same size as the dataset!"
        _new_base_dataset = [(old_path, new_label) for (old_path, old_label), new_label in zip(self.dataset, new_labels)
                        ]
        self.dataset = self.make_dataset(base_dataset=_new_base_dataset)
        print("New Labels Loaded!")
        return

    def __getitem__(self, ix):
        added_ix = self.shuffled_ix_mapping[ix]
        image, label = self.dataset[added_ix]

        # Check if the ix is of interest to augment
        if added_ix in self.ixs_to_augment:
            data = self.augment_and_transform(image)
        else:
            data = self.transform(image)

        return added_ix, data, label

    def __len__(self):
        return len(self.dataset)

class LONGTAIL_CIFAR10_DYNAMIC_TEMP(Dataset):

    def __init__(self, dataset_npz, augment_indicator, copy_indicator, num_additional_copies=0):
        """A dataset class that performs Selective Augmentation on LONGTAIL_CIFAR10

        Args:
                augment_indicator: A 1-hot numpy array indicating which images to augment
                copy_indicator: A 1-hot numpy array indicating which images to make copies of
                num_additional_copies: Number of additional augmented copies to add back to the dynamic dataset
        """
        self.augment_indicator = augment_indicator
        self.copy_indicator = copy_indicator
        self.num_additional_copies = num_additional_copies

        self.augment_and_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        # Collect the Ixs for __getitem__
        self.ixs_to_augment = np.where(self.augment_indicator==1)[0]

        print(f"TEMP A : {np.sum(self.copy_indicator)}")
        self.ixs_to_copy = np.where(self.copy_indicator == 1)[0]
        print(f"TEMP B : {self.ixs_to_copy.shape}")

        # Get Original Dataset without any transform/augmentation
        _orig_dataset = LONGTAIL_CIFAR10(dataset_npz=dataset_npz, apply_transform=False, apply_augmentation=False).dataset
        self.dataset = self.make_dataset(base_dataset = _orig_dataset)


        # shuffle so added images aren't all at the end
        self.shuffled_ix_mapping = np.random.permutation(len(self.dataset))

    def make_dataset(self, base_dataset: List[Tuple]):

        expanded_dataset = []
        for ix in self.ixs_to_copy:
            expanded_dataset.extend([base_dataset[ix] for _ in range(self.num_additional_copies)])


        print(f"IXS to COPY SUM : {np.sum(self.ixs_to_copy)}")
        print(f"NUM OF ADD COPIES : {self.num_additional_copies}")
        assert len(expanded_dataset) == np.sum(self.ixs_to_copy) * self.num_additional_copies, len(expanded_dataset)

         # Add the newly added ixs
        self.ixs_to_augment = np.concatenate(
            (self.ixs_to_augment, np.arange(len(base_dataset), len(expanded_dataset))))

        dataset = base_dataset + expanded_dataset

        print(f"Increased the Size of the Dataset from {len(base_dataset)} to {len(dataset)}")

        return dataset

    def make_dataset_new_labels(self, new_labels:List[str]):
        assert len(self.dataset) == len(new_labels), "New Labels aren't the same size as the dataset!"
        _new_base_dataset = [(old_path, new_label) for (old_path, old_label), new_label in zip(self.dataset, new_labels)
                        ]
        self.dataset = self.make_dataset(base_dataset=_new_base_dataset)
        print("New Labels Loaded!")
        return

    def __getitem__(self, ix):
        added_ix = self.shuffled_ix_mapping[ix]
        image, label = self.dataset[added_ix]

        # Check if the ix is of interest to augment
        if added_ix in self.ixs_to_augment:
            data = self.augment_and_transform(image)
        else:
            data = self.transform(image)

        return added_ix, data, label

    def __len__(self):
        return len(self.dataset)



class CIFAR10_TEST(Dataset):
    def __init__(self):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transforms
        )
        self.classes = self.dataset.classes

        _labels = np.array(self.dataset.targets)

        self.class_ixs = np.stack(
            [np.where(_labels == c)[0] for c in range(len(self.classes))]
        )


    def __getitem__(self, ix):
        data, target = self.dataset[ix]
        return ix, data, target

    def __len__(self):
        return len(self.dataset)

#######################################################################################
class CIFAR100(Dataset):
    def __init__(self, apply_transform=True, apply_augmentation=False):
        augmentation = (
            transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                    ),
                ]
            )
            if apply_augmentation == True
            else transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                    ),
                ]
            )
        )

        self.transforms = augmentation if apply_transform == True else None

        self.dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
        self.dataset_name = repr(self.dataset)

        _labels = np.array(self.dataset.targets)

        self.classes = list(self.dataset.classes)

        self.class_ixs = np.stack(
            [np.where(_labels == c)[0] for c in range(len(self.classes))]
        )

    def __getitem__(self, ix):
        data, target = self.dataset[ix]
        if self.transforms:
            data = self.transforms(data)
        return ix, data, target

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        repr_str = f'*** {self.__class__.__name__}({self.dataset_name})'
        return repr_str

class CIFAR100_DYNAMIC(Dataset):

    def __init__(self, augment_indicator, num_additional_copies=0):
        """A dataset class that performs Selective Augmentation on CIFAR100

        Args:
                augment_indicator: A 1-hot numpy array indicating which images to augment
                num_additional_copies: Number of additional augmented copies to add back to the dynamic dataset
        """
        self.augment_indicator = augment_indicator
        self.num_additional_copies = num_additional_copies

        self.augment_and_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        # Collect the Ixs for __getitem__
        self.ixs_to_augment = np.where(self.augment_indicator == 1)[0]


        # Get Original Dataset without any transform/augmentation
        _orig_dataset = CIFAR100(apply_augmentation=False).dataset
        self.dataset = self.make_dataset(base_dataset = _orig_dataset)

        # shuffle so added images aren't all at the end
        self.shuffled_ix_mapping = np.random.permutation(len(self.dataset))

    def make_dataset(self, base_dataset: List[Tuple]):

        expanded_dataset = []
        for ix in self.ixs_to_augment:
            expanded_dataset.extend([base_dataset[ix] for _ in range(self.num_additional_copies)])

        assert len(expanded_dataset) == np.sum(self.augment_indicator) * self.num_additional_copies, len(expanded_dataset)

         # Add the newly added ixs
        self.ixs_to_augment = np.concatenate(
            (self.ixs_to_augment, np.arange(len(base_dataset), (len(base_dataset) + len(self.ixs_to_augment)))))

        dataset = base_dataset + expanded_dataset
        return dataset

    def make_dataset_new_labels(self, new_labels:List[str]):
        assert len(self.dataset) == len(new_labels), "New Labels aren't the same size as the dataset!"
        self.dataset = [(old_path, new_label) for (old_path, old_label),new_label in zip(self.dataset, new_labels)
                        ]
        print("New Labels Loaded!")
        return

    def __getitem__(self, ix):
        added_ix = self.shuffled_ix_mapping[ix]
        image, label = self.dataset[added_ix]

        # Check if the ix is of interest to augment
        if added_ix in self.ixs_to_augment:
            data = self.augment_and_transform(image)
        else:
            data = self.transform(image)

        return added_ix, data, label

    def __len__(self):
        return len(self.dataset)

class LONGTAIL_CIFAR100(Dataset):
    def __init__(self, dataset_npz, apply_transform=True, apply_augmentation=False):
        augmentation = (
            transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                    ),
                ]
            )
            if apply_augmentation == True
            else transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                    ),
                ]
            )
        )

        self.transforms = augmentation if apply_transform == True else None

        _dataset_npz = np.load(dataset_npz)

        self.dataset = self.make_dataset(_dataset_npz)

        _labels = np.array(_dataset_npz['label_data'])
        self.dataset_name = _dataset_npz['repr_data']

        self.num_of_dupes = _dataset_npz['num_dupes_data']

        _indicator = _dataset_npz['indicator_data']
        self.selected_ixs_for_noisy = np.where(_indicator == -3)[0]
        self.selected_ixs_for_atypical = np.where(_indicator == -2)[0]
        self.selected_ixs_for_typical = np.where(_indicator == -1)[0]

        self.classes = ['apple',
                        'aquarium_fish',
                        'baby',
                        'bear',
                        'beaver',
                        'bed',
                        'bee',
                        'beetle',
                        'bicycle',
                        'bottle',
                        'bowl',
                        'boy',
                        'bridge',
                        'bus',
                        'butterfly',
                        'camel',
                        'can',
                        'castle',
                        'caterpillar',
                        'cattle',
                        'chair',
                        'chimpanzee',
                        'clock',
                        'cloud',
                        'cockroach',
                        'couch',
                        'crab',
                        'crocodile',
                        'cup',
                        'dinosaur',
                        'dolphin',
                        'elephant',
                        'flatfish',
                        'forest',
                        'fox',
                        'girl',
                        'hamster',
                        'house',
                        'kangaroo',
                        'keyboard',
                        'lamp',
                        'lawn_mower',
                        'leopard',
                        'lion',
                        'lizard',
                        'lobster',
                        'man',
                        'maple_tree',
                        'motorcycle',
                        'mountain',
                        'mouse',
                        'mushroom',
                        'oak_tree',
                        'orange',
                        'orchid',
                        'otter',
                        'palm_tree',
                        'pear',
                        'pickup_truck',
                        'pine_tree',
                        'plain',
                        'plate',
                        'poppy',
                        'porcupine',
                        'possum',
                        'rabbit',
                        'raccoon',
                        'ray',
                        'road',
                        'rocket',
                        'rose',
                        'sea',
                        'seal',
                        'shark',
                        'shrew',
                        'skunk',
                        'skyscraper',
                        'snail',
                        'snake',
                        'spider',
                        'squirrel',
                        'streetcar',
                        'sunflower',
                        'sweet_pepper',
                        'table',
                        'tank',
                        'telephone',
                        'television',
                        'tiger',
                        'tractor',
                        'train',
                        'trout',
                        'tulip',
                        'turtle',
                        'wardrobe',
                        'whale',
                        'willow_tree',
                        'wolf',
                        'woman',
                        'worm']

        self.class_ixs = np.stack(
            [np.where(_labels == c)[0] for c in range(len(self.classes))]
        )

    def make_dataset(self, _dataset_npz):
        dataset = [(Image.fromarray(image_array), label) for image_array, label in
         zip(list(_dataset_npz['image_data']), _dataset_npz['label_data'])]
        return dataset

    def __getitem__(self, ix):
        data, target = self.dataset[ix]
        if self.transforms:
            data = self.transforms(data)
        return ix, data, target

    def __len__(self):
        return len(self.dataset)

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
            repr_str = f'*** {self.__class__.__name__}({self.dataset_name}): {len(self.dataset)} ***\t\nNoisy({_noisy_pct}%)-> {_num_of_noisy_ixs} Unique Images with Shuffled Labels \nAtypical({_atypical_pct}%)-> {_num_of_atypical_ixs} Unique Images with Lowest C-Scores\nTypical({_typical_pct}%)-> {_num_of_typical_ixs} Remaining Unique Images'
        else:
            # Frequency Based Dataset
            repr_str = f'*** {self.__class__.__name__}({self.dataset_name}): {len(self.dataset)} ***\t\nNoisy({_noisy_pct}%)-> {_num_of_noisy_ixs} Unique Images with Shuffled Labels \nAtypical({_atypical_pct}%)-> {_num_of_atypical_ixs} Unique Images with Original Labels\nTypical({_typical_pct}%)-> {_num_of_typical_ixs} Unique Images with Original Labels X {_num_of_typical_dupes} Copies'

        return repr_str

class LONGTAIL_CIFAR100_DYNAMIC(Dataset):

    def __init__(self, dataset_npz, augment_indicator, num_additional_copies=0):
        """A dataset class that performs Selective Augmentation on LONGTAIL_CIFAR100

        Args:
                augment_indicator: A 1-hot numpy array indicating which images to augment
                num_additional_copies: Number of additional augmented copies to add back to the dynamic dataset
        """
        self.augment_indicator = augment_indicator
        self.num_additional_copies = num_additional_copies

        self.augment_and_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        # Collect the Ixs for __getitem__
        self.ixs_to_augment = np.where(self.augment_indicator == 1)[0]

        # Get Original Dataset without any transform/augmentation
        _orig_dataset = LONGTAIL_CIFAR100(dataset_npz=dataset_npz, apply_transform=False,
                                          apply_augmentation=False).dataset
        self.dataset = self.make_dataset(base_dataset = _orig_dataset)

        # shuffle so added images aren't all at the end
        self.shuffled_ix_mapping = np.random.permutation(len(self.dataset))

    def make_dataset(self, base_dataset: List[Tuple]):

        expanded_dataset = []
        for ix in self.ixs_to_augment:
            expanded_dataset.extend([base_dataset[ix] for _ in range(self.num_additional_copies)])

        assert len(expanded_dataset) == np.sum(self.augment_indicator) * self.num_additional_copies, len(expanded_dataset)

         # Add the newly added ixs
        self.ixs_to_augment = np.concatenate(
            (self.ixs_to_augment, np.arange(len(base_dataset), (len(base_dataset) + len(self.ixs_to_augment)))))

        dataset = base_dataset + expanded_dataset
        return dataset

    def make_dataset_new_labels(self, new_labels:List[str]):
        assert len(self.dataset) == len(new_labels), "New Labels aren't the same size as the dataset!"
        self.dataset = [(old_path, new_label) for (old_path, old_label),new_label in zip(self.dataset, new_labels)
                        ]
        print("New Labels Loaded!")
        return

    def __getitem__(self, ix):
        added_ix = self.shuffled_ix_mapping[ix]
        image, label = self.dataset[added_ix]

        # Check if the ix is of interest to augment
        if added_ix in self.ixs_to_augment:
            data = self.augment_and_transform(image)
        else:
            data = self.transform(image)

        return added_ix, data, label

    def __len__(self):
        return len(self.dataset)

class CIFAR100_TEST(Dataset):
    def __init__(self):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        self.dataset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=self.transforms
        )
        self.classes = self.dataset.classes

    def __getitem__(self, ix):
        data, target = self.dataset[ix]
        return ix, data, target

    def __len__(self):
        return len(self.dataset)
#######################################################################################
