# Libraries

# Imports
from classes import LONGTAIL_IMAGENET


dataset = LONGTAIL_IMAGENET(train_directory="imagenet/train/", dataset_npz='N20_A20_T60.npz',apply_augmentation=False,apply_transform=False)

print("Length of Dataset : {0}".format(len(dataset)))

print("Noisy Ixs : {0}".format(dataset.selected_ixs_for_noisy[:10]))

print("Atypical Ixs : {0}".format(dataset.selected_ixs_for_atypical[:10]))

print("Typical Ixs : {0}".format(dataset.selected_ixs_for_typical[:10]))

