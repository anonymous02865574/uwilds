# Code for ICLR 2022 submission: Extending the WILDS Benchmark for Unsupervised Adaptation

The U-WILDS repository extends the WILDS package with data loaders for the new unlabeled data;
dataset preprocessing scripts for the unlabeled data;
and compatible implementations of all of the methods that we benchmarked in the U-WILDS submission.

The URLs to download the datasets have been removed to maintain anonymity.

All of the U-WILDS unlabeled data loaders are compatible with the original WILDS labeled data
loaders, as illustrated in this code snippet:
```py
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms

# Load the labeled data
dataset = get_dataset(dataset="fmow", download=True)
labeled_subset = dataset.get_subset("train", transform=transforms.ToTensor())
data_loader = get_train_loader("standard", labeled_subset, batch_size=16)

# Load the unlabeled data
dataset = get_dataset(dataset="fmow", unlabeled=True, download=True)
unlabeled_subset = dataset.get_subset("test_unlabeled", transform=transforms.ToTensor())
unlabeled_data_loader = get_train_loader("standard", unlabeled_subset, batch_size=64)

# Train loop
for labeled_batch, unlabeled_batch in zip(data_loader, unlabeled_data_loader):
   x, y, metadata = labeled_batch
   unlabeled_x, unlabeled_metadata = unlabeled_batch
```
