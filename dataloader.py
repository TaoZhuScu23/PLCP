import os

import torch
from torch import Tensor
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS, CoraFull
import torch_geometric.transforms as T


def load_dataset(dataset_name, transform=None):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.', 'data', dataset_name)

    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path, dataset_name, split="public", transform=T.NormalizeFeatures())
    elif dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['CS', 'Physics']:
        dataset = Coauthor(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['Wikics']:
        dataset = WikiCS(path)
    elif dataset_name in ['Corafull']:
        dataset = CoraFull(path, transform=T.NormalizeFeatures())
    data = dataset[0]

    if dataset_name in ['Computers', 'Photo', 'CS', 'Physics', 'Wikics', 'Corafull']:
        # data.train_mask, data.val_mask, data.test_mask = get_split(data.x.shape[0])
        # data.train_mask, data.val_mask, data.test_mask = split_dataset(data.y)
        # data.train_mask, data.val_mask, data.test_mask = random_split_dataset_v3(data.y,0.1,0.1,0.8)
        # data.train_mask, data.val_mask, data.test_mask = random_split_dataset_v2(data.y)
        data.train_mask, data.val_mask, data.test_mask = random_split_dataset(data.y)
    return data


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8, num_splits: int = 1):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)

    indices = torch.randperm(num_samples)

    train_mask = torch.zeros(num_samples, dtype=torch.bool)
    train_mask.fill_(False)
    train_mask[indices[:train_size]] = True

    test_mask = torch.zeros(num_samples, dtype=torch.bool)
    test_mask.fill_(False)
    test_mask[indices[train_size: test_size + train_size]] = True

    val_mask = torch.zeros(num_samples, dtype=torch.bool)
    val_mask.fill_(False)
    val_mask[indices[test_size + train_size:]] = True

    return train_mask, val_mask, test_mask


def split_dataset(label: Tensor):
    # Initialize counters and masks
    class_counts = torch.zeros(len(torch.unique(label)), dtype=torch.int64)
    train_mask = torch.zeros_like(label, dtype=torch.bool)

    last_train_index = 0
    # Count the number of samples for each class until we have 20 samples for each
    for idx, label_idx in enumerate(label):
        if class_counts[label_idx] < 20:
            train_mask[idx] = True
            class_counts[label_idx] += 1

        # If all classes have reached the desired count, break the loop
        if (class_counts == 20).all():
            last_train_index = idx
            break

    # Define the validation and test masks
    val_mask = torch.zeros_like(label, dtype=torch.bool)
    test_mask = torch.zeros_like(label, dtype=torch.bool)

    # Validation set starts from the next index after the last training sample
    val_start_index = last_train_index + 1
    val_end_index = min(val_start_index + 500, len(label))
    val_mask[val_start_index:val_end_index] = True

    # Test set takes the last 1000 samples
    test_mask[-1000:] = True

    return train_mask, val_mask, test_mask


def random_split_dataset(label: torch.Tensor):
    unique_labels = label.unique()
    num_classes = len(unique_labels)

    # Initialize masks
    train_mask = torch.zeros_like(label, dtype=torch.bool)
    val_mask = torch.zeros_like(label, dtype=torch.bool)
    test_mask = torch.zeros_like(label, dtype=torch.bool)

    # Randomly select 20 samples per class for training
    for class_label in unique_labels:
        class_indices = (label == class_label).nonzero(as_tuple=True)[0]
        selected_indices = class_indices[torch.randperm(len(class_indices))[:20]]
        train_mask[selected_indices] = True

    # Remaining indices after selecting training samples
    remaining_indices = (~train_mask).nonzero(as_tuple=True)[0]

    # Randomly select 500 samples for validation
    val_indices = remaining_indices[torch.randperm(len(remaining_indices))[:500]]
    val_mask[val_indices] = True

    # Remaining indices after selecting validation samples
    remaining_indices = (~train_mask & ~val_mask).nonzero(as_tuple=True)[0]

    # Randomly select 1000 samples for testing
    test_indices = remaining_indices[torch.randperm(len(remaining_indices))[:1000]]
    test_mask[test_indices] = True

    # Ensure no overlap between sets
    assert not (train_mask & val_mask).any(), "Overlap found between training and validation sets."
    assert not (train_mask & test_mask).any(), "Overlap found between training and test sets."
    assert not (val_mask & test_mask).any(), "Overlap found between validation and test sets."

    return train_mask, val_mask, test_mask


def random_split_dataset_v2(label: torch.Tensor):
    unique_labels = label.unique()
    num_classes = len(unique_labels)

    # Initialize masks
    train_mask = torch.zeros_like(label, dtype=torch.bool)
    val_mask = torch.zeros_like(label, dtype=torch.bool)
    test_mask = torch.zeros_like(label, dtype=torch.bool)

    # Randomly select 20 samples per class for training
    for class_label in unique_labels:
        class_indices = (label == class_label).nonzero(as_tuple=True)[0]
        selected_indices = class_indices[torch.randperm(len(class_indices))[:20]]
        train_mask[selected_indices] = True

    # Remaining indices after selecting training samples
    remaining_indices = (~train_mask).nonzero(as_tuple=True)[0]

    # Randomly select 500 samples for validation
    val_indices = remaining_indices[torch.randperm(len(remaining_indices))[:500]]
    val_mask[val_indices] = True

    # Remaining indices after selecting validation samples
    remaining_indices = (~train_mask & ~val_mask).nonzero(as_tuple=True)[0]

    # All remaining nodes as test set
    test_mask[remaining_indices] = True

    # Ensure no overlap between sets
    assert not (train_mask & val_mask).any(), "Overlap found between training and validation sets."
    assert not (train_mask & test_mask).any(), "Overlap found between training and test sets."
    assert not (val_mask & test_mask).any(), "Overlap found between validation and test sets."

    return train_mask, val_mask, test_mask


def random_split_dataset_v3(label: torch.Tensor, train_ratio: float, val_ratio: float, test_ratio: float,
                            seed: int = None):
    """
    Split dataset into training, validation, and test sets based on given ratios.

    Parameters:
    - label: Tensor containing the labels of the dataset.
    - num_classes: Number of classes in the dataset.
    - train_ratio: Ratio of the dataset to be used as training set.
    - val_ratio: Ratio of the dataset to be used as validation set.
    - test_ratio: Ratio of the dataset to be used as test set.
    - seed: Seed for reproducibility. Default is None (no specific seed).

    Returns:
    - train_mask: Boolean mask indicating the indices for the training set.
    - val_mask: Boolean mask indicating the indices for the validation set.
    - test_mask: Boolean mask indicating the indices for the test set.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of ratios must equal 1."

    # Initialize masks
    train_mask = torch.zeros_like(label, dtype=torch.bool)
    val_mask = torch.zeros_like(label, dtype=torch.bool)
    test_mask = torch.zeros_like(label, dtype=torch.bool)

    # Shuffle indices
    shuffled_indices = torch.randperm(len(label))

    # Calculate the sizes of each set
    total_samples = len(label)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size

    # Assign indices to each set
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]

    # Create masks
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # Ensure no overlap between sets
    assert not (train_mask & val_mask).any(), "Overlap found between training and validation sets."
    assert not (train_mask & test_mask).any(), "Overlap found between training and test sets."
    assert not (val_mask & test_mask).any(), "Overlap found between validation and test sets."

    # Ensure all samples are assigned to a set
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == total_samples, "Not all samples were assigned to a set."

    return train_mask, val_mask, test_mask
