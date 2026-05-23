# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import random
from pathlib import Path

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import torch.distributed as dist


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def generate_subset_file(root, subset_file, percentage, seed=42):
    """
    Dynamically generate an ImageNet subset list file with stratified sampling.
    Ensures reproducibility via a fixed seed.
    """
    print(f"Generating subset list for {percentage*100}% data...")
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data root {root} does not exist.")

    # 1. Scan classes
    classes = [d for d in root.iterdir() if d.is_dir()]
    classes.sort()  # Ensure consistent order

    if not classes:
        raise RuntimeError(f"No class directories found in {root}")

    subset_files = []

    # Use a local random instance to avoid affecting global state
    rng = random.Random(seed)

    for i, class_dir in enumerate(classes):
        images = sorted([f.name for f in class_dir.glob("*.JPEG")])
        if not images:
            continue

        # Stratified sampling
        k = max(1, int(len(images) * percentage))

        # Shuffle deterministically per class
        # We use a class-specific seed derived from the base seed to ensure
        # that adding/removing classes doesn't change the selection of others
        # (though ImageNet classes are fixed, this is good practice)
        # class_seed = seed + hash(class_dir.name) % 100000
        class_seed = seed + i
        class_rng = random.Random(class_seed)

        # Create a copy to shuffle
        shuffled_images = list(images)
        class_rng.shuffle(shuffled_images)

        selected = shuffled_images[:k]
        subset_files.extend(selected)

    # Save to file
    subset_file.parent.mkdir(parents=True, exist_ok=True)
    with open(subset_file, "w") as f:
        for filename in subset_files:
            f.write(filename + "\n")

    print(f"Generated subset list with {len(subset_files)} images at {subset_file}")


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000

        subset_name = getattr(args, 'imagenet_train_subset', 'full')
        if is_train and subset_name and subset_name.lower() != 'full':
            subset_name = subset_name.lower()
            subset_dir = Path(__file__).resolve().parent / 'imagenet_subsets'
            subset_file = subset_dir / f'{subset_name}.txt'

            # Dynamic generation logic
            if not subset_file.is_file() and "percent" in subset_name:
                # Only rank 0 generates the file in distributed settings
                if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                    try:
                        # Extract percentage from name like "20percent" -> 0.2
                        pct_str = subset_name.replace("percent", "")
                        percentage = float(pct_str) / 100.0
                        if 0 < percentage < 1:
                            # Use a fixed seed (e.g. 42) to ensure the subset is always the same
                            # regardless of the training seed
                            generate_subset_file(root, subset_file, percentage, seed=42)
                    except ValueError:
                        print(f"Could not parse percentage from {subset_name}, skipping generation.")

                # Wait for rank 0 to finish generation
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

            if not subset_file.is_file():
                raise FileNotFoundError(f"ImageNet subset list not found: {subset_file}")
            with subset_file.open('r') as handle:
                keep_files = {line.strip() for line in handle if line.strip()}
            if not keep_files:
                raise RuntimeError(f"Subset file {subset_file} is empty")

            original_samples = len(dataset.samples)
            filtered_samples = [
                (path, target)
                for (path, target) in dataset.samples
                if os.path.basename(path) in keep_files
            ]
            if not filtered_samples:
                raise RuntimeError(
                    f"No ImageNet samples matched subset '{subset_name}'. "
                    "Ensure the subset list corresponds to files under the training root."
                )
            dataset.samples = filtered_samples
            dataset.imgs = filtered_samples
            if hasattr(dataset, 'targets'):
                dataset.targets = [target for _, target in filtered_samples]
            kept = len(filtered_samples)
            print(
                f"Using ImageNet {subset_name} subset: retained {kept} / {original_samples} training samples"
            )
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
