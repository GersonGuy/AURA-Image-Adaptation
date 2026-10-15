
import os
import json
import torch
import logging
from typing import Optional, Sequence

from robustbench.data import CORRUPTIONS, PREPROCESSINGS, load_cifar10c, load_cifar100c
from robustbench.loaders import CustomImageFolder, CustomCifarDataset
import numpy as np
import torchvision
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def create_cifarc_dataset(
    dataset_name: str = 'cifar10_c',
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    domain = []
    x_test = torch.tensor([])
    y_test = torch.tensor([])
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]

    for cor in corruptions_seq:
        if dataset_name == 'cifar10_c':
            x_tmp, y_tmp = load_cifar10c(severity=severity,
                                         data_dir=data_dir,
                                         corruptions=[cor])
        elif dataset_name == 'cifar100_c':
            x_tmp, y_tmp = load_cifar100c(severity=severity,
                                          data_dir=data_dir,
                                          corruptions=[cor])
        else:
            raise ValueError(f"Dataset {dataset_name} is not suported!")

        x_test = torch.cat([x_test, x_tmp], dim=0)
        y_test = torch.cat([y_test, y_tmp], dim=0)
        domain += [cor] * x_tmp.shape[0]

    x_test = x_test.numpy().transpose((0, 2, 3, 1))
    y_test = y_test.numpy()
    samples = [[x_test[i], y_test[i], domain[i]] for i in range(x_test.shape[0])]

    return CustomCifarDataset(samples=samples, transform=transform)


def create_imagenetc_dataset(
    n_examples: Optional[int] = -1,
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    # create the dataset which loads the default test list from robust bench containing 5000 test samples
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]
    corruption_dir_path = os.path.join(data_dir, corruptions_seq[0], str(severity))
    dataset_test = CustomImageFolder(corruption_dir_path, transform)

    if "mixed_domains" in setting or "correlated" in setting or n_examples != -1:
        # load imagenet class to id mapping from robustbench
        with open(os.path.join("robustbench", "data", "imagenet_class_to_id_map.json"), 'r') as f:
            class_to_idx = json.load(f)

        if n_examples != -1 or "correlated" in setting:
            # create file path of file containing all 50k image ids
            file_path = os.path.join("datasets", "imagenet_list", "imagenet_val_ids_50k.txt")
        else:
            # create file path of default test list from robustbench
            file_path = os.path.join("robustbench", "data", "imagenet_test_image_ids.txt")

        # load file containing file ids
        with open(file_path, 'r') as f:
            fnames = f.readlines()

        item_list = []
        for cor in corruptions_seq:
            corruption_dir_path = os.path.join(data_dir, cor, str(severity))
            item_list += [(os.path.join(corruption_dir_path, fn.split('\n')[0]), class_to_idx[fn.split(os.sep)[0]]) for fn in fnames]
        dataset_test.samples = item_list

    return dataset_test


class GradualMixedCifarCDataset(Dataset):
    def __init__(self, dataset_name, severity, data_dir, corruptions_seq, transform, batch_size, total_frames=50, rng_seed=1, schedule_type="timeline", random_max_step=0.05, alpha_seed=-1, waypoint_speed=0.03, waypoint_arrival_tol=0.02, waypoint_concentration=0.5, waypoint_noise_scale=0.002):
        self.dataset_name = dataset_name
        self.severity = severity
        self.data_dir = data_dir
        self.corruptions_seq = corruptions_seq
        self.transform = transform
        self.batch_size = batch_size
        self.total_frames = total_frames
        self.rng_seed = rng_seed
        self.schedule_type = schedule_type
        self.random_max_step = random_max_step
        self.alpha_seed = alpha_seed
        self.waypoint_speed = waypoint_speed
        self.waypoint_arrival_tol = waypoint_arrival_tol
        self.waypoint_concentration = waypoint_concentration
        self.waypoint_noise_scale = waypoint_noise_scale
        
        # Load clean CIFAR data (Test set only exactly aligning with robustbench arrays)
        if dataset_name == 'cifar10_c':
            clean_ds = torchvision.datasets.CIFAR10(root=os.path.dirname(data_dir), train=False, download=True)
        elif dataset_name == 'cifar100_c':
            clean_ds = torchvision.datasets.CIFAR100(root=os.path.dirname(data_dir), train=False, download=True)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported for gradual mixed.")
            
        self.clean_x = clean_ds.data  # [N, 32, 32, 3]
        self.y = clean_ds.targets
        
        self.num_samples = self.total_frames * self.batch_size
        self.num_corruptions = len(corruptions_seq)
        self.domain = "gradual_mixed"
        
        # Load all requested corrupted arrays unconditionally (RobustBench ships test sets only)
        self.corr_x_list = []
        for cor in corruptions_seq:
            if dataset_name == 'cifar10_c':
                x_tmp, _ = load_cifar10c(severity=severity, data_dir=data_dir, corruptions=[cor])
            else:
                x_tmp, _ = load_cifar100c(severity=severity, data_dir=data_dir, corruptions=[cor])
            assert np.array_equal(np.array(self.y), _.numpy()), f"Label mismatch for corruption {cor}"
            self.corr_x_list.append(x_tmp.numpy().transpose((0, 2, 3, 1)))
                
        # --- Pre-seeded reproducible sequence generation ---
        # 1. Image sampling RNG
        self.image_rng_seed = self.rng_seed
        self.image_rng = np.random.default_rng(self.image_rng_seed)
        self.image_indices = self.image_rng.integers(0, len(self.y), size=self.num_samples)
        
        # 2. Alpha drift RNG
        if self.alpha_seed == -1:
            import secrets
            self.alpha_rng_seed = secrets.randbelow(2**32 - 1)
        else:
            self.alpha_rng_seed = self.alpha_seed
        self.alpha_rng = np.random.default_rng(self.alpha_rng_seed)
        
        # Pre-calculate alpha_schedule for reproducible saving and consistent per-frame access
        if self.schedule_type == "timeline" or self.schedule_type == "linear_overlap":
            self.alpha_schedule = self._generate_timeline_schedule()
        elif self.schedule_type == "random":
            self.alpha_schedule = self._generate_random_schedule()
        else:
            raise NotImplementedError(f"Schedule type '{self.schedule_type}' not supported.")
        
        # Save metadata dict for export
        self.metadata = {
            "schedule_type": self.schedule_type,
            "image_rng_seed": self.image_rng_seed,
            "alpha_rng_seed": self.alpha_rng_seed,
            "corruptions_seq": self.corruptions_seq,
            "severity": self.severity,
            "total_frames": self.total_frames,
            "batch_size": self.batch_size
        }

    @staticmethod
    def _project_simplex(v):
        """Project vector v onto the probability simplex."""
        n = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta, 0)

    def _generate_timeline_schedule(self):
        N = self.num_corruptions + 1
        schedule = np.zeros((self.total_frames, N), dtype=np.float32)
        for batch_idx in range(self.total_frames):
            center = (batch_idx / max(1, self.total_frames - 1)) * (N - 1)
            alphas = np.zeros(N, dtype=np.float32)
            for j in range(N):
                dist = abs(center - j)
                alphas[j] = max(0.0, 1.0 - dist)
            schedule[batch_idx] = alphas / np.sum(alphas)
        return schedule

    def _generate_random_schedule(self):
        N = self.num_corruptions + 1
        schedule = np.zeros((self.total_frames, N), dtype=np.float32)
        
        # Frame 0: Start entirely clean (index 0)
        current_alpha = np.zeros(N, dtype=np.float32)
        current_alpha[0] = 1.0
        schedule[0] = current_alpha
        
        # First target
        target = self.alpha_rng.dirichlet(np.ones(N) * self.waypoint_concentration).astype(np.float32)
        
        for t in range(1, self.total_frames):
            direction = target - current_alpha
            distance = np.linalg.norm(direction)
            
            # 1. Select new target if close enough
            if distance < self.waypoint_arrival_tol:
                target = self.alpha_rng.dirichlet(np.ones(N) * self.waypoint_concentration).astype(np.float32)
                direction = target - current_alpha
                distance = np.linalg.norm(direction)
                
            # 2. Compute bounded step
            if distance > 0:
                step_size = min(self.waypoint_speed, float(distance))
                step = (step_size / distance) * direction
            else:
                step = np.zeros(N, dtype=np.float32)
                
            # 3. Add optional micro-noise
            noise = (self.alpha_rng.standard_normal(N) * self.waypoint_noise_scale).astype(np.float32)
            noise = noise - np.mean(noise)
            
            # 4. Apply and project
            p = current_alpha + step + noise
            current_alpha = self._project_simplex(p)
            schedule[t] = current_alpha
            
        return schedule

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        batch_idx = idx // self.batch_size
        alphas = self.alpha_schedule[batch_idx]
        
        # Deterministic lookup (O(1))
        base_idx = self.image_indices[idx]
        
        clean_img = self.clean_x[base_idx].astype(np.float32)/255.0
        x_mix = alphas[0] * clean_img

        for k in range(self.num_corruptions):
            x_mix += alphas[k+1] * self.corr_x_list[k][base_idx].astype(np.float32)
            
        x_mix = np.clip(x_mix, 0, 1).astype(np.float32)
        
        if self.transform is not None:
            x_mix = self.transform(x_mix)
        else:
            import torchvision.transforms as transforms
            x_mix = transforms.ToTensor()(x_mix)
            
        return x_mix, self.y[base_idx], self.domain, alphas

def create_gradual_mixed_cifarc_dataset(dataset_name, severity, data_dir, corruptions_seq, transform, batch_size, total_frames=50, rng_seed=1, schedule_type="timeline", random_max_step=0.05, alpha_seed=-1, waypoint_speed=0.03, waypoint_arrival_tol=0.02, waypoint_concentration=0.5, waypoint_noise_scale=0.002):
    return GradualMixedCifarCDataset(dataset_name, severity, data_dir, corruptions_seq, transform, batch_size, total_frames, rng_seed, schedule_type, random_max_step, alpha_seed, waypoint_speed, waypoint_arrival_tol, waypoint_concentration, waypoint_noise_scale)


class GradualMixedImagenetCDataset(Dataset):
    """ImageNet-C analogue of GradualMixedCifarCDataset.

    CIFAR blends pixels across corruptions per sample. ImageNet images are too
    big / too many to blend in memory, so here each sample is drawn from a
    *single* corruption chosen stochastically with probability = alpha[frame].
    The batch-level corruption distribution still follows the alpha schedule,
    it's just sampled at the image level instead of mixed at the pixel level.

    There's no "clean" slot here (no clean ImageNet val images on this box),
    so alpha vectors are length `num_corruptions` not `num_corruptions + 1`.
    """

    def __init__(self, data_dir, corruptions_seq, severity, transform,
                 batch_size, total_frames=50, rng_seed=1, schedule_type="random",
                 random_max_step=0.05, alpha_seed=-1, waypoint_speed=0.03,
                 waypoint_arrival_tol=0.02, waypoint_concentration=0.5,
                 waypoint_noise_scale=0.002):
        self.data_dir = data_dir
        self.corruptions_seq = list(corruptions_seq)
        self.severity = severity
        self.transform = transform
        self.batch_size = batch_size
        self.total_frames = total_frames
        self.rng_seed = rng_seed
        self.schedule_type = schedule_type
        self.random_max_step = random_max_step
        self.alpha_seed = alpha_seed
        self.waypoint_speed = waypoint_speed
        self.waypoint_arrival_tol = waypoint_arrival_tol
        self.waypoint_concentration = waypoint_concentration
        self.waypoint_noise_scale = waypoint_noise_scale

        self.num_corruptions = len(self.corruptions_seq)
        self.num_samples = total_frames * batch_size
        self.domain = "gradual_mixed"

        # Per-corruption CustomImageFolder. All folders share the same class layout.
        self.folders = []
        for cor in self.corruptions_seq:
            path = os.path.join(data_dir, cor, str(severity))
            if not os.path.isdir(path):
                raise FileNotFoundError(
                    f"ImageNet-C corruption dir missing: {path}. "
                    f"Extract the corresponding tar under {data_dir}."
                )
            # Use torchvision ImageFolder (not CustomImageFolder) — the latter
            # needs a robustbench test-ID subset file that isn't on this box.
            # We iterate all 50k val images per corruption.
            self.folders.append(torchvision.datasets.ImageFolder(path, transform=transform))

        self.num_images_per_folder = len(self.folders[0])
        for i, f in enumerate(self.folders[1:], start=1):
            if len(f) != self.num_images_per_folder:
                raise ValueError(
                    f"Folder size mismatch: {self.corruptions_seq[0]} has "
                    f"{self.num_images_per_folder} but {self.corruptions_seq[i]} has {len(f)}"
                )

        # ---- RNGs ----
        self.image_rng_seed = rng_seed
        self.image_rng = np.random.default_rng(self.image_rng_seed)
        self.image_indices = self.image_rng.integers(
            0, self.num_images_per_folder, size=self.num_samples
        )

        if alpha_seed == -1:
            import secrets
            self.alpha_rng_seed = secrets.randbelow(2**32 - 1)
        else:
            self.alpha_rng_seed = alpha_seed
        self.alpha_rng = np.random.default_rng(self.alpha_rng_seed)

        # ---- Alpha schedule (N = num_corruptions, no clean slot) ----
        if schedule_type in ("timeline", "linear_overlap"):
            self.alpha_schedule = self._generate_timeline_schedule()
        elif schedule_type == "random":
            self.alpha_schedule = self._generate_random_schedule()
        else:
            raise NotImplementedError(f"Schedule type '{schedule_type}' not supported.")

        # Per-sample corruption assignment — choose one corruption per sample
        # using p=alpha[frame]. Deterministic given rng_seed.
        corr_rng = np.random.default_rng(rng_seed + 1)
        self.corruption_assignments = np.zeros(self.num_samples, dtype=np.int32)
        for i in range(self.num_samples):
            frame = i // batch_size
            p = self.alpha_schedule[frame]
            p = p / p.sum()  # guard against numerical drift
            self.corruption_assignments[i] = corr_rng.choice(self.num_corruptions, p=p)

        self.metadata = {
            "schedule_type": schedule_type,
            "image_rng_seed": self.image_rng_seed,
            "alpha_rng_seed": self.alpha_rng_seed,
            "corruptions_seq": self.corruptions_seq,
            "severity": severity,
            "total_frames": total_frames,
            "batch_size": batch_size,
            "mixing_mode": "image_sampled",  # vs "pixel_blended" in CIFAR
        }

    # --- schedule helpers (adapted from GradualMixedCifarCDataset, N=num_corruptions) ---

    def _generate_timeline_schedule(self):
        N = self.num_corruptions
        schedule = np.zeros((self.total_frames, N), dtype=np.float32)
        for t in range(self.total_frames):
            center = (t / max(1, self.total_frames - 1)) * (N - 1)
            alphas = np.zeros(N, dtype=np.float32)
            for j in range(N):
                alphas[j] = max(0.0, 1.0 - abs(center - j))
            if alphas.sum() == 0:
                alphas[int(center)] = 1.0
            schedule[t] = alphas / alphas.sum()
        return schedule

    def _generate_random_schedule(self):
        N = self.num_corruptions
        schedule = np.zeros((self.total_frames, N), dtype=np.float32)
        current = np.ones(N, dtype=np.float32) / N
        schedule[0] = current
        target = self.alpha_rng.dirichlet(np.ones(N) * self.waypoint_concentration).astype(np.float32)
        for t in range(1, self.total_frames):
            direction = target - current
            distance = np.linalg.norm(direction)
            if distance < self.waypoint_arrival_tol:
                target = self.alpha_rng.dirichlet(
                    np.ones(N) * self.waypoint_concentration
                ).astype(np.float32)
                direction = target - current
                distance = np.linalg.norm(direction)
            step = direction * self.waypoint_speed
            noise = self.alpha_rng.normal(0, self.waypoint_noise_scale, N).astype(np.float32)
            current = current + step + noise
            current = GradualMixedCifarCDataset._project_simplex(current).astype(np.float32)
            schedule[t] = current
        return schedule

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        frame = idx // self.batch_size
        alphas = self.alpha_schedule[frame]
        cor_idx = int(self.corruption_assignments[idx])
        img_idx = int(self.image_indices[idx])
        img, label = self.folders[cor_idx][img_idx]
        return img, label, self.domain, alphas


def create_gradual_mixed_imagenetc_dataset(data_dir, corruptions_seq, severity, transform,
                                           batch_size, total_frames=50, rng_seed=1,
                                           schedule_type="random", random_max_step=0.05,
                                           alpha_seed=-1, waypoint_speed=0.03,
                                           waypoint_arrival_tol=0.02,
                                           waypoint_concentration=0.5,
                                           waypoint_noise_scale=0.002):
    return GradualMixedImagenetCDataset(
        data_dir=data_dir, corruptions_seq=corruptions_seq, severity=severity,
        transform=transform, batch_size=batch_size, total_frames=total_frames,
        rng_seed=rng_seed, schedule_type=schedule_type, random_max_step=random_max_step,
        alpha_seed=alpha_seed, waypoint_speed=waypoint_speed,
        waypoint_arrival_tol=waypoint_arrival_tol,
        waypoint_concentration=waypoint_concentration,
        waypoint_noise_scale=waypoint_noise_scale,
    )


class GradualMixedMnistCDataset(Dataset):
    """MNIST-C analogue of GradualMixedCifarCDataset. Pixel-blends across
    corruptions per sample (MNIST is small enough to fully load in RAM).

    Expects layout: <data_dir>/mnist_c/<corruption>/test_images.npy and
    test_labels.npy, where 'identity' is the clean MNIST.
    """

    _project_simplex = staticmethod(GradualMixedCifarCDataset._project_simplex)
    _generate_timeline_schedule = GradualMixedCifarCDataset._generate_timeline_schedule
    _generate_random_schedule = GradualMixedCifarCDataset._generate_random_schedule

    def __init__(self, data_dir, corruptions_seq, transform, batch_size,
                 total_frames=50, rng_seed=1, schedule_type="random",
                 random_max_step=0.05, alpha_seed=-1, waypoint_speed=0.03,
                 waypoint_arrival_tol=0.02, waypoint_concentration=0.5,
                 waypoint_noise_scale=0.002):
        self.data_dir = data_dir
        self.corruptions_seq = list(corruptions_seq)
        self.transform = transform
        self.batch_size = batch_size
        self.total_frames = total_frames
        self.rng_seed = rng_seed
        self.schedule_type = schedule_type
        self.random_max_step = random_max_step
        self.alpha_seed = alpha_seed
        self.waypoint_speed = waypoint_speed
        self.waypoint_arrival_tol = waypoint_arrival_tol
        self.waypoint_concentration = waypoint_concentration
        self.waypoint_noise_scale = waypoint_noise_scale

        # Load clean (identity) and corrupted test images
        clean_path = os.path.join(data_dir, "identity")
        clean_x = np.load(os.path.join(clean_path, "test_images.npy"))  # [N, 28, 28, 1]
        clean_y = np.load(os.path.join(clean_path, "test_labels.npy"))
        self.clean_x = clean_x
        self.y = clean_y

        self.corr_x_list = []
        for cor in self.corruptions_seq:
            cor_x = np.load(os.path.join(data_dir, cor, "test_images.npy"))
            cor_y = np.load(os.path.join(data_dir, cor, "test_labels.npy"))
            assert np.array_equal(cor_y, clean_y), f"Label mismatch for corruption {cor}"
            self.corr_x_list.append(cor_x)

        self.num_corruptions = len(self.corruptions_seq)
        self.num_samples = total_frames * batch_size
        self.domain = "gradual_mixed"

        # ---- RNGs ----
        self.image_rng_seed = rng_seed
        self.image_rng = np.random.default_rng(self.image_rng_seed)
        self.image_indices = self.image_rng.integers(0, len(self.y), size=self.num_samples)

        if alpha_seed == -1:
            import secrets
            self.alpha_rng_seed = secrets.randbelow(2**32 - 1)
        else:
            self.alpha_rng_seed = alpha_seed
        self.alpha_rng = np.random.default_rng(self.alpha_rng_seed)

        if schedule_type in ("timeline", "linear_overlap"):
            self.alpha_schedule = self._generate_timeline_schedule()
        elif schedule_type == "random":
            self.alpha_schedule = self._generate_random_schedule()
        else:
            raise NotImplementedError(f"Schedule type '{schedule_type}' not supported.")

        self.metadata = {
            "schedule_type": schedule_type,
            "image_rng_seed": self.image_rng_seed,
            "alpha_rng_seed": self.alpha_rng_seed,
            "corruptions_seq": self.corruptions_seq,
            "total_frames": total_frames,
            "batch_size": batch_size,
            "mixing_mode": "pixel_blended",
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        batch_idx = idx // self.batch_size
        alphas = self.alpha_schedule[batch_idx]
        base_idx = self.image_indices[idx]

        clean_img = self.clean_x[base_idx].astype(np.float32) / 255.0
        x_mix = alphas[0] * clean_img
        for k in range(self.num_corruptions):
            x_mix += alphas[k + 1] * self.corr_x_list[k][base_idx].astype(np.float32) / 255.0
        x_mix = np.clip(x_mix, 0, 1).astype(np.float32)  # [28, 28, 1]

        if self.transform is not None:
            x_mix = self.transform(x_mix)
        else:
            import torchvision.transforms as transforms
            x_mix = transforms.ToTensor()(x_mix)

        return x_mix, int(self.y[base_idx]), self.domain, alphas


def create_gradual_mixed_mnistc_dataset(data_dir, corruptions_seq, transform, batch_size,
                                        total_frames=50, rng_seed=1, schedule_type="random",
                                        random_max_step=0.05, alpha_seed=-1, waypoint_speed=0.03,
                                        waypoint_arrival_tol=0.02, waypoint_concentration=0.5,
                                        waypoint_noise_scale=0.002):
    return GradualMixedMnistCDataset(
        data_dir=data_dir, corruptions_seq=corruptions_seq, transform=transform,
        batch_size=batch_size, total_frames=total_frames, rng_seed=rng_seed,
        schedule_type=schedule_type, random_max_step=random_max_step, alpha_seed=alpha_seed,
        waypoint_speed=waypoint_speed, waypoint_arrival_tol=waypoint_arrival_tol,
        waypoint_concentration=waypoint_concentration, waypoint_noise_scale=waypoint_noise_scale,
    )
