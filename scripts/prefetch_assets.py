#!/usr/bin/env python
"""
Pre-stage everything a run needs from the network, and verify the local data tree.

Run this ONCE on a machine/node that has internet (e.g. a cluster login node)
before submitting jobs: compute nodes are frequently offline, and the first
`timm.create_model(..., pretrained=True)` / HF-hub download would then fail.

Everything lands in the HuggingFace cache, so set HF_HOME to a shared, quota-
generous location and export the same value in the job script:

    export HF_HOME=/path/to/shared/cache/huggingface
    python scripts/prefetch_assets.py --cfg cfgs/cifar10_c/aura_vit.yaml

Without --cfg it fetches the default ViT assets used by the ViT experiments.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("prefetch")

# (timm architecture, optional fine-tuned checkpoint) pairs used by the ViT configs
DEFAULT_ASSETS = [
    ("vit_base_patch16_224", "hf_hub:edadaltocg/vit_base_patch16_224_in21k_ft_cifar10"),  # CIFAR-10 ViT
    ("vit_base_patch16_224", ""),                                                         # ImageNet-1k ViT
]

# Dataset directory names, relative to DATA_DIR (see conf.complete_data_dir_path)
DATASET_DIRS = {
    "cifar10_c": ["CIFAR-10-C"],
    "cifar100_c": ["CIFAR-100-C"],
    "imagenet_c": ["ImageNet-C"],
    "mnist_c": ["mnist_c"],
}


def prefetch_model(arch: str, checkpoint: str):
    import timm
    from models.model import load_timm_checkpoint
    label = f"{arch} <- {checkpoint}" if checkpoint else f"{arch} (timm default weights)"
    logger.info(f"fetching {label}")
    if checkpoint:
        # num_classes comes from the checkpoint; create empty then load strictly
        import json
        from huggingface_hub import hf_hub_download
        num_classes = None
        if checkpoint.startswith("hf_hub:"):
            try:
                with open(hf_hub_download(checkpoint[len("hf_hub:"):], "config.json")) as f:
                    num_classes = json.load(f).get("num_classes")
            except Exception as e:
                logger.warning(f"  could not read config.json ({e}); assuming the arch default")
        model = timm.create_model(arch, pretrained=False,
                                  **({"num_classes": num_classes} if num_classes else {}))
        load_timm_checkpoint(model, checkpoint)
    else:
        model = timm.create_model(arch, pretrained=True)
    logger.info(f"  OK — {sum(p.numel() for p in model.parameters()):,} params, "
                f"num_classes={model.num_classes}")


# The clean CIFAR tarballs come from www.cs.toronto.edu, which is heavily throttled
# (observed: >1 h for 170 MB from the cluster). These HF mirrors serve the *identical*
# archive — the md5 is torchvision's own expected value and is verified after the
# download, so a tampered or stale mirror is rejected and we fall back to the origin.
CIFAR_TARBALL_MIRRORS = {
    "cifar10": {
        "filename": "cifar-10-python.tar.gz",
        "md5": "c58f30108f718f92721af3b95e74349a",
        "repo_id": "liangnanying/cifar-10-python",
    },
    "cifar100": {
        "filename": "cifar-100-python.tar.gz",
        "md5": "eb9058c3a382ffc7106e4002c42a8d85",
        "repo_id": "nakroy/cifar100-python",
    },
}


def fetch_clean_cifar_tarball(data_dir: str, which: str) -> bool:
    """
    Place the clean CIFAR archive in data_dir from a fast HF mirror.

    torchvision skips its own download when the archive is already present with the
    expected md5 ("Using downloaded and verified file"), so this only has to land the
    file. Returns True if the archive is in place, False to let torchvision fetch it
    from the original URL.
    """
    import hashlib
    import shutil

    spec = CIFAR_TARBALL_MIRRORS[which]
    target = os.path.join(data_dir, spec["filename"])

    def md5(path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    if os.path.isfile(target) and md5(target) == spec["md5"]:
        logger.info(f"{spec['filename']} already present and verified")
        return True

    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"fetching {spec['filename']} from the {spec['repo_id']} mirror "
                    f"(the cs.toronto.edu origin is throttled)")
        cached = hf_hub_download(spec["repo_id"], spec["filename"], repo_type="dataset")
        if md5(cached) != spec["md5"]:
            logger.warning(f"mirror md5 mismatch for {spec['filename']} — ignoring the mirror")
            return False
        os.makedirs(data_dir, exist_ok=True)
        shutil.copyfile(cached, target)
        logger.info(f"  OK — {os.path.getsize(target) / 2**20:.0f} MiB verified into {target}")
        return True
    except Exception as e:
        logger.warning(f"mirror fetch failed ({type(e).__name__}: {e}); "
                       f"falling back to the original download")
        return False


# Zenodo serves CIFAR-10-C as one 2.7 GiB stream and is slow from the cluster.
# This HF repo hosts the identical archive (verified: extracted members are
# md5-identical to the Zenodo copy, original 2018 mtimes preserved) at ~25 MB/s.
# The checksums below are from the verified local Zenodo copy and cover labels
# plus the six corruptions the Table 1 protocol uses; any mismatch rejects the
# mirror and falls back to Zenodo.
CIFAR10C_MIRROR = {
    "repo_id": "Sijiax/CIFAR-10-C",
    "filename": "CIFAR-10-C.tar",
    "subdir": "CIFAR-10-C",
    "md5": {
        "labels.npy":         "c439b113295ed5254878798ffe28fd54",
        "gaussian_noise.npy": "ecaf8b9a2399ffeda7680934c33405fd",
        "shot_noise.npy":     "3a7239bb118894f013d9bf1984be7f11",
        "impulse_noise.npy":  "2090e01c83519ec51427e65116af6b1a",
        "defocus_blur.npy":   "7d1322666342a0702b1957e92f6254bc",
        "glass_blur.npy":     "7361fb4019269e02dbf6925f083e8629",
        "motion_blur.npy":    "fffa5f852ff7ad299cfe8a7643f090f4",
    },
}


def _md5(path, chunk=1 << 20):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch_cifar10c_from_mirror(data_dir: str) -> bool:
    """
    Fetch + extract CIFAR-10-C from the fast HF mirror, verifying every checksum
    we know. Returns True only if the extracted tree passes verification.
    """
    import tarfile

    target_dir = os.path.join(data_dir, CIFAR10C_MIRROR["subdir"])

    def verify():
        for name, digest in CIFAR10C_MIRROR["md5"].items():
            path = os.path.join(target_dir, name)
            if not os.path.isfile(path) or _md5(path) != digest:
                return False
        return True

    if verify():
        logger.info(f"{target_dir} already present and verified")
        return True

    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"fetching CIFAR-10-C (2.7 GiB) from the {CIFAR10_MIRROR_NAME} mirror "
                    f"(Zenodo is slow from the cluster)")
        tar_path = hf_hub_download(CIFAR10C_MIRROR["repo_id"], CIFAR10C_MIRROR["filename"],
                                   repo_type="dataset")
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"extracting into {data_dir}")
        with tarfile.open(tar_path) as tar:
            # filter="data" (py>=3.11.4) blocks absolute paths / traversal from a
            # third-party archive; older interpreters fall back to plain extractall.
            try:
                tar.extractall(data_dir, filter="data")
            except TypeError:
                tar.extractall(data_dir)
        if not verify():
            logger.warning("mirror archive failed checksum verification — falling back to Zenodo")
            return False
        logger.info("  OK — CIFAR-10-C verified against the known Zenodo checksums")
        return True
    except Exception as e:
        logger.warning(f"mirror fetch failed ({type(e).__name__}: {e}); falling back to Zenodo")
        return False


CIFAR10_MIRROR_NAME = CIFAR10C_MIRROR["repo_id"]


def download_data(data_dir: str, datasets):
    """
    Fetch the datasets that have an automatic downloader (login node only).

    CIFAR-10-C / CIFAR-100-C come from the RobustBench Zenodo records; the clean
    CIFAR test set is needed too because the gradual-mixed dataset blends the
    clean images in (and would otherwise try to download it from a compute node).
    ImageNet-C has no automatic download — upload it yourself.
    """
    import torchvision
    from pathlib import Path
    from robustbench.data import ZENODO_CORRUPTIONS_LINKS, CORRUPTIONS_DIR_NAMES
    from robustbench.model_zoo.enums import BenchmarkDataset
    from robustbench.zenodo_download import zenodo_download

    os.makedirs(data_dir, exist_ok=True)
    for name in datasets:
        if name == "cifar10_c":
            logger.info("fetching clean CIFAR-10 test set")
            fetch_clean_cifar_tarball(data_dir, "cifar10")
            torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
            if not fetch_cifar10c_from_mirror(data_dir):
                record, files = ZENODO_CORRUPTIONS_LINKS[BenchmarkDataset.cifar_10]
                logger.info(f"fetching CIFAR-10-C from Zenodo record {record} (~2.9 GB, resumable)")
                zenodo_download(record, files, save_dir=Path(data_dir))
        elif name == "cifar100_c":
            logger.info("fetching clean CIFAR-100 test set")
            fetch_clean_cifar_tarball(data_dir, "cifar100")
            torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
            record, files = ZENODO_CORRUPTIONS_LINKS[BenchmarkDataset.cifar_100]
            logger.info(f"fetching CIFAR-100-C from Zenodo record {record}")
            zenodo_download(record, files, save_dir=Path(data_dir))
        else:
            logger.warning(f"no automatic download for '{name}' — upload it to {data_dir} yourself")


def check_data(data_dir: str, datasets):
    ok = True
    for name in datasets:
        for sub in DATASET_DIRS.get(name, []):
            path = os.path.join(data_dir, sub)
            if os.path.isdir(path):
                logger.info(f"data OK: {path}")
            else:
                logger.error(f"data MISSING: {path} (needed for {name}) — upload it before submitting jobs")
                ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", action="append", default=[],
                        help="Config file(s) whose MODEL.ARCH / MODEL.CKPT_PATH / CORRUPTION.DATASET should be fetched/checked. Repeatable.")
    parser.add_argument("--data-dir", default=None, help="Override DATA_DIR for the dataset check")
    parser.add_argument("--download-data", action="store_true",
                        help="Also download the datasets that have an automatic downloader (CIFAR-10-C / CIFAR-100-C)")
    args = parser.parse_args()

    logger.info(f"HF_HOME={os.environ.get('HF_HOME', '(unset — using ~/.cache/huggingface)')}")

    assets, datasets, data_dir = [], set(), args.data_dir or "./data"
    if args.cfg:
        from conf import cfg
        for path in args.cfg:
            cfg.defrost()
            cfg.merge_from_file(path)
            assets.append((cfg.MODEL.ARCH, cfg.MODEL.CKPT_PATH))
            datasets.add(cfg.CORRUPTION.DATASET)
            if args.data_dir is None:
                data_dir = cfg.DATA_DIR
    else:
        assets = DEFAULT_ASSETS
        datasets = {"cifar10_c"}

    failures = []
    for arch, checkpoint in dict.fromkeys(assets):
        try:
            prefetch_model(arch, checkpoint)
        except Exception as e:
            logger.error(f"  FAILED {arch} / {checkpoint}: {type(e).__name__}: {e}")
            failures.append((arch, checkpoint))

    if args.download_data:
        try:
            download_data(data_dir, datasets)
        except Exception as e:
            logger.error(f"dataset download failed: {type(e).__name__}: {e}")
            failures.append(("dataset-download", data_dir))

    data_ok = check_data(data_dir, datasets)

    if failures or not data_ok:
        logger.error("prefetch incomplete — fix the errors above before submitting cluster jobs")
        sys.exit(1)
    logger.info("all assets present — safe to submit jobs on offline compute nodes")


if __name__ == "__main__":
    main()
