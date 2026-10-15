# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ---------------------------------- Misc options --------------------------- #

# Setting - see README.md for more information
_C.SETTING = "continual"

# Data directory
_C.DATA_DIR = "./data"

# Weight directory
_C.CKPT_DIR = "./ckpt"

# Output directory
_C.SAVE_DIR = "./output"

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# Enables printing intermediate results every x batches.
# Default -1 corresponds to no intermediate results
_C.PRINT_EVERY = -1

# Seed to use. If None, seed is not set!
# Note that non-determinism is still present due to non-deterministic GPU ops.
_C.RNG_SEED = 1

# Deterministic experiments.
_C.DETERMINISM = False

# Precision
_C.MIXED_PRECISION = False

# Optional description of a config
_C.DESC = ""

# # Config destination (in SAVE_DIR)
# _C.CFG_DEST = "cfg.yaml"


# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Some of the available models can be found here:
# Torchvision: https://pytorch.org/vision/0.14/models.html
# timm: https://github.com/huggingface/pytorch-image-models/tree/v0.6.13
# RobustBench: https://github.com/RobustBench/robustbench
# OpenCLIP: https://github.com/mlfoundations/open_clip
_C.MODEL.ARCH = 'Standard'

# Type of pre-trained weights
# For torchvision models see: https://pytorch.org/vision/0.14/models.html
# For OpenClip models, use either 'openai' (for the original OpenAI weights) or see https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv
_C.MODEL.WEIGHTS = "IMAGENET1K_V1"

# Whether to use a CLIP based architecture
_C.MODEL.USE_CLIP = False

# Path to a specific checkpoint
_C.MODEL.CKPT_PATH = ""

# Inspect the cfgs directory to see all possibilities
_C.MODEL.ADAPTATION = 'source'

# Reset the model before every new batch
_C.MODEL.EPISODIC = False

# Reset the model after a certain amount of update steps (e.g., used in RDumb)
_C.MODEL.RESET_AFTER_NUM_UPDATES = 0

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.DATASET = 'cifar10_c'

# Check https://github.com/hendrycks/robustness for corruption details
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]

# Number of examples to evaluate. If num_ex != -1, each sequence is sub-sampled to the specified amount
# For ImageNet-C, RobustBench loads a list containing 5000 samples.
_C.CORRUPTION.NUM_EX = -1

# ------------------------------- Batch norm options ------------------------ #
_C.BN = CfgNode()

# BN alpha (1-alpha) * src_stats + alpha * test_stats
_C.BN.ALPHA = 0.1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Number of updates per batch
_C.OPTIM.STEPS = 1

# Learning rate
_C.OPTIM.LR = 1e-3

# Optimizer choices: Adam, AdamW, SGD
_C.OPTIM.METHOD = 'Adam'

# Beta1 for Adam based optimizers
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0

# --------------------------------- Mean teacher options -------------------- #
_C.M_TEACHER = CfgNode()

# Mean teacher momentum for EMA update
_C.M_TEACHER.MOMENTUM = 0.999

# --------------------------------- Contrastive options --------------------- #
_C.CONTRAST = CfgNode()

# Temperature term for contrastive learning
_C.CONTRAST.TEMPERATURE = 0.1

# Output dimension of projector
_C.CONTRAST.PROJECTION_DIM = 128

# Contrastive mode
_C.CONTRAST.MODE = 'all'

# --------------------------------- CoTTA options --------------------------- #
_C.COTTA = CfgNode()

# Restore probability
_C.COTTA.RST = 0.01

# Average probability for TTA
_C.COTTA.AP = 0.92

# --------------------------------- GTTA options ---------------------------- #
_C.GTTA = CfgNode()

_C.GTTA.STEPS_ADAIN = 1
_C.GTTA.PRETRAIN_STEPS_ADAIN = 20000
_C.GTTA.LAMBDA_MIXUP = 1/3
_C.GTTA.USE_STYLE_TRANSFER = False

# --------------------------------- RMT options ----------------------------- #
_C.RMT = CfgNode()

_C.RMT.LAMBDA_CE_SRC = 1.0          # Lambda for source replay. Set to 0 for source-free variant
_C.RMT.LAMBDA_CE_TRG = 1.0          # Lambda for self-training
_C.RMT.LAMBDA_CONT = 1.0            # Lambda for contrastive learning
_C.RMT.NUM_SAMPLES_WARM_UP = 50000  # Number of samples used during the mean teacher warm-up

# --------------------------------- SANTA options --------------------------- #
_C.SANTA = CfgNode()

_C.SANTA.LAMBDA_CE_TRG = 1.0        # Lambda for self-training
_C.SANTA.LAMBDA_CONT = 1.0          # Lambda for contrastive learning

# --------------------------------- AdaContrast options --------------------- #
_C.ADACONTRAST = CfgNode()

_C.ADACONTRAST.QUEUE_SIZE = 16384
_C.ADACONTRAST.CONTRAST_TYPE = "class_aware"
_C.ADACONTRAST.CE_TYPE = "standard" # ["standard", "symmetric", "smoothed", "soft"]
_C.ADACONTRAST.ALPHA = 1.0          # Lambda for classification loss
_C.ADACONTRAST.BETA = 1.0           # Lambda for instance loss
_C.ADACONTRAST.ETA = 1.0            # Lambda for diversity loss

_C.ADACONTRAST.DIST_TYPE = "cosine"         # ["cosine", "euclidean"]
_C.ADACONTRAST.CE_SUP_TYPE = "weak_strong"  # ["weak_all", "weak_weak", "weak_strong", "self_all"]
_C.ADACONTRAST.REFINE_METHOD = "nearest_neighbors"
_C.ADACONTRAST.NUM_NEIGHBORS = 10

# --------------------------------- LAME options ---------------------------- #
_C.LAME = CfgNode()

_C.LAME.AFFINITY = "rbf"
_C.LAME.KNN = 5
_C.LAME.SIGMA = 1.0
_C.LAME.FORCE_SYMMETRY = False

# --------------------------------- EATA options ---------------------------- #
_C.EATA = CfgNode()

# Fisher alpha. If set to 0.0, EATA becomes ETA and no EWC regularization is used
_C.EATA.FISHER_ALPHA = 2000.0

# Diversity margin
_C.EATA.D_MARGIN = 0.05
_C.EATA.MARGIN_E0 = 0.4             # Will be multiplied by: EATA.MARGIN_E0 * math.log(num_classes)

# --------------------------------- SAR options ---------------------------- #
_C.SAR = CfgNode()

# Threshold e_m for model recovery scheme
_C.SAR.RESET_CONSTANT_EM = 0.2

# --------------------------------- DeYO options ---------------------------- #
_C.DEYO = CfgNode()

_C.DEYO.REWEIGHT_ENT = True
_C.DEYO.REWEIGHT_PLPD = True
_C.DEYO.PLPD = 0.2
_C.DEYO.MARGIN = 0.5                # Will be multiplied by: DEYO.MARGIN * math.log(num_classes)
_C.DEYO.AUG_TYPE = "patch"          # Choose from: ['occ', 'patch', 'pixel']
_C.DEYO.OCCLUSION_SIZE = 112        # For aug_type occ
_C.DEYO.ROW_START = 56              # For aug_type occ
_C.DEYO.COLUMN_START = 56           # For aug_type occ
_C.DEYO.PATCH_LEN = 4               # For aug_type patch

# --------------------------------- ROTTA options -------------------------- #
_C.ROTTA = CfgNode()

_C.ROTTA.MEMORY_SIZE = 64
_C.ROTTA.UPDATE_FREQUENCY = 64
_C.ROTTA.NU = 0.001
_C.ROTTA.ALPHA = 0.05
_C.ROTTA.LAMBDA_T = 1.0
_C.ROTTA.LAMBDA_U = 1.0

# --------------------------------- RPL options ---------------------------- #
_C.RPL = CfgNode()

# Q value of GCE loss
_C.RPL.Q = 0.8

# --------------------------------- ROID options --------------------------- #
_C.ROID = CfgNode()

_C.ROID.USE_WEIGHTING = True        # Whether to use loss weighting
_C.ROID.USE_PRIOR_CORRECTION = True # Whether to use prior correction
_C.ROID.USE_CONSISTENCY = True      # Whether to use consistency loss
_C.ROID.MOMENTUM_SRC = 0.99         # Momentum for weight ensembling (param * model + (1-param) * model_src)
_C.ROID.MOMENTUM_PROBS = 0.9        # Momentum for diversity weighting
_C.ROID.TEMPERATURE = 1/3           # Temperature for weights
_C.ROID.SUPERVISED = False          # True for supervised CE loss
_C.ROID.ADAPT_SAMPLES = -1          # Fixed number of samples per batch to use for adaptation.

# --------------------------------- CMF options --------------------------- #
_C.CMF = CfgNode()

_C.CMF.ALPHA = 0.99
_C.CMF.GAMMA = 0.99
_C.CMF.Q = 0.005
_C.CMF.TYPE = "lp"

# ------------------------------- CLIP options ---------------------------- #
_C.CLIP = CfgNode()

_C.CLIP.PROMPT_MODE = "custom"                  # Choose from: custom, ensemble, cupl, all_prompts
_C.CLIP.PROMPT_TEMPLATE = ["a photo of a {}."]  # List of custom prompt templates
_C.CLIP.PROMPT_PATH = "datasets/cupl_prompts/CuPL_ImageNet_prompts.json" # Path to .json file containing CuPL prompts for example
_C.CLIP.PRECISION = "fp16"                      # Precision of the restored weights
_C.CLIP.FREEZE_TEXT_ENCODER = True              # Whether to freeze the text encoder in ZeroShotCLIP

# ------------------------------- TPT options ----------------------------- #
_C.TPT = CfgNode()

_C.TPT.SELECTION_P = 0.1            # Percentile of the most certain prediction
_C.TPT.N_CTX = 4                    # Number of tunable context tokens
_C.TPT.CTX_INIT = "a_photo_of_a"    # Context initialization
_C.TPT.CLASS_TOKEN_POS = "end"      # Position of the class token. Choose from: [end, middle, front]

# --------------------------------- AURA_FC options --------------------------- #
_C.AURA_FC = CfgNode()

_C.AURA_FC.LATENT_DIM = 64
_C.AURA_FC.OBS_COV_SCALE = 0.1
_C.AURA_FC.GAMMA = 0.999
_C.AURA_FC.INIT_COV_SCALE = 0.01
_C.AURA_FC.OFFLINE_CKPT = ""
_C.AURA_FC.NUM_EPISODES = 1000
# Number of samples from each batch used for adaptation (-1 means all samples)
_C.AURA_FC.NUM_ADAPT_SAMPLES = 10

# --------------------------------- AURA options ------------------------- #
_C.AURA = CfgNode()

_C.AURA.LATENT_DIM = 64
_C.AURA.OBS_COV_SCALE = 0.1
_C.AURA.GAMMA = 0.999
_C.AURA.INIT_COV_SCALE = 0.01
_C.AURA.OFFLINE_CKPT = ""
_C.AURA.BN_GROUPING_MODE = "shared_all_bn" # or "per_stage_bn"
_C.AURA.STAGE_SELECTION = ["stage_3"] # Applicable when mode is per_stage_bn
_C.AURA.NUM_EPISODES = 1000
_C.AURA.NUM_ADAPT_SAMPLES = 10
# Jacobian computation mode for the AURA_FC observation function.
# "auto" -> picks jacfwd when num_classes >= latent_dim (cheaper for wide outputs),
# "rev"  -> force torch.func.jacrev (reverse-mode, scales with num_classes),
# "fwd"  -> force torch.func.jacfwd (forward-mode, scales with latent_dim).
_C.AURA.JACOBIAN_MODE = "auto"
_C.AURA.JAC_CHUNK_SIZE = 0    # >0 chunks jacrev over num_classes to bound peak memory (ImageNet with stage_1)

# Nonlinear lifting map: residual correction on top of the linear map.
# AURA.LIFTING.TYPE = "linear"    -> theta = phi + z @ A               (default)
# AURA.LIFTING.TYPE = "nonlinear" -> theta = phi + z @ A + delta(z)
#
# delta: Linear(d, H) -> activation -> Linear(H, p)
# H     = HIDDEN_DIMS[0] (one hidden layer; list for extensibility)
# Final output layer zero-initialized (ZERO_INIT_OUTPUT = True)
# RESIDUAL_LINEAR = True means A is kept and jointly trained (recommended)
_C.AURA.LIFTING = CfgNode()
_C.AURA.LIFTING.TYPE             = "linear"      # "linear" | "nonlinear"
_C.AURA.LIFTING.RESIDUAL_LINEAR  = True           # keep + train A alongside delta
_C.AURA.LIFTING.HIDDEN_DIMS      = [500]          # hidden layer widths (one element = one hidden layer)
_C.AURA.LIFTING.ACTIVATION       = "gelu"         # "gelu" | "relu" | "leaky_relu" | "tanh"
_C.AURA.LIFTING.NORMALIZATION    = "none"         # "none"  (no norm in the first experiments)
_C.AURA.LIFTING.OUTPUT_SCALE     = 1.0            # scalar multiplier on delta output
_C.AURA.LIFTING.ZERO_INIT_OUTPUT = True           # zero-init final linear layer

# --------------------------------- EKF_FC options ------------------------- #
_C.EKF_FC = CfgNode()
_C.EKF_FC.OBS_COV_SCALE = 0.001
_C.EKF_FC.GAMMA = 0.999
_C.EKF_FC.INIT_COV_SCALE = 0.01
_C.EKF_FC.NUM_ADAPT_SAMPLES = 15
_C.EKF_FC.DIAGONAL_COV = False   # True -> store P as [D] vector instead of [D,D]; required for D > ~20k
_C.EKF_FC.TOP_K_CLASSES = 0      # >0 -> use only top-K highest-probability classes as Kalman observations (shrinks H, makes ImageNet feasible)

# --------------------------------- EKF_FC options ------------------------- #
_C.EKF_FC = CfgNode()

_C.EKF_FC.OBS_COV_SCALE = 0.1
_C.EKF_FC.GAMMA = 0.999
_C.EKF_FC.INIT_COV_SCALE = 0.01
_C.EKF_FC.NUM_ADAPT_SAMPLES = -1

# --------------------------------- SGD_BASE options ----------------------- #
_C.SGD_BASE = CfgNode()
_C.SGD_BASE.LR = 1e-3
_C.SGD_BASE.MOMENTUM = 0.9
_C.SGD_BASE.WEIGHT_DECAY = 0.0
_C.SGD_BASE.NESTEROV = False
_C.SGD_BASE.OPTIM_METHOD = "Adam"              # "Adam", "AdamW", or "SGD"
_C.SGD_BASE.NUM_UPDATES_PER_SAMPLE = 1         # SGD steps per sample
_C.SGD_BASE.NUM_ADAPT_SAMPLES = 10             # Samples from each batch to adapt on (-1 = all)
_C.SGD_BASE.STAGE_SELECTION = ["fc"]            # "fc", "stage_1", "stage_2", "stage_3", "all"
_C.SGD_BASE.LOSS = "cross_entropy"              # "cross_entropy" or "entropy"

# --------------------------------- LORA options --------------------------- #
_C.LORA = CfgNode()
_C.LORA.RANK                   = 4          # LoRA rank r
_C.LORA.ALPHA                  = 8.0        # Scaling: alpha/r applied in forward
_C.LORA.LR                     = 1e-3       # Adapter learning rate
_C.LORA.OPTIM_METHOD           = "Adam"     # "Adam", "AdamW", or "SGD"
_C.LORA.WEIGHT_DECAY           = 0.0
_C.LORA.MOMENTUM               = 0.9        # SGD only
_C.LORA.NESTEROV               = False
_C.LORA.NUM_UPDATES_PER_SAMPLE = 1          # Gradient steps per labeled sample
_C.LORA.NUM_ADAPT_SAMPLES      = 15         # Labeled samples used per batch
_C.LORA.SCOPE                  = "fc"       # "fc" | "fc_stage1"
_C.LORA.TARGET_MODULES         = []         # Explicit module names (overrides SCOPE)

# --------------------------------- Timing options ------------------------- #
_C.TIMING = CfgNode()
_C.TIMING.ENABLED = True                        # Master switch — OFF by default
_C.TIMING.WARMUP_SAMPLES = 50                   # Cumulative samples to skip before measuring
_C.TIMING.NUM_MEASUREMENTS = 10                 # Total per-sample measurements to collect

# --------------------------------- Offline Training options ---------------- #
_C.OFFLINE = CfgNode()
_C.OFFLINE.NUM_EPISODES = 2000
_C.OFFLINE.NUM_UPDATES_PER_EPISODE = 1
_C.OFFLINE.PRINT_FREQ = 50
_C.OFFLINE.LR = 1e-3
_C.OFFLINE.Q_SAMPLES = 5
# Optional domain-generalization override for the offline corruption pool (M).
# If non-empty, offline training uses this list instead of MIXED_GRADUAL.CORRUPTION_TYPES.
# The online eval continues to read MIXED_GRADUAL.CORRUPTION_TYPES (the test-time pool N),
# allowing the offline (M) and online (N) pools to be disjoint without overloading MIXED_GRADUAL.
_C.OFFLINE.CORRUPTION_POOL = []
_C.OFFLINE.OPTIMIZER = "Adam"    # or "SGD" (lighter memory; useful for ImageNet-scale A matrices)
_C.OFFLINE.MOMENTUM = 0.9        # used only when OPTIMIZER=="SGD"
_C.OFFLINE.RESET_STATE_PER_EPISODE = False   # Fix 1: if True, re-init z/cov at each episode start (prevents drift)
_C.OFFLINE.Z_NORM_CLIP = 0.0     # Fix 2: if >0, clip ||z|| per block after each AURA_FC update (prevents intra-episode drift)
_C.OFFLINE.EARLY_STOP_PATIENCE = 0  # >0: stop training after N consecutive PRINT_FREQ windows without loss improvement

# --------------------------------- Mixed Gradual options ------------------- #
_C.MIXED_GRADUAL = CfgNode()
_C.MIXED_GRADUAL.FRAMES = 1000
_C.MIXED_GRADUAL.BATCH_SIZE = 6
_C.MIXED_GRADUAL.CORRUPTION_TYPES = ["gaussian_noise", "shot_noise", "impulse_noise"]
_C.MIXED_GRADUAL.SEVERITY = 1
_C.MIXED_GRADUAL.SCHEDULE_TYPE = "timeline"
_C.MIXED_GRADUAL.ALPHA_SEED = -1
_C.MIXED_GRADUAL.RANDOM_MAX_STEP = 0.02
_C.MIXED_GRADUAL.WAYPOINT_SPEED = 0.01
_C.MIXED_GRADUAL.WAYPOINT_ARRIVAL_TOL = 0.005
_C.MIXED_GRADUAL.WAYPOINT_CONCENTRATION = 0.1
_C.MIXED_GRADUAL.WAYPOINT_NOISE_SCALE = 0.001
_C.MIXED_GRADUAL.NUM_UPDATES_PER_EPISODE = 15

# ------------------------------- Source options -------------------------- #
_C.SOURCE = CfgNode()

# Number of workers for source data loading
_C.SOURCE.NUM_WORKERS = 4

# Percentage of source samples used
_C.SOURCE.PERCENTAGE = 1.0   # (0, 1] Possibility to reduce the number of source samples

# Possibility to define the number of source samples. The default setting corresponds to all source samples
_C.SOURCE.NUM_SAMPLES = -1

# ------------------------------- Testing options ------------------------- #
_C.TEST = CfgNode()

# Number of workers for test data loading
_C.TEST.NUM_WORKERS = 4

# Batch size for evaluation (and updates)
_C.TEST.BATCH_SIZE = 128

# If the batch size is 1, a sliding window approach can be applied by setting window length > 1
_C.TEST.WINDOW_LENGTH = 1

# Number of augmentations for methods relying on TTA (test time augmentation)
_C.TEST.N_AUGMENTATIONS = 32

# Total frames to generate across sequential data generators (like gradual mixed)
_C.TEST.FRAMES = 50

# The value of the Dirichlet distribution used for sorting the class labels.
_C.TEST.DELTA_DIRICHLET = 0.0

# Debuging mode
_C.TEST.DEBUG = False

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# --------------------------------- EcoTTA options -------------------------- #
# EcoTTA: Memory-Efficient Continual Test-time Adaptation via Self-distilled
# Regularization (Song et al., CVPR 2023) -- https://arxiv.org/abs/2303.01904
# Defaults are PAPER-first. Every behaviour of the unofficial reference
# implementation (github.com/Lily-Le/EcoTTA) is reachable via a yaml override;
# the differences are called out per key below.
_C.ECOTTA = CfgNode()

_C.ECOTTA.K = 5                     # Number of encoder groups / meta networks. 4 or 5.
                                    # Partition per backbone family (paper Tab. 13's principle:
                                    # encoder only, non-uniform, denser at the shallow end):
                                    #   wrn28     (12 blocks) K=4 [2,2,4,4]  K=5 [2,2,2,2,4]
                                    #   resnext29 ( 9 blocks) K=4 [1,2,3,3]  K=5 [1,2,1,2,3]
_C.ECOTTA.CONV_KERNEL_SIZE = 0      # Meta conv kernel size; 0 = the backbone family's default.
                                    # Paper Sec. 4.2 / Tab. 14, verbatim: "we use one and three as
                                    # the kernel size with ResNet and WideResNet, respectively."
                                    # So wrn28 -> 3, resnext29 -> 1. Kernel 3 on a bottleneck net is
                                    # ruinous (paper: 142.2% params on ResNet-50; measured 130% on
                                    # this ResNeXt) for ~0.2pp of error (Tab. 14: 14.4 vs 14.2).
_C.ECOTTA.BN_PLACEMENT = "before"   # Where the meta BN sits relative to its frozen group.
                                    # "before" = paper Sec. 4.2: the BN normalizes the features ENTERING
                                    #            the frozen group; both the frozen path and the conv
                                    #            bypass consume the normalized input. THE PAPER'S METHOD.
                                    # "after"  = the BN consumes the frozen group's OUTPUT. This is the
                                    #            paper's explicitly REJECTED ablation "design (i)"
                                    #            (Table 4a: 18.1 vs 16.8 on CIFAR-10-C / WRN-28).
                                    #            Available as an ablation; do not label it EcoTTA.
_C.ECOTTA.LR = 5e-3                 # Online SGD lr for the meta networks (paper / ref --lr_tta)
_C.ECOTTA.MOMENTUM = 0.9            # Online SGD momentum
_C.ECOTTA.WD = 0.0                  # Online weight decay (the reference uses none)
_C.ECOTTA.USE_OWN_OPTIM = True      # False -> use cfg.OPTIM.* via TTAMethod.setup_optimizer
_C.ECOTTA.LAMBDA_REG = 0.5          # Weight of the self-distilled L1 reg. Paper: 0.5 (CIFAR). Reference code: 0.25
_C.ECOTTA.E_MARGIN = 0.4            # Entropy filter threshold H_0, multiplied by log(num_classes)
_C.ECOTTA.REG_TEACHER = "source_path"   # Which tensor plays x_k in Eq. (4)'s R_k = ||x~_k - x_k||_1.
                                      # "inner_frozen" = the frozen part's output inside the adapted
                                      #     network, i.e. R_k penalizes DRIFT off the frozen path
                                      #     (measured R_k ~ 0.3, commensurate with the entropy term).
                                      # "source_path"  = the frozen part applied to the UN-normalized
                                      #     input (measured R_k ~ 5).
                                      # Default is "source_path": it is the faithful analogue of the
                                      # reference, whose target out1 = original_part(x) is a function of the
                                      # group INPUT and so is PARAMETER-INDEPENDENT -- a real anchor to
                                      # source behaviour. Under "inner_frozen" the (detached) target still
                                      # moves with meta_bn between steps, so R_k pins nothing to source.
_C.ECOTTA.REG_REDUCTION = "mean_all"  # How ||.||_1 in Eq. (4) is reduced.
                                      # "mean_all"      = mean over ALL elements. THE REFERENCE'S CHOICE
                                      #     (wideresnet28.py:132-133: L1Loss(reduction='none')(...).mean()).
                                      # "per_sample_l1" = sum over features, mean over batch -- the literal
                                      #     reading of ||.||_1, which the paper's notation implies.
                                      # "sum"           = sum over everything.
_C.ECOTTA.REG_DETACH_TARGET = True    # Detach x_k in R_k = ||x~_k - x_k||_1. True = the reference
                                      # (wideresnet28.py:133 uses out1.detach()). A non-detached target is
                                      # not a regularizer: both sides could drift together.
_C.ECOTTA.NORMALIZE_REG_BY_K = False  # False = sum R_k over groups (paper/reference); True = mean over groups
_C.ECOTTA.DETACH_GROUP_INPUT = True   # Algorithm 1 `Detach_parts`: detach between partitions during the
                                      # regularization pass so each R_k only touches theta_k. Never applied to
                                      # the entropy pass or to warm-up (that would starve all but the last group)
_C.ECOTTA.UPDATE_RULE = "algorithm1"  # "algorithm1"     = paper Alg. 1: entropy backward, Detach_parts,
                                      #                    regularization backward, then ONE optimizer step
                                      # "two_step"       = unofficial reference: two separate optimizer steps
                                      # "single_forward" = 1 forward + combined loss (cheaper, but loses
                                      #                    Algorithm 1's per-partition gradient isolation)
_C.ECOTTA.FROZEN_BN_MODE = "eval"   # "eval"        = frozen encoder/classifier BNs keep source stats (paper)
                                    # "train"       = reference behaviour (frozen BNs adapt AND overwrite running stats)
                                    # "batch_stats" = host idiom (track_running_stats=False, running_mean/var=None)
_C.ECOTTA.ZERO_INIT_CONV = False    # True -> the meta conv bypass starts as an exact no-op (cold-start ablation)

# ---- warm-up (source-data meta-network pre-training), see warmup_train_ecotta.py ----
_C.ECOTTA.WARMUP_CKPT = ""          # Explicit warm-up checkpoint path. "" -> auto-derive the default path
_C.ECOTTA.REQUIRE_WARMUP = True     # Hard RuntimeError (instead of a warning) if no warm-up checkpoint is found
_C.ECOTTA.LOAD_WARMUP = True        # False -> build the meta networks from scratch and skip checkpoint loading
                                    # entirely. warmup_train_ecotta.py sets this so a warm-up run always trains
                                    # from random init instead of silently resuming an existing checkpoint
_C.ECOTTA.WARMUP_EPOCHS = 10        # Paper: 10 epochs for CIFAR. Lower it for smoke tests
_C.ECOTTA.WARMUP_LR = 5e-2          # Paper: SGD, lr 5e-2
_C.ECOTTA.WARMUP_MOMENTUM = 0.9
_C.ECOTTA.WARMUP_BATCH_SIZE = 64    # Reference --b 64
_C.ECOTTA.WARMUP_LR_MILESTONES = [3, 6, 8]  # MultiStepLR milestones (epochs)
_C.ECOTTA.WARMUP_LR_GAMMA = 0.2
_C.ECOTTA.WARMUP_LINEAR_EPOCHS = 1  # Epochs of linear lr ramp-up (reference --warm 1)
_C.ECOTTA.WARMUP_IDENTITY_INIT = False  # Start warm-up AT the source function instead of from random
                                      # init: zero the conv bypass and calibrate each meta BN to an
                                      # identity from source activation statistics
                                      # (methods/ecotta.py::calibrate_identity_init).
                                      # MEASURED but NOT USEFUL, hence default False. Identity init does
                                      # reproduce the source function (78.92% clean vs the source model's
                                      # 78.89%), and CE warm-up then moves AWAY from it (75.12% after one
                                      # epoch at the paper's lr 5e-2). But clean accuracy turned out to be
                                      # the wrong target: under corruption the randomly-initialised warmed
                                      # network scores 33.65% error on the repo's CIFAR-100 gradual protocol
                                      # versus Source's 38.13%, i.e. the 2.5pp clean deficit buys
                                      # robustness, because the meta BNs normalize with test-batch
                                      # statistics. Kept only as a diagnostic.
_C.ECOTTA.WARMUP_AUGMENT = True     # ColorJitter(0.4) / GaussianBlur / RandomGrayscale, as in the paper & reference
_C.ECOTTA.WARMUP_BN_MODE = "eval"   # "eval"  = frozen BN running stats stay at their source values (recommended:
                                    #           keeps warm-up and online normalization consistent)
                                    # "train" = reference behaviour (frozen BN stats drift during warm-up; the
                                    #           drifted buffers are then persisted in the checkpoint)

# --------------------------------- TinyTTA options ------------------------- #
# TinyTTA: Efficient Test-time Adaptation via Early-exit Ensembles on Edge Devices
# (Jia et al., NeurIPS 2024).
#
# RECONSTRUCTION FROM PROSE. The official release (github.com/h-jia/TTE) contains NO TinyTTA
# algorithm -- no backward pass, no entropy, no exits, no weight standardization; see
# docs/tinytta/code_forensics.md. 0 of 19 ambiguities are resolvable from code. The ids A1..A19
# below refer to docs/tinytta/paper_spec.md section 8, which is the operative contract; the mapping
# and every provenance label live in docs/tinytta/implementation_notes.md.
_C.TINYTTA = CfgNode()

# ---- partition (A9) ----
_C.TINYTTA.K = 4                       # Submodules / exits. The paper uses 4 for every image
                                       # backbone (App. D); App. H's audio model uses 3.
_C.TINYTTA.PARTITION = []              # Blocks per submodule, shallow->deep. [] = the MEASURED
                                       # activation-balanced family default:
                                       #   wrn28     (12 blocks) K=4 [2,2,3,5]  max/min 1.215
                                       #   resnext29 ( 9 blocks) K=4 [1,2,2,4]  max/min 1.280
                                       # Both are the global optimum over contiguous block splits.

# ---- exit heads (A8, A12) ----
_C.TINYTTA.HEAD_KERNEL_SIZE = 1        # WS conv kernel. Never stated; 1 matches the App. D Table 3
                                       # parameter fit.
_C.TINYTTA.HEAD_WIDTH_MULT = 1.0       # c = round(mult * C_in) output channels of the WS conv.
                                       # 1.0 = the RegNet reading of Table 3 (c == C_in);
                                       # 6.0 = the MobileNet/EfficientNet reading (expansion width).
_C.TINYTTA.HEAD_AFFINE = True          # Per-channel gain+shift after the WS conv (Table 3's +2c).
                                       # Needed in practice: per-channel WS is scale-invariant.
_C.TINYTTA.HEAD_POOL = "gap"           # "gap" -> Linear(c, C); "avgpool2" -> 2x2 pool, Linear(4c, C)
_C.TINYTTA.EXIT_K_MODE = "reuse_classifier"  # A12: App. D "the last exit layer retains the same
                                       # memory and parameters as the original model", and Table 3
                                       # lists only exits 1-3 for K=4. "new_head" builds a K-th WS
                                       # head instead (uniform ablation, not the paper's reading).
_C.TINYTTA.EXIT_K_TRAINABLE = True     # With True the PRETRAINED classifier drifts, and the exit-K
                                       # path is byte-for-byte sgd_base(STAGE_SELECTION=['fc'],
                                       # LOSS='entropy'). False = predict at exit K, never update.

# ---- weight standardization (A5, A6, A14) ----
_C.TINYTTA.WS_MODE = "per_out_channel" # A5. "per_out_channel" = ref. [12] (Qiao et al.), the method
                                       # the paper claims to use; reduction dims (1,2,3).
                                       # "whole_tensor" = Eq. 4 read literally ("over all channels
                                       # and dimensions"), a single (mu, sigma). THE most likely
                                       # fidelity dispute: Fig. 7 credits WS with 12-15% accuracy,
                                       # so both must be measured.
_C.TINYTTA.WS_EPS = 1e-5               # A14: never stated.
_C.TINYTTA.WS_EPS_IN_SQRT = False      # False = Eq. 3 as printed, (W-mu)/(sigma+eps).
                                       # True  = ref. [12] numerics, (W-mu)/sqrt(var+eps).
_C.TINYTTA.WS_TARGET = "conv"          # A6. "conv" | "linear" | "both" | "none".
_C.TINYTTA.WS_ABLATION_REPLACEMENT = "identity"  # With WS_TARGET="none": "identity" (Fig. 7's
                                       # "w/o WS") or "groupnorm" (App. E / Fig. 11's TinyTTA+GN).
_C.TINYTTA.WS_IN_OFFLINE = True        # WS active during offline training too. False makes the
                                       # offline artifact fit a different function -> invalid.

# ---- early exit rule (A10, A11) ----
_C.TINYTTA.GAMMA_MODE = "calibrated"   # A10. App. G publishes ONE threshold per (model, dataset),
                                       # NOT per exit, and none of its four backbones is ours -- so
                                       # gamma_1..gamma_K are unpublished for every model.
                                       # "calibrated" = read gammas from the offline checkpoint,
                                       #   produced on CLEAN source validation data only. No leak.
                                       # "scalar"     = broadcast GAMMA_SCALAR to every exit (the
                                       #   paper's App. G form).
                                       # "explicit"   = use TINYTTA.GAMMA verbatim.
_C.TINYTTA.GAMMA = []                  # gamma_1..gamma_{K-1}, nats. Used by "explicit".
_C.TINYTTA.GAMMA_SCALAR = 0.0          # Used by "scalar". App. G, for reference (other backbones):
                                       # CIFAR10C 1.12/1.34/1.53/1.26, CIFAR100C 3.26/2.58/3.10/2.31
_C.TINYTTA.GAMMA_TARGET_FRACTIONS = [] # Calibration targets: fraction of CLEAN source samples that
                                       # should exit at exits 1..K-1. [] = uniform 1/K each.
                                       # Cross-check: Table 2's 9.1% latency saving (50.7 vs 55.8 ms,
                                       # INCLUDING an on-device backward) implies the paper's own
                                       # thresholds make early exits fire rarely, so something like
                                       # [0.05, 0.05, 0.10] is more Table-2-consistent than uniform.
_C.TINYTTA.CALIB_SPLIT = 0.1           # Fraction of the source TRAIN split held out for calibration
                                       # (the heads are fitted on the rest, so gammas are
                                       # out-of-sample). The TEST split is never touched.
_C.TINYTTA.CALIB_NUM_SAMPLES = 10000   # Cap on calibration samples (-1 = all of the held-out part).

# ---- test-time update (A1, A2, A3, A4) ----
_C.TINYTTA.TEST_LOSS = "entropy_exited"  # A1: the loss is NEVER written down in the paper. This is
                                       # INFERRED from App. A Eq. 7 plus section 1's framing and from
                                       # self-consistency with the exit rule.
                                       # "entropy_exited" = H(p^k) at the exit the sample took.
                                       # "entropy_mean_upto_k" = mean of H(p^1..p^k) (A1 sub-case).
                                       # "none" = no update (static self-ensemble reference line).
_C.TINYTTA.UPDATE_HEADS = "exited"     # A2: three contradictory statements in the paper.
                                       # "exited" = only phi_k (section 3.3, the most specific).
                                       # "upto_k" = every exit the sample was evaluated at
                                       #   (section 3.2's literal reading).
                                       # "all"    = every exit for every sample; forces a full
                                       #   forward and defeats the early exit.
_C.TINYTTA.EXIT_MODE = "grouped"       # "grouped" = one forward with the active tensor shrinking as
                                       #   samples exit, samples grouped by chosen exit, ONE
                                       #   optimizer step per batch. Exact w.r.t. the forward
                                       #   because the frozen backbone is pointwise in the batch.
                                       # "per_sample" = the paper's batch-1 loop, B steps per batch.
                                       #   ~20-50x slower; validation runs only.
_C.TINYTTA.LOSS_NORM = "batch_mean"    # "batch_mean" = total / B, the exact B->1 limit of TENT's
                                       #   .mean(0); occupancy-weighted across exits.
                                       # "per_exit_mean" = mean within each exit, summed.
_C.TINYTTA.USE_OWN_OPTIM = True        # False -> use cfg.OPTIM.* via TTAMethod.setup_optimizer.
_C.TINYTTA.OPTIM_METHOD = "SGD"        # A4: TRANSPLANTED from App. H, which is the AUDIO experiment
_C.TINYTTA.LR = 1e-5                   # (MicroNets / Speech Commands) and coincidentally from
_C.TINYTTA.MOMENTUM = 0.9              # App. J, which describes the TTT BASELINE. NO image-domain
_C.TINYTTA.WD = 0.0                    # optimizer/lr/momentum/wd exists anywhere in the paper.
                                       # lr 1e-5 was tuned at BATCH 1: at batch 100 with
                                       # EXIT_MODE="grouped" a 1e5-sample run takes ~1,000 steps
                                       # instead of ~100,000, so total parameter displacement is
                                       # ~100x smaller and TinyTTA may be nearly INERT. Left
                                       # unrescaled deliberately; measured by LOG_PARAM_DRIFT.
_C.TINYTTA.FROZEN_BN_MODE = "eval"     # A19. "eval" = frozen backbone keeps its source running
                                       # statistics -- the only choice compatible with batch size 1,
                                       # with the batch-agnosticism claim, and with the grouped
                                       # forward's row-dropping. "train" / "batch_stats" silently
                                       # add a BN-Stats baseline on top of TinyTTA and are rejected
                                       # when EXIT_MODE="grouped".

# ---- offline self-ensemble stage (A13, A15, A16, A17) ----
_C.TINYTTA.OFFLINE_CKPT = ""           # "" -> auto-derive ckpt/tinytta/ckpt_tinytta_{dataset}_{arch}
                                       #       _K{K}_ws{WS_MODE}_seed{seed}.pth
_C.TINYTTA.REQUIRE_OFFLINE = True      # Hard error rather than a silent cold start. The Limitations
                                       # section is explicit: "TinyTTA requires retraining the model
                                       # via self-ensemble." MEASURED: untrained heads of this
                                       # design are OVER-confident (median entropy 0.01-0.27 nats vs
                                       # ln 10 = 2.303) because per-channel WS amplifies by
                                       # ~sqrt(fan_in), so they fire eagerly and return garbage.
_C.TINYTTA.LOAD_OFFLINE = True         # offline_train_tinytta.py forces False so a fresh run never
                                       # silently RESUMES from the artifact at the derived path.
_C.TINYTTA.OFFLINE_SOURCE = "clean"    # "clean" = clean labelled source TRAIN split (paper 3.4,
                                       #   "the training dataset" / "the source data"). FAITHFUL.
                                       # "corrupted_gradual" = a corrupted gradual-mixture stream
                                       #   (offline_train_aura.py's regime). Recorded in the payload
                                       #   as source_kind; a "clean" run refuses such an artifact.
_C.TINYTTA.OFFLINE_LAMBDA = 0.9        # A13: lambda in Eq. 6. No value is stated for ANY image
                                       # experiment; 0.9 is borrowed from App. J where it is the
                                       # TTT BASELINE's weight. Sweep {0.5,0.7,0.9} if the fit is poor.
_C.TINYTTA.OFFLINE_DISTILL_TARGET = "final_logits"  # A17: z_k in L2. "final_logits" = the frozen
                                       # model's C-dim logits (no projection needed; Eq. 1 uses z^k
                                       # for the exit's logits). "none" disables L2.
_C.TINYTTA.OFFLINE_L2_REDUCTION = "mean"  # Eq. 5 prints ||.||_1. "mean" |
                                       # "sum_over_classes_mean_over_batch" | "sum". The EcoTTA port
                                       # established that a literal sum is catastrophically
                                       # mis-scaled against the CE term; both scales are logged.
_C.TINYTTA.OFFLINE_SUM_OVER_EXITS = True  # Eq. 5 has no explicit sum over k, but 3.4's "concurrent
                                       # training of all models" implies all K jointly.
_C.TINYTTA.OFFLINE_FREEZE_BACKBONE = True  # A16: 3.1/3.3 say the submodules are frozen, Eq. 6's
                                       # min_theta says otherwise. Freezing makes L2 well-posed.
_C.TINYTTA.OFFLINE_TRAIN_EXIT_K = False   # Whether the reused pretrained classifier is also trained
                                       # offline. False = do not perturb the source model.
_C.TINYTTA.OFFLINE_EPOCHS = 50         # A15: the entire offline recipe is a guess. 50 epochs / SGD /
_C.TINYTTA.OFFLINE_LR = 5e-3           # momentum 0.9 / cosine / batch 128 follows TinyTL [31] as
                                       # cited in App. D. (App. D's "50 epochs" is actually the
                                       # MCUNet BACKBONE fine-tune, not the self-ensemble stage.)
                                       #
                                       # LR IS TUNED, AND DELIBERATELY SO. TinyTL's cited 0.05
                                       # provably DIVERGES on these exit heads: epoch-1 CE goes
                                       # 9.63 -> 914,483, every exit lands at chance, and the
                                       # entropies saturate to 0 so the heads fire eagerly and
                                       # return garbage. 5e-3 trains (CE 4.68, exit clean accs
                                       # 0.347/0.372/0.696 after ONE epoch) and yields calibrated
                                       # gammas inside App. G's published CIFAR-10 range. A
                                       # default that reliably diverges is worse than a documented
                                       # tuned one; set 0.05 explicitly to reproduce the paper's
                                       # cited value and watch it blow up.
_C.TINYTTA.OFFLINE_MOMENTUM = 0.9
_C.TINYTTA.OFFLINE_WD = 5e-4
_C.TINYTTA.OFFLINE_BATCH_SIZE = 128
_C.TINYTTA.OFFLINE_SCHEDULE = "cosine" # "cosine" | "none"
_C.TINYTTA.OFFLINE_AUGMENT = True      # RandomCrop(32, padding=4) + RandomHorizontalFlip.
_C.TINYTTA.OFFLINE_WORKERS = 0         # Dataloader workers for the offline stage.

# ---- diagnostics (not paper-derived; load-bearing, see the module docstring) ----
_C.TINYTTA.LOG_EXIT_DIST = True        # Per-exit histogram every LOG_EVERY batches + exit_hist.json.
                                       # A run where everything exits at the last head is NOT
                                       # TinyTTA and is otherwise indistinguishable from a good one,
                                       # because the paper's own effect size is only ~4.3pp.
_C.TINYTTA.LOG_EVERY = 50
_C.TINYTTA.MIN_EARLY_EXIT_FRAC = 0.01  # Warn loudly if fewer than this fraction exit before exit K.
_C.TINYTTA.WARN_AFTER_BATCHES = 20
_C.TINYTTA.LOG_PARAM_DRIFT = True      # Per-exit ||theta - theta_offline||, cumulative path length,
                                       # and the fraction of steps with a non-zero update. This is
                                       # the INERTNESS detector for the lr question above.

# --------------------------------- ELaTTA options -------------------------- #
# ELaTTA: Efficient Test-Time Adaptation through Latent Subspace Coefficients Search
# (Luo et al., arXiv:2510.11068 v2, under review) -- https://arxiv.org/abs/2510.11068
#
# NO CODE HAS BEEN RELEASED, in any language, and there is no third-party reimplementation, so every
# unstated detail below is a judgement call. Defaults are PAPER-first except where marked; the full
# ambiguity list with provenance is docs/elatta/implementation_notes.md (A1..A14).
#
# The paper never evaluates CIFAR-10-C and never evaluates a gradual/mixed schedule, so nothing this
# section produces can be checked against a published number. Its own effect size is +2.79 points of
# ImageNet-C accuracy over source.
_C.ELATTA = CfgNode()

# ---- the four hyperparameters the paper actually gives ----
_C.ELATTA.K = 16                       # PAPER. Subspace dimension k (16 for image classification;
                                       # 2 for the keyword-spotting experiment). Ablated over
                                       # {8, 16, 32}: small k is not expressive enough, large k
                                       # admits noisy directions.
_C.ELATTA.N_ITERS = 8                  # PAPER. Number of CMA-ES generations n. A HARD stopping
                                       # criterion, not convergence: the paper's own ablation has
                                       # n=10 DEGRADING relative to n=8 (over-collapse onto the
                                       # unsupervised entropy proxy). Do not add "run until
                                       # converged".
                                       # 0 is a NON-PAPER value that disables the search entirely
                                       # and makes ELaTTA return the source prediction exactly --
                                       # the hard identity test (A13).
_C.ELATTA.POPULATION = 0               # PAPER (as a formula). lambda. 0 -> auto
                                       # 4 + floor(3 ln k) = 12 at k=16, which is Hansen's CMA-ES
                                       # default and what the paper prints. lambda*n = 96 fitness
                                       # evaluations per sample, all of them CLASSIFIER-HEAD ONLY.
_C.ELATTA.N_SOURCE_SAMPLES = 20        # PAPER. N clean source images used to build V_k. The paper
                                       # ablates up to 50,000 and finds 20 sufficient. Note
                                       # rank(Z_centered) <= N-1, so N=20 with k=16 is fine but
                                       # N <= K is rejected.

# ---- sigma0 and the basis parameterization: THE reproduction hazard (spec hazard #4) ----
# sigma0, the CMA-ES initial step size, is NOT REPORTED ANYWHERE in the paper (main text or
# appendix), and the scale it must have depends entirely on how V_k's columns are scaled. Get this
# wrong by an order of magnitude in either direction and the method is a silent no-op.
_C.ELATTA.BASIS_SCALING = "orthonormal"  # PAPER-LITERAL, and now the default.
                                       # "orthonormal" = V_k with orthonormal columns, which the
                                       #     paper states TWICE and its theory REQUIRES: Prop. III.1's
                                       #     proof reduces the projector to P = V_k V_k^T only when
                                       #     V_k^T V_k = I, and Prop. III.3(ii) Eq. (26)'s margin
                                       #     transfer, ||z* - z_m|| = ||V_k(p* - m)|| = ||p* - m||,
                                       #     is an isometry that is simply FALSE otherwise. A
                                       #     coefficient is then in raw feature-projection units
                                       #     whose source std is S_j/sqrt(N-1), so sigma0 is
                                       #     backbone-specific.
                                       # "whitened"    = columns rescaled by S_j/sqrt(N-1), so
                                       #     p_j = 1 means "one source std along direction j".
                                       #     Retained as an ABLATION only.
                                       # WAS the default here, on the argument that the two span the
                                       # same subspace and reach the same set of z_adapted, so it is
                                       # "a change of units, not of method". That argument is WRONG
                                       # for the search: the reachable set is identical but CMA-ES's
                                       # INITIAL sampling ellipsoid is not -- whitening makes it
                                       # N(z_t, sigma0^2 V diag(S_j^2) V^T) instead of
                                       # N(z_t, sigma0^2 V V^T), a 13.75x condition-number change on
                                       # the shipped basis -- and with only n=2..8 generations and
                                       # c_1=0.0066 / c_mu=0.0137 the covariance barely moves
                                       # (measured ||C - I||_F = 0.35, cond(C) = 1.28 after 8 gens),
                                       # so the geometry is set almost entirely by the column
                                       # scaling. Measured swing: -3.28 pp (whitened) vs -0.44 pp
                                       # (orthonormal) at equal nominal sigma0. Also ||V^T V - I||_F
                                       # = 8.66 whitened vs 2.1e-07 orthonormal, i.e. Eq. (26) fails
                                       # outright. See docs/elatta/fidelity_review.md.
# sigma0 provenance: the paper gives NO value in any version (v1/v2/v3) -- it is not even an input
# to Algorithm 1. The closest available reference is FOA (Niu et al., ICML 2024, arXiv:2404.01650),
# which ELaTTA builds on, compares against, and whose lambda = 4 + 3 log(dim) formula it reuses
# verbatim; FOA's released code (github.com/mr-eggplant/FOA, tta_library/foa.py) pins
# `cma.CMAEvolutionStrategy(dim * [0], 1, ...)` i.e. sigma0 = 1.0, zero mean, pycma defaults. So
# sigma0 = 1.0 is the defensible BORROWED default -- and it is what AUTO produces under "whitened".
# On CIFAR-10-C / WRN-28-10 we MEASURE that it fails badly (53.59 % error vs Source 43.52 %); the
# value this repo ships, 0.05, is tuned. Independent corroboration that 0.05 is the right order for
# this family: PACE (ECML PKDD 2026, arXiv:2603.28678) does CMA-ES over a low-dimensional subspace of
# ViT norm parameters and independently chose a perturbation scale eps = 0.045.
_C.ELATTA.SIGMA0 = -1.0                # <= 0 -> AUTO: 1.0 under "whitened" (one source std), and
                                       # mean(S_k)/sqrt(N-1) (the mean source coefficient std,
                                       # cached in the basis checkpoint) under "orthonormal". Any
                                       # positive value overrides. The resolved value is logged at
                                       # INFO on every run -- it is OUR number, not the paper's.

# ---- offline basis (mandatory; the paper has no cold-start variant, spec hazard #9) ----
_C.ELATTA.BASIS_CKPT = ""              # "" -> ckpt/elatta/<dataset>__<arch>__k<k>__N<N>__seed<s>.pth
                                       # The payload stores the ORTHONORMAL V_k plus the singular
                                       # values; BASIS_SCALING is applied at load time, so one
                                       # artifact serves both settings and is NOT in the filename.
_C.ELATTA.AUTO_BUILD_BASIS = True      # Build + cache on a cache miss at method init. It is ~20
                                       # forward passes (seconds). False -> hard error instead, for
                                       # runs that must not touch source data implicitly.
_C.ELATTA.BASIS_SOURCE_SPLIT = "train" # A1: the paper says "clean source samples" without naming a
                                       # split. "train" is the only leak-free choice -- the test
                                       # stream is built from the CIFAR *test* images.
_C.ELATTA.BASIS_SAMPLE_ORDER = "random"  # A2: which N images. "random" = a subset drawn with a
                                       # generator seeded off RNG_SEED (reproducible, not class- or
                                       # order-biased); "first" = the first N of the split.

# ---- protocol ----
_C.ELATTA.INIT_MODE = "strict"         # PAPER's headline protocol. The two protocols differ ONLY in
                                       # the CMA-ES mean initialization.
                                       # "strict"    = m^(0) = 0 for every sample, full state reset.
                                       # "continual" = m^(0) = p_prev. The paper warm-starts from the
                                       #     previous SAMPLE (worth ~2 points on ImageNet-C: 59.79 vs
                                       #     57.82). Our batch grouping makes intra-batch order
                                       #     arbitrary, so we keep ONE global p_prev, use it for
                                       #     every sample of the next batch, and set it to the mean
                                       #     of that batch's winning p* -- a batch-grouping-forced
                                       #     DEVIATION (A6). Only the mean is ever inherited, never
                                       #     sigma or C (the literal reading of the algorithm box).
                                       #     Unbounded-drift risk with no reset and no shrinkage;
                                       #     the paper offers no safeguard either (hazard #8).
_C.ELATTA.SELECTION = "best_ever"      # PAPER (Step 3, spec hazard #6): p* = arg min over ALL
                                       # coefficients sampled across ALL generations.
                                       # "last_gen_best" = argmin within the final generation only;
                                       # "final_mean"    = the CMA-ES mean's prediction (+1 head
                                       #                   eval/sample). Both are ablations.
_C.ELATTA.CMA_CONSTANTS = "tutorial"   # A8. The paper names no CMA-ES library, seed, recombination
                                       # weights, restart or stopping rule. Our CMA-ES lives in
                                       # methods/cma_es.py; this key selects the strategy-parameter
                                       # set:
                                       # "tutorial" = Hansen, arXiv:1604.00772 Table 1, verbatim.
                                       # "pycma"    = cma.CMAEvolutionStrategy's defaults, which
                                       #     differ from Table 1 in exactly two constants:
                                       #     c_sigma uses (n + mueff + 3) rather than +5, and c_mu
                                       #     adds pycma's rankmu_offset of 0.25. At k=16:
                                       #     c_sigma 0.23169 vs 0.25207, c_mu 0.012191 vs 0.013716
                                       #     (pycma values measured from cma 4.4.4).
                                       # With "pycma" selected, our implementation reproduces
                                       # pycma's (mean, sigma, C) to 1e-16 per generation
                                       # (tests/test_cma_es.py test 2), modulo pycma's ~1e-5
                                       # spread in the initial spectrum and its more accurate
                                       # E||N(0,I)||. NEITHER SET IS THE PAPER'S -- the paper does
                                       # not say. Active CMA (pycma's default negative weights),
                                       # restarts and stopping criteria are not implemented in
                                       # either mode: n is a hard stop.
_C.ELATTA.INCLUDE_UNADAPTED = False    # NON-PAPER when True: admits p = 0 (the untouched source
                                       # prediction) as a candidate in the argmin, which makes the
                                       # returned fitness provably <= the source's. False is
                                       # paper-faithful; True is the natural safety net, kept as an
                                       # available ablation.

# ---- normalization statistics: a DELIBERATE, NON-PAPER COMPOSITION KNOB ----
# READ THIS BEFORE QUOTING ANY NUMBER PRODUCED WITH NORM_MODE != "frozen_source".
# The paper is explicit: "The framework is model-agnostic and DOES NOT REQUIRE UPDATING NORMALIZATION
# STATISTICS", and that is precisely what makes its batch-size-1, single-instance protocol viable.
# ELaTTA, as published, keeps the model in eval() with the SOURCE BN running statistics.
#
# Why the knob exists anyway: on CIFAR-10-C with WRN-28-10 the dominant failure mode under corruption
# is the BN-STATISTIC MISMATCH, not the classifier's decision geometry. Recalibrating BN alone takes
# the Source baseline from 43.52 % error to ~20 % on the 15-corruption severity-5 protocol, whereas
# the faithful ELaTTA cannot beat Source at all (best faithful config 43.58 %, the paper's own
# k=16/n=8 default 48.33 %). With frozen source statistics the PCA subspace is computed from, and the
# entropy is measured on, badly miscalibrated features -- the entropy of a confidently WRONG model
# carries no signal about correctness. This knob lets us measure what the latent-coefficient search
# contributes ON TOP OF a backbone whose features are not broken.
#
# "bn_test" IS NOT ELaTTA. Anything measured with it must be labelled
# "ELaTTA over a BN-recalibrated backbone", never "ELaTTA".
_C.ELATTA.NORM_MODE = "frozen_source"  # "frozen_source" = PAPER, and the DEFAULT. Whole model in
                                       #     eval(); BN uses the stored source running statistics;
                                       #     `track_running_stats` untouched. Every existing ELaTTA
                                       #     number in this repo was produced with this path, and it
                                       #     is bit-for-bit unchanged by the existence of the knob
                                       #     (tests/test_elatta.py section 7a).
                                       # "bn_test" = NON-PAPER DEPARTURE. The encoder's BatchNorm
                                       #     layers normalize with the CURRENT TEST BATCH's
                                       #     statistics instead of the source running statistics --
                                       #     i.e. exactly what this repo's `bntest` / norm_test
                                       #     baseline does (methods/norm.py::BNTest.configure_model).
                                       #     Implemented by putting the BN modules in train() mode
                                       #     and setting `track_running_stats = False`, which makes
                                       #     nn.BatchNorm pass None for the running buffers: batch
                                       #     statistics are used for the forward and NOTHING is
                                       #     written -- running_mean, running_var and
                                       #     num_batches_tracked all stay bitwise unchanged, so the
                                       #     method remains STATELESS across batches. Everything else
                                       #     about ELaTTA is untouched: still gradient-free, no
                                       #     weight update, frozen affine parameters, one encoder
                                       #     pass per batch, lambda*n classifier-head evaluations.
                                       #     CONSEQUENCES, both real: (1) the method becomes
                                       #     BATCH-DEPENDENT, so it is NO LONGER SINGLE-INSTANCE TTA
                                       #     and the paper's batch-1 protocol is not reachable --
                                       #     TEST.BATCH_SIZE 1 makes the BN2d statistics degenerate
                                       #     (a warning is logged); (2) the encoder stops being
                                       #     pointwise in the batch, so the "batching is exact"
                                       #     argument in methods/elatta.py's docstring does not
                                       #     apply. With ELATTA.N_ITERS 0 this mode reproduces the
                                       #     `bntest` baseline exactly, at equal TEST.BATCH_SIZE,
                                       #     which is the check that the composition really is
                                       #     "BN-Test backbone + ELaTTA search on top".

# ---- diagnostics ----
_C.ELATTA.LOG_EVERY = 50               # Every N batches: mean entropy before vs after adaptation,
                                       # mean ||p*||, the fraction of samples whose argmax changed,
                                       # and the encoder-forward / head-eval counters that make the
                                       # efficiency claim (1 encoder pass + lambda*n head evals per
                                       # sample) verifiable from the log.

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_from_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, f"{cfg.MODEL.ADAPTATION}_{cfg.CORRUPTION.DATASET}_{current_time}")
    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    if cfg.RNG_SEED:
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

        if cfg.DETERMINISM:
            # enforce determinism
            if hasattr(torch, "set_deterministic"):
                torch.set_deterministic(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)


def complete_data_dir_path(data_root_dir: str, dataset_name: str):
    # map dataset name to data directory name
    mapping = {"imagenet": "imagenet2012",
               "imagenet_c": "ImageNet-C",
               "imagenet_r": "imagenet-r",
               "imagenet_a": "imagenet-a",
               "imagenet_k": os.path.join("ImageNet-Sketch", "sketch"),
               "imagenet_v2": os.path.join("imagenet-v2", "imagenetv2-matched-frequency-format-val"),
               "imagenet_d": "imagenet-d",      # do not change
               "imagenet_d109": "imagenet-d",   # do not change
               "domainnet126": "DomainNet-126", # directory containing the 6 splits of "cleaned versions" from http://ai.bu.edu/M3SDA/#dataset
               "cifar10": "",       # do not change
               "cifar10_c": "",     # do not change
               "cifar100": "",      # do not change
               "cifar100_c": "",    # do not change
               "mnist": "mnist_c",       # train images come from mnist_c/identity
               "mnist_c": "mnist_c",
               "caltech101": os.path.join("caltech101", "101_ObjectCategories"),
               "dtd": os.path.join("dtd", "dtd", "images"),
               "eurosat": os.path.join("eurosat", "2750"),                      # automatic download fails
               "fgvc_aircraft": os.path.join("fgvc-aircraft-2013b", "data"),    # do not add 'images' in path
               "flowers102": os.path.join("flowers-102", "jpg"),
               "food101": os.path.join("food-101", "images"),
               "oxford_pets": os.path.join("oxford-iiit-pet", "images"),
               "stanford_cars": os.path.join("stanford_cars"),                  # automatic download fails
               "sun397": os.path.join("sun397"),                                # automatic download fails
               "ucf101": os.path.join("ucf101", "UCF-101-midframes"),           # automatic download fails
               "ccc": "",
               }
    assert dataset_name in mapping.keys(),\
        f"Dataset '{dataset_name}' is not supported! Choose from: {list(mapping.keys())}"
    return os.path.join(data_root_dir, mapping[dataset_name])


generalization_dataset_names = [
    "flowers102", "dtd", "oxford_pets", "stanford_cars", "ucf101",
    "caltech101", "food101", "sun397", "fgvc_aircraft", "eurosat"
]


def ds_name2pytorch_ds_name(ds_name: str):
    # converts the dataset name into the pytorch name convention (see: https://pytorch.org/vision/stable/datasets.html)
    lookup_table = {
        "flowers102": "Flowers102",
        "dtd": "DTD",
        "oxford_pets": "OxfordIIITPet",
        "stanford_cars": "StanfordCars",
        "ucf101": "UCF101",
        "caltech101": "Caltech101",
        "food101": "Food101",
        "sun397": "SUN397",
        "fgvc_aircraft": "FGVCAircraft",
        "eurosat": "EuroSAT",
    }
    assert ds_name in lookup_table.keys(), \
        f"There is no mapping for dataset name '{ds_name}'! Supported dataset names are: {list(lookup_table.keys())}"
    return lookup_table[ds_name]


def get_num_classes(dataset_name: str):
    dataset_name2num_classes = {"cifar10": 10, "cifar10_c": 10, "cifar100": 100,  "cifar100_c": 100,
                                "mnist": 10, "mnist_c": 10,
                                "imagenet": 1000, "imagenet_v2": 1000, "imagenet_c": 1000, "ccc": 1000,
                                "imagenet_k": 1000, "imagenet_r": 200, "imagenet_a": 200,
                                "imagenet_d": 164, "imagenet_d109": 109, "imagenet200": 200,
                                "domainnet126": 126,
                                "eurosat": 10, "flowers102": 102, "oxford_pets": 37,
                                "dtd": 47, "food101": 101, "sun397": 397, "caltech101": 100,
                                "ucf101": 101, "stanford_cars": 196, "fgvc_aircraft": 100
                                }
    assert dataset_name in dataset_name2num_classes.keys(), \
        f"Dataset '{dataset_name}' is not supported! Choose from: {list(dataset_name2num_classes.keys())}"
    return dataset_name2num_classes[dataset_name]


def ckpt_path_to_domain_seq(ckpt_path: str):
    assert ckpt_path.endswith('.pth') or ckpt_path.endswith('.pt')
    domain = ckpt_path.replace('.pth', '').split(os.sep)[-1].split('_')[1]
    mapping = {"real": ["clipart", "painting", "sketch"],
               "clipart": ["sketch", "real", "painting"],
               "painting": ["real", "sketch", "clipart"],
               "sketch": ["painting", "clipart", "real"],
               }
    return mapping[domain]
