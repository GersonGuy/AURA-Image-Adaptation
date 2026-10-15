import os
import sys
import logging
import argparse
import csv
import torch

# Ensure repository root is in python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args, get_num_classes
from models.model import get_model
from utils.registry import ADAPTATION_REGISTRY
import methods
from datasets.corruptions_datasets import create_cifarc_dataset

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

CORRUPTIONS = [
    ("pixelate", "Spatial downsampling and block resolution reduction"),
    ("contrast", "Global dynamic range compression"),
    ("zoom_blur", "Dynamic focal motion blur radiating from center"),
    ("fog", "Atmospheric fog & haze effect (additive veil)"),
    ("jpeg_compression", "DCT quantization block artifacts")
]


def evaluate_model_on_dataset(model, dataset, device, batch_size=128):
    """Evaluates error rate of a model on a dataset in eval mode."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    
    # Store initial model mode
    training_state = model.training if hasattr(model, "training") else False
    if hasattr(model, "eval"):
        model.eval()

    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            if hasattr(model, "forward_eval"):
                outputs = model.forward_eval(imgs)
            elif hasattr(model, "model"):
                outputs = model.model(imgs)
            else:
                outputs = model(imgs)

            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if hasattr(model, "train") and training_state:
        model.train()

    err = 100.0 * (1.0 - correct / total) if total > 0 else 0.0
    return err


def run_baseline_evaluation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = args.dataset
    num_classes = get_num_classes(dataset_name)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    csv_file = open(args.output_csv, "w", newline="")
    writer = csv.writer(csv_file)

    writer.writerow([
        "method", "corruption", "corruption_description",
        "initial_unadapted_error", "final_adapted_error",
        "abs_error_reduction", "rel_error_reduction_pct"
    ])

    METHOD_ALIAS_MAP = {
        "bn": "bntest",
        "norm": "bntest",
        "bn_test": "bntest",
        "bntest": "bntest",
        "bn_alpha": "bnalpha",
        "bnalpha": "bnalpha",
        "bn_beta": "bnalpha",
        "bn_ema": "bnema",
        "bnema": "bnema",
        "roid": "roid",
        "source": "source"
    }

    methods_to_evaluate = args.methods.split(",")

    for method_input in methods_to_evaluate:
        method_input = method_input.strip().lower()
        method_name = METHOD_ALIAS_MAP.get(method_input, method_input)

        logger.info(f"========== Starting Evaluation for Method: '{method_name}' ==========")

        # Load fresh base config
        cfg.merge_from_file(args.cfg)
        cfg.MODEL.ADAPTATION = method_name

        for corruption, description in CORRUPTIONS:
            logger.info(f"--- Method: {method_name} | Corruption: {corruption} ---")

            # Load dataset for current corruption
            ds = create_cifarc_dataset(
                dataset_name=dataset_name,
                severity=5,
                data_dir=cfg.DATA_DIR,
                corruption=corruption,
                transform=None
            )

            # 1. Fresh base model & adaptation method instantiation
            base_model, model_preprocess = get_model(cfg, num_classes, device)
            base_model.model_preprocess = model_preprocess

            method_cls = ADAPTATION_REGISTRY.get(method_name)
            model = method_cls(cfg=cfg, model=base_model, num_classes=num_classes)

            # 2. STEP 1: Evaluate initial UNADAPTED error (inference before adaptation)
            init_err = evaluate_model_on_dataset(model, ds, device)
            logger.info(f"  [Initial Unadapted Inference] {corruption} Error: {init_err:.2f}%")

            if method_name.lower() == "source":
                final_err = init_err
            else:
                # 3. STEP 2: Adapt method on corruption stream
                loader = torch.utils.data.DataLoader(ds, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=2)
                for batch in loader:
                    imgs = batch[0].to(device)
                    # Perform TTA step for method
                    _ = model([imgs])

                # 4. STEP 3: Evaluate final ADAPTED error (inference after adaptation)
                final_err = evaluate_model_on_dataset(model, ds, device)

            abs_drop = init_err - final_err
            rel_drop_pct = (abs_drop / init_err * 100.0) if init_err > 0 else 0.0

            logger.info(f"  [Final Adapted Inference] {corruption} Error: {final_err:.2f}% (Drop: -{abs_drop:.2f}%, Rel: {rel_drop_pct:.1f}%)")

            writer.writerow([
                method_name, corruption, description,
                round(init_err, 2), round(final_err, 2),
                round(abs_drop, 2), round(rel_drop_pct, 2)
            ])
            csv_file.flush()

    csv_file.close()
    logger.info(f"Saved baseline evaluation results to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline TTA methods (BN, ROID, Source) on corruptions")
    parser.add_argument("--cfg", type=str, default="cfgs/cifar10_c/aura_nonlinear.yaml")
    parser.add_argument("--dataset", type=str, default="cifar10_c")
    parser.add_argument("--methods", type=str, default="source,norm,roid")
    parser.add_argument("--output_csv", type=str, default="logs/abrupt_recovery/baselines_summary.csv")
    args = parser.parse_args()

    run_baseline_evaluation(args)
