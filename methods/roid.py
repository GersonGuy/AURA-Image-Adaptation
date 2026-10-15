import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

from copy import deepcopy
from methods.base import TTAMethod
from models.model import ResNetDomainNet126
from augmentations.transforms_cotta import get_tta_transforms
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy, SymmetricCrossEntropy, SoftLikelihoodRatio
from utils.misc import ema_update_model


@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x


@ADAPTATION_REGISTRY.register()
class ROID(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.use_weighting = cfg.ROID.USE_WEIGHTING
        self.use_prior_correction = cfg.ROID.USE_PRIOR_CORRECTION
        self.use_consistency = cfg.ROID.USE_CONSISTENCY
        self.momentum_src = cfg.ROID.MOMENTUM_SRC
        self.momentum_probs = cfg.ROID.MOMENTUM_PROBS
        self.temperature = cfg.ROID.TEMPERATURE
        self.batch_size = cfg.TEST.BATCH_SIZE
        self.class_probs_ema = 1 / self.num_classes * torch.ones(self.num_classes).to(self.device)
        self.tta_transform = get_tta_transforms(self.img_size, padding_mode="reflect", cotta_augs=False)

        # setup loss functions
        self.slr = SoftLikelihoodRatio()
        self.symmetric_cross_entropy = SymmetricCrossEntropy()
        self.softmax_entropy = Entropy()  # not used as loss

        # copy and freeze the source model
        if isinstance(model, ResNetDomainNet126):  # https://github.com/pytorch/pytorch/issues/28594
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        delattr(module, hook.name)

        # note: reduce memory consumption by only saving normalization parameters
        self.src_model = deepcopy(self.model).cpu()
        for param in self.src_model.parameters():
            param.detach_()

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.src_model, self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    def loss_calculation(self, x, labels=None):
        imgs_test = x[0]

        batch_size = imgs_test.shape[0]
        k = self.cfg.ROID.ADAPT_SAMPLES if self.cfg.ROID.ADAPT_SAMPLES > 0 else batch_size
        k = min(k, batch_size)

        subset_idx = torch.randperm(batch_size, device=self.device)[:k]

        # Full-batch forward for eval + adapt-subset slice
        outputs = self.model(imgs_test)
        outputs_adapt = outputs[subset_idx]
        imgs_adapt = imgs_test[subset_idx]
        
        if getattr(self.cfg.ROID, "SUPERVISED", False):
            assert labels is not None, "Labels must be provided for supervised mode"
            labels_adapt = labels[subset_idx]
            loss = F.cross_entropy(outputs_adapt, labels_adapt)
            return outputs, loss, k

        if self.use_weighting:
            with torch.no_grad():
                # calculate diversity based weight
                weights_div = 1 - F.cosine_similarity(self.class_probs_ema.unsqueeze(dim=0), outputs_adapt.softmax(1), dim=1)
                weights_div = (weights_div - weights_div.min()) / (weights_div.max() - weights_div.min())
                mask = weights_div < weights_div.mean()

                # calculate certainty based weight
                weights_cert = - self.softmax_entropy(logits=outputs_adapt)
                weights_cert = (weights_cert - weights_cert.min()) / (weights_cert.max() - weights_cert.min())

                # calculate the final weights
                weights = torch.exp(weights_div * weights_cert / self.temperature)
                weights[mask] = 0.

                self.class_probs_ema = update_model_probs(x_ema=self.class_probs_ema, x=outputs_adapt.softmax(1).mean(0), momentum=self.momentum_probs)

        # calculate the soft likelihood ratio loss
        loss_out = self.slr(logits=outputs_adapt)

        # weight the loss
        if self.use_weighting:
            loss_out = loss_out * weights
            loss_out = loss_out[~mask]
        loss = loss_out.sum() / k

        # calculate the consistency loss
        if self.use_consistency:
            outputs_aug = self.model(self.tta_transform(imgs_adapt[~mask]))
            loss += (self.symmetric_cross_entropy(x=outputs_aug, x_ema=outputs_adapt[~mask]) * weights[~mask]).sum() / k

        return outputs, loss, k

    @torch.enable_grad()
    def forward_and_adapt(self, x, labels=None):
        import time
        if getattr(self.cfg, "TIMING", None) and self.cfg.TIMING.ENABLED:
            if not hasattr(self, "_timing_results"):
                self._timing_results = []
                self._timing_warmup_done = 0
                
            measuring_this_call = False
            if self._timing_warmup_done >= self.cfg.TIMING.WARMUP_SAMPLES:
                if len(self._timing_results) < self.cfg.TIMING.NUM_MEASUREMENTS:
                    measuring_this_call = True
            
            if measuring_this_call:
                torch.cuda.synchronize()
                start_record = time.perf_counter()

        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss, k = self.loss_calculation(x, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss, k = self.loss_calculation(x, labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        if getattr(self.cfg, "TIMING", None) and self.cfg.TIMING.ENABLED:
            if measuring_this_call:
                torch.cuda.synchronize()
                end_record = time.perf_counter()
                # Store SECONDS per adaptation sample. test_time.py (and the timing.json
                # that plot_pareto.py reads) multiplies these by 1000, and every other
                # method appends perf_counter deltas, i.e. seconds. Appending
                # milliseconds here inflated every reported ROID latency by 1000x.
                self._timing_results.append((end_record - start_record) / k)
                # one step covers the k samples the update was computed from
                self._timing_units_per_frame = k
            self._timing_warmup_done += k

        self.model = ema_update_model(
            model_to_update=self.model,
            model_to_merge=self.src_model,
            momentum=self.momentum_src,
            device=self.device
        )

        with torch.no_grad():
            if self.use_prior_correction:
                prior = outputs.softmax(1).mean(0)
                smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / torch.max(prior)
                smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
                outputs *= smoothed_prior
            
            if getattr(self.cfg.ROID, "SUPERVISED", False):
                outputs = self.model(x[0])

        return outputs

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.class_probs_ema = 1 / self.num_classes * torch.ones(self.num_classes).to(self.device)

    def collect_params(self):
        """Collect the affine scale + shift parameters from normalization layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model."""
        self.model.eval()   # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        self.model.requires_grad_(False)  # disable grad, to (re-)enable only necessary parts
        # re-enable gradient for normalization layers
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
