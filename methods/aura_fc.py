"""
AURA-FC — Adaptive Update through Representation Adaptation (FC-only variant).

Adapts the last FC layer parameters in a low-dimensional latent space using
Kalman filtering with diagonal covariance and OU prediction dynamics.

Offline training:
  - Trains A (projection matrix) only, using ground-truth labels.
  - z and cov evolve continuously across episodes (TBPTT detach between steps).
  - forward_offline_step() provides an explicit stateless API (caller owns z/cov).

Online adaptation:
  - Uses loaded A and phi (from AURA_FC.OFFLINE_CKPT) to define the latent space.
  - Runs a Kalman filter online using TRUE labels (fully supervised).
  - z and cov are persistent across all batches in a domain sequence.

Jacobian computation:
  - Both offline and online use torch.func.jacrev for a unified, general path.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from copy import deepcopy
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from models.model import split_up_model

logger = logging.getLogger(__name__)


@ADAPTATION_REGISTRY.register()
class AURA_FC(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        # AURA_FC hyperparameters
        self.latent_dim = cfg.AURA_FC.LATENT_DIM
        self.obs_cov_scale = cfg.AURA_FC.OBS_COV_SCALE
        self.gamma = cfg.AURA_FC.GAMMA
        self.init_cov_scale = cfg.AURA_FC.INIT_COV_SCALE

        # Split model into encoder + classifier
        self.encoder, self.classifier = split_up_model(
            self.model, cfg.MODEL.ARCH, cfg.CORRUPTION.DATASET
        )
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        # For timm backbones split_up_model strips the classifier off self.model
        # (reset_classifier(0)), which invalidates the state snapshot taken in
        # TTAMethod.__init__ — re-take it so reset() can still load it strictly.
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        # Extract FC layer parameters as a flat vector
        # classifier is nn.Linear(in_features, num_classes)
        fc_weight = self.classifier.weight.data.detach().clone().flatten()  # [num_classes * in_features]
        fc_bias = self.classifier.bias.data.detach().clone()               # [num_classes]
        self.fc_theta_source = torch.cat([fc_weight, fc_bias]).to(self.device)  # source params
        self.num_fc_params = self.fc_theta_source.shape[0]
        self.in_features = self.classifier.in_features

        logger.info(f"AURA_FC: FC params={self.num_fc_params}, latent_dim={self.latent_dim}, "
                    f"in_features={self.in_features}, num_classes={num_classes}")

        # Projection matrix A and bias phi
        """self.A = torch.zeros(self.latent_dim, self.num_fc_params, device=self.device)
        nn.init.xavier_uniform_(self.A)"""
        self.A = torch.eye(self.latent_dim, self.num_fc_params, device=self.device)
        self.phi = self.fc_theta_source.unsqueeze(0)  # [1, num_fc_params] — bias = source params

        # ---- Offline checkpoint loading ----
        """if hasattr(cfg.AURA_FC, "OFFLINE_CKPT") and cfg.AURA_FC.OFFLINE_CKPT != "":
            self._load_and_validate_checkpoint(cfg)"""

        # Initialize online Kalman state:
        # z = 0 → decodes to source params via z @ A + phi = phi = source params
        self.z = torch.zeros(self.latent_dim, device=self.device)
        self.cov = torch.eye(self.latent_dim, device=self.device) * self.init_cov_scale

        # Store initial state for OU process
        self.z_init = self.z.clone()
        self.cov_init = self.cov.clone()

        # Observation noise covariance
        self.R = torch.eye(num_classes, device=self.device) * self.obs_cov_scale

        # Store AURA_FC state for reset (used by test_time.py reset logic)
        self._aura_fc_z_init = self.z.clone()
        self._aura_fc_cov_init = self.cov.clone()

        # Timing state
        self._timing_enabled = cfg.TIMING.ENABLED
        self._timing_warmup = cfg.TIMING.WARMUP_SAMPLES
        self._timing_target = cfg.TIMING.NUM_MEASUREMENTS
        self._timing_results = []
        self._timing_sample_counter = 0

    # ------------------------------------------------------------------
    # Checkpoint loading + validation
    # ------------------------------------------------------------------

    def _load_and_validate_checkpoint(self, cfg):
        """Load offline-trained A and phi, with full compatibility validation."""
        ckpt_path = cfg.AURA_FC.OFFLINE_CKPT+"/last_params.pth"
        state = torch.load(ckpt_path, map_location=self.device)

        # ---- Hard errors (incompatible structure) ----
        """hard_checks = {
            "latent_dim":    (cfg.AURA_FC.LATENT_DIM,           "AURA_FC.LATENT_DIM"),
            "num_fc_params": (self.num_fc_params,             "num_fc_params (model architecture)"),
            "num_classes":   (self.num_classes,               "num_classes"),
            "dataset":       (cfg.CORRUPTION.DATASET,         "CORRUPTION.DATASET"),
            "model_arch":    (cfg.MODEL.ARCH,                 "MODEL.ARCH"),
        }
        for key, (expected, cfg_name) in hard_checks.items():
            if key in state and state[key] != expected:
                raise ValueError(
                    f"AURA_FC checkpoint mismatch on '{key}': "
                    f"checkpoint={state[key]!r}, current config {cfg_name}={expected!r}. "
                    f"Checkpoint: {ckpt_path}"
                )"""

        # ---- Soft warnings (hyperparameters that may be intentionally changed) ----
        soft_checks = {
            "gamma":          (cfg.AURA_FC.GAMMA,           "AURA_FC.GAMMA"),
            "init_cov_scale": (cfg.AURA_FC.INIT_COV_SCALE,  "AURA_FC.INIT_COV_SCALE"),
            "obs_cov_scale":  (cfg.AURA_FC.OBS_COV_SCALE,   "AURA_FC.OBS_COV_SCALE"),
        }
        for key, (expected, cfg_name) in soft_checks.items():
            if key in state and state[key] != expected:
                logger.warning(
                    f"AURA_FC checkpoint '{key}' differs: "
                    f"checkpoint={state[key]}, current {cfg_name}={expected}. "
                    f"Using current config value."
                )

        # ---- Scenario warnings ----
        if "corruption_types" in state:
            ckpt_types = sorted(state["corruption_types"])
            curr_types = sorted(cfg.CORRUPTION.TYPE)
            if ckpt_types != curr_types:
                logger.warning(
                    f"AURA_FC checkpoint corruption_types={ckpt_types} "
                    f"!= current CORRUPTION.TYPE={curr_types}"
                )
        if "severity" in state:
            if sorted(state["severity"]) != sorted(cfg.CORRUPTION.SEVERITY):
                logger.warning(
                    f"AURA_FC checkpoint severity={state['severity']} "
                    f"!= current CORRUPTION.SEVERITY={cfg.CORRUPTION.SEVERITY}"
                )

        # ---- Restore parameters ----
        self.A = state["A"].to(self.device)
        self.phi = state["phi"].to(self.device)
        logger.info(f"AURA_FC: Loaded A [shape={tuple(self.A.shape)}] and phi from checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # TTAMethod interface
    # ------------------------------------------------------------------

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

    def collect_params(self):
        return [], []

    def setup_optimizer(self):
        return None

    def copy_model_and_optimizer(self):
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        return model_states, None

    def reset(self):
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        self.z = self._aura_fc_z_init.clone()
        self.cov = self._aura_fc_cov_init.clone()

    def _expand_to_fc(self, z):
        """Expand latent z to FC layer parameters (weight, bias)."""
        theta = z @ self.A + self.phi  # [1, num_fc_params]
        theta = theta.squeeze(0)
        W = theta[:self.num_classes * self.in_features].view(self.num_classes, self.in_features)
        b = theta[self.num_classes * self.in_features:]
        return W, b

    @staticmethod
    def _obs_fn(z, feat_i, A, phi, num_classes, in_features):
        """
        Observation function: latent z → classifier logits.
        Standalone (no self) so it can be passed directly to torch.func.jacrev.
        Args:
            z      : [latent_dim]
            feat_i : [in_features]
            A      : [latent_dim, num_fc_params]
            phi    : [1, num_fc_params]
        Returns:
            y_pred : [num_classes]
        """
        theta = z @ A + phi.squeeze(0)   # [num_fc_params]
        W = theta[:num_classes * in_features].view(num_classes, in_features)
        b = theta[num_classes * in_features:]
        return feat_i @ W.T + b       # [num_classes]

    def _compute_jacobian(self, z, feat_i):
        """
        Compute H = d(obs_fn)/d(z) via torch.func.jacrev.
        Returns H : [num_classes, latent_dim].
        Both offline and online use this single code path (unified).
        """
        H = torch.func.jacrev(self._obs_fn, argnums=0)(
            z, feat_i, self.A, self.phi,
            self.num_classes, self.in_features)
            
        return H  # [num_classes, latent_dim]

    def _aura_fc_update_step(self, feat_i, y, z, cov):
        """
        One fully-differentiable Bayesian (Kalman) update step.
        Jacobian computed via torch.func.jacrev — unified for offline and online.

        Args:
            feat_i : [in_features]   — encoder feature for one sample
            y      : [num_classes]   — one-hot observation (ground-truth or pseudo)
            z      : [latent_dim]    — current latent mean
            cov    : [latent_dim, latent_dim] — current latent covariance

        Returns:
            z_new, cov_new
        """
        # OU predict step
        predict_mean = self.gamma * z + (1 - self.gamma) * self.z_init
        predict_cov = (self.gamma ** 2) * cov + (1 - self.gamma ** 2) * self.cov_init

        W, b = self._expand_to_fc(z)
        y_pred = F.softmax(feat_i @ W.T + b)  # [num_classes]

        # Jacobian H = d(y_pred)/d(z) via jacrev
        H = self._compute_jacobian(z, feat_i)  # [num_classes, latent_dim]

        # Kalman gain
        S   = self.R + H @ predict_cov @ H.T   # [num_classes, num_classes]
        C   = predict_cov @ H.T                 # [latent_dim, num_classes]
        K_T = torch.linalg.solve(S, C.T)        # [num_classes, latent_dim]
        K   = K_T.T                             # [latent_dim, num_classes]

        """print(K.norm())"""
        # Update
        z_new   = predict_mean + K @ (y - y_pred)
        cov_new = (torch.eye(self.latent_dim, device=self.device) - K @ H) @ predict_cov

        return z_new, cov_new

    # ------------------------------------------------------------------
    # Offline training API
    # ------------------------------------------------------------------

    def forward_offline_step(self, ctx_imgs, ctx_labels, qry_imgs, qry_labels, z, cov):
        """
        One offline training step with EXPLICIT z/cov state (stateless API).
        The caller owns z and cov and passes them in each episode.
        This enables continuous state evolution across the full episode sequence.

        Uses ground-truth labels for both context (Kalman observation) and
        query (cross-entropy loss).

        Args:
            ctx_imgs   : [1, C, H, W]
            ctx_labels : [1]            — ground-truth label for context
            qry_imgs   : [Q, C, H, W]
            qry_labels : [Q]            — ground-truth labels for query
            z          : [latent_dim]   — current latent mean (carried from last step)
            cov        : [latent_dim, latent_dim] — current covariance (carried)

        Returns:
            logits_q : [Q, num_classes]
            loss_q   : scalar (CE loss over query samples)
            z_new    : [latent_dim]  — updated latent mean (IN the autograd graph)
            cov_new  : [latent_dim, latent_dim] — updated covariance (IN graph)
        """
        feat_c = self.encoder(ctx_imgs)   # [1, in_features]
        feat_q = self.encoder(qry_imgs)   # [Q, in_features]

        feat_i = feat_c[0]                # [in_features]
        y_c    = ctx_labels[0]

        # One-hot observation from TRUE ground-truth context label
        W0, b0 = self._expand_to_fc(z)
        num_classes = b0.shape[0]
        y = torch.zeros(num_classes, device=z.device, dtype=z.dtype)
        y[y_c] = 1.0                      # ground-truth one-hot

        # Kalman update — differentiable through A (and z/cov carry)
        z_new, cov_new = self._aura_fc_update_step(feat_i, y, z, cov)

        # Decode updated z_new for query prediction
        W_new, b_new = self._expand_to_fc(z_new)
        logits_q = feat_q @ W_new.T + b_new
        loss_q = F.cross_entropy(logits_q, qry_labels)

        return logits_q, loss_q, z_new, cov_new

    # ------------------------------------------------------------------
    # Online adaptation (fully supervised)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_and_adapt(self, x, labels=None):
        """
        Online Kalman adaptation — FULLY SUPERVISED.
        Uses true labels when provided; falls back to pseudo-labels only if
        labels=None (for backward compatibility, with a warning).

        Args:
            x      : list with x[0] = imgs [B, C, H, W]
            labels : [B] integer ground-truth class labels (strongly preferred)
        """
        imgs_test = x[0]
        features = self.encoder(imgs_test)   # [B, in_features]

        if labels is None:
            logger.warning(
                "AURA_FC.forward_and_adapt called WITHOUT labels — falling back to "
                "pseudo-labels. For fully supervised online mode, pass true labels."
            )

        z = self.z.clone()
        cov = self.cov.clone()

        num_adapt_samples = getattr(self.cfg.AURA_FC, "NUM_ADAPT_SAMPLES", -1)
        if num_adapt_samples > 0:
            num_updates = min(num_adapt_samples, features.shape[0])
        else:
            num_updates = features.shape[0]

        for i in range(num_updates):
            _should_time = (
                self._timing_enabled
                and self._timing_sample_counter >= self._timing_warmup
                and len(self._timing_results) < self._timing_target
            )
            if _should_time:
                torch.cuda.synchronize()
                _t0 = time.perf_counter()

            feat_i = features[i].detach()

            W_eval, b_eval = self._expand_to_fc(z)
            y_pred = feat_i @ W_eval.T + b_eval   # [num_classes]

            if labels is not None:
                # Supervised: use ground-truth one-hot
                y = torch.zeros(self.num_classes, device=self.device)
                y[labels[i]] = 1.0
            else:
                # Pseudo-label fallback (deprecated path)
                y = torch.zeros_like(y_pred)
                y[y_pred.argmax()] = 1.0

            z, cov = self._aura_fc_update_step(feat_i, y, z, cov)

            if _should_time:
                torch.cuda.synchronize()
                self._timing_results.append(time.perf_counter() - _t0)
            self._timing_sample_counter += 1

        self.z = z.detach()
        self.cov = cov.detach()

        W, b = self._expand_to_fc(self.z)
        outputs = features @ W.T + b
        return outputs

    @torch.no_grad()
    def forward_eval(self, imgs):
        """
        Evaluate classification logits on a batch of test images using the current
        adapted latent state z without modifying latent state z or covariance P.
        """
        features = self.encoder(imgs)
        W, b = self._expand_to_fc(self.z)
        return features @ W.T + b
