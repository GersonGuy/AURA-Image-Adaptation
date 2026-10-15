"""
EKF_FC — Extended Kalman Filter on Full FC Parameters.

A standalone test-time adaptation method that performs Bayesian online
updates directly on the full FC layer parameter vector, with no
dimensionality reduction, no latent space, and no projection matrices.

State:
  - theta : [D]    — full FC parameter vector (weight + bias, flattened)
  - P     : [D, D] — state covariance matrix

Update:
  - OU prediction dynamics revert theta/P toward source values
  - EKF update using the observation h(theta, feat) = feat @ W^T + b
  - Jacobian computed via torch.func.jacrev

This method is completely independent of AURA_FC and AURA.
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
class EKF_FC(TTAMethod):
    """Extended Kalman Filter on full FC parameters — no dimensionality reduction."""

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        # EKF_FC hyperparameters
        self.obs_cov_scale = cfg.EKF_FC.OBS_COV_SCALE
        self.gamma = cfg.EKF_FC.GAMMA
        self.init_cov_scale = cfg.EKF_FC.INIT_COV_SCALE
        self.diagonal_cov = bool(getattr(cfg.EKF_FC, "DIAGONAL_COV", False))
        # top_k_classes: if > 0, only use the top-K highest-probability classes
        # as observations in the Kalman update. Shrinks H from [C,D] to [K,D]
        # and K from [D,C] to [D,K] — essential for ImageNet (C=1000, D=2M).
        self.top_k_classes = int(getattr(cfg.EKF_FC, "TOP_K_CLASSES", 0))

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
        self.theta_source = torch.cat([fc_weight, fc_bias]).to(self.device)
        self.num_fc_params = self.theta_source.shape[0]
        self.in_features = self.classifier.in_features

        logger.info(
            f"EKF_FC: FC params D={self.num_fc_params}, "
            f"in_features={self.in_features}, num_classes={num_classes}"
        )

        # Initialize online EKF state:
        # theta starts at source parameters
        self.theta = self.theta_source.clone()
        if self.diagonal_cov:
            # Diagonal covariance: P is stored as a [D] vector (the diagonal entries).
            # Memory O(D) instead of O(D^2) — critical for D=100k+ fc blocks.
            self.P = torch.full((self.num_fc_params,), self.init_cov_scale,
                                device=self.device)
            logger.info("EKF_FC: DIAGONAL covariance mode enabled (P stored as vector)")
        else:
            self.P = torch.eye(self.num_fc_params, device=self.device) * self.init_cov_scale

        # Store initial state for OU process (anchor)
        self.theta_init = self.theta.clone()
        self.P_init = self.P.clone()

        # Observation noise covariance
        self.R = torch.eye(num_classes, device=self.device) * self.obs_cov_scale

        # Store initial state for reset
        self._ekf_theta_init = self.theta.clone()
        self._ekf_P_init = self.P.clone()

        # Timing state
        self._timing_enabled = cfg.TIMING.ENABLED
        self._timing_warmup = cfg.TIMING.WARMUP_SAMPLES
        self._timing_target = cfg.TIMING.NUM_MEASUREMENTS
        self._timing_results = []
        self._timing_sample_counter = 0

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
        self.theta = self._ekf_theta_init.clone()
        self.P = self._ekf_P_init.clone()

    # ------------------------------------------------------------------
    # FC parameter reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_fc(self, theta):
        """Split flat theta vector into (W, b) for the FC layer."""
        W = theta[:self.num_classes * self.in_features].view(
            self.num_classes, self.in_features
        )
        b = theta[self.num_classes * self.in_features:]
        return W, b

    # ------------------------------------------------------------------
    # Observation function (static, for jacrev)
    # ------------------------------------------------------------------

    @staticmethod
    def _obs_fn(theta, feat_i, num_classes, in_features):
        """
        Observation function: full FC params theta -> logits for one sample.
        Standalone (no self) so it can be passed directly to torch.func.jacrev.

        Args:
            theta    : [D]            — full FC parameter vector
            feat_i   : [in_features]  — encoder feature for one sample
            num_classes : int
            in_features : int
        Returns:
            logits   : [num_classes]
        """
        W = theta[:num_classes * in_features].view(num_classes, in_features)
        b = theta[num_classes * in_features:]
        return feat_i @ W.T + b  # [num_classes]

    @staticmethod
    def _obs_fn_topk(theta, feat_i, num_classes, in_features, topk_idx):
        """
        Same as _obs_fn but returns softmax probabilities for only the top-K
        selected classes. Used when TOP_K_CLASSES > 0 to shrink H from [C,D]
        to [K,D] and make ImageNet EKF_FC memory-tractable on 8 GB.
        """
        W = theta[:num_classes * in_features].view(num_classes, in_features)
        b = theta[num_classes * in_features:]
        logits = feat_i @ W.T + b  # [num_classes]
        probs = torch.softmax(logits, dim=0)
        return probs[topk_idx]  # [K]

    # ------------------------------------------------------------------
    # EKF update step
    # ------------------------------------------------------------------

    def _ekf_update_step(self, feat_i, y, theta, P):
        """
        One EKF predict + update cycle on the full FC parameter vector.

        Args:
            feat_i : [in_features]         — encoder feature for one sample
            y      : [num_classes]         — one-hot observation
            theta  : [D]                   — current parameter mean
            P      : [D, D]               — current parameter covariance

        Returns:
            theta_new : [D]
            P_new     : [D, D]
        """
        D = self.num_fc_params

        # ---- OU prediction step ----
        predict_mean = self.gamma * theta + (1 - self.gamma) * self.theta_init
        if self.diagonal_cov:
            # P is a [D] vector of diagonal entries
            predict_P = (self.gamma ** 2) * P + (1 - self.gamma ** 2) * self.P_init
        else:
            predict_P = (self.gamma ** 2) * P + (1 - self.gamma ** 2) * self.P_init

        # ---- Forward pass (predicted logits) ----
        W, b = self._reconstruct_fc(predict_mean)
        y_pred = F.softmax(feat_i @ W.T + b, dim=0)  # [num_classes]

        # ---- Decide effective observation width (full C or top-K) ----
        use_topk = self.top_k_classes > 0 and self.top_k_classes < self.num_classes
        if use_topk:
            K_cls = self.top_k_classes
            topk_idx = torch.topk(y_pred, K_cls).indices           # [K_cls]
            H = torch.func.jacrev(self._obs_fn_topk, argnums=0)(
                predict_mean, feat_i, self.num_classes, self.in_features, topk_idx
            )                                                        # [K_cls, D]
            y_obs = y[topk_idx]                                      # [K_cls]
            y_pred_obs = y_pred[topk_idx]                            # [K_cls]
            R_obs = torch.eye(K_cls, device=self.device) * self.obs_cov_scale  # [K_cls, K_cls]
        else:
            # Full C observations: use logits jacobian (same as before)
            H = torch.func.jacrev(self._obs_fn, argnums=0)(
                predict_mean, feat_i, self.num_classes, self.in_features
            )                                                        # [C, D]
            y_obs = y                                                # [C]
            y_pred_obs = y_pred                                      # [C]
            R_obs = self.R                                           # [C, C]

        if self.diagonal_cov:
            # predict_P is [D]; effective cov is diag(predict_P).
            H_P = H * predict_P.unsqueeze(0)                 # [C_eff, D]
            S = R_obs + H_P @ H.T                            # [C_eff, C_eff]
            K_T = torch.linalg.solve(S, H_P)                 # [C_eff, D]
            K = K_T.T                                         # [D, C_eff]

            theta_new = predict_mean + K @ (y_obs - y_pred_obs)

            # Diagonal approximation to Joseph form
            KH_diag = (K * H.T).sum(dim=1)                   # [D]
            KRK_diag = (K * (K @ R_obs)).sum(dim=1)          # [D]
            P_new = (1.0 - KH_diag).pow(2) * predict_P + KRK_diag
            P_new = P_new.clamp_min(1e-12)
            return theta_new, P_new

        # ---- Full-covariance path (original) ----
        S = R_obs + H @ predict_P @ H.T                # [C_eff, C_eff]
        C_mat = predict_P @ H.T                         # [D, C_eff]
        K_T = torch.linalg.solve(S, C_mat.T)            # [C_eff, D]
        K = K_T.T                                        # [D, C_eff]

        theta_new = predict_mean + K @ (y_obs - y_pred_obs)

        I_KH = torch.eye(D, device=self.device) - K @ H  # [D, D]
        P_new = I_KH @ predict_P @ I_KH.T + K @ R_obs @ K.T
        P_new = (P_new + P_new.T) * 0.5

        return theta_new, P_new

    # ------------------------------------------------------------------
    # Online adaptation (fully supervised)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_and_adapt(self, x, labels=None):
        """
        Two-phase online adaptation:
          Phase 1 (Adapt):    Per-sample EKF updates on a configured subset.
          Phase 2 (Evaluate): Single batched forward on the FULL batch.

        Args:
            x      : list with x[0] = imgs [B, C, H, W]
            labels : [B] integer ground-truth class labels
        """
        imgs_test = x[0]
        features = self.encoder(imgs_test)  # [B, in_features]

        if labels is None:
            logger.warning(
                "EKF_FC.forward_and_adapt called WITHOUT labels — "
                "falling back to pseudo-labels. Pass true labels for supervised mode."
            )

        # Determine how many samples to use for adaptation
        num_adapt_samples = getattr(self.cfg.EKF_FC, "NUM_ADAPT_SAMPLES", -1)
        if num_adapt_samples > 0:
            num_updates = min(num_adapt_samples, features.shape[0])
        else:
            num_updates = features.shape[0]

        theta = self.theta.clone()
        P = self.P.clone()

        # ================================================================
        # Phase 1: Adaptation — per-sample EKF updates
        # ================================================================
        self._timing_units_per_frame = num_updates   # updates performed per frame
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

            # Build one-hot observation
            if labels is not None:
                y = torch.zeros(self.num_classes, device=self.device)
                y[labels[i].long()] = 1.0
            else:
                # Pseudo-label fallback
                W_eval, b_eval = self._reconstruct_fc(theta)
                y_pred_logits = feat_i @ W_eval.T + b_eval
                y = torch.zeros(self.num_classes, device=self.device)
                y[y_pred_logits.argmax()] = 1.0

            theta, P = self._ekf_update_step(feat_i, y, theta, P)

            if _should_time:
                torch.cuda.synchronize()
                self._timing_results.append(time.perf_counter() - _t0)
            self._timing_sample_counter += 1

        # Persist the updated state
        self.theta = theta.detach()
        self.P = P.detach()

        # ================================================================
        # Phase 2: Evaluation — single batched forward on the FULL batch
        # ================================================================
        W, b = self._reconstruct_fc(self.theta)
        outputs = features @ W.T + b  # [B, num_classes]

        return outputs