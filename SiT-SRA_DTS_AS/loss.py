import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import smooth_l1_loss


class Simpleloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, loss_type="sml1"):
        if loss_type == "sml1":
            align_loss = smooth_l1_loss(a, b, beta=0.05)
        elif loss_type == "l2":
            align_loss = F.mse_loss(a, b)
        elif loss_type == "l1":
            align_loss = F.l1_loss(a, b)
        elif loss_type == "cos":
            align_loss = 1 - F.cosine_similarity(a, b)
        else:
            raise NotImplementedError()
        return align_loss


def mean_flat(x):
    """Take the mean over all non-batch dimensions."""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def _normalize_mask_ratios(mask_ratio):
    if isinstance(mask_ratio, (int, float)):
        ratios = [float(mask_ratio)]
    else:
        ratios = [float(ratio) for ratio in mask_ratio]

    if not ratios:
        raise ValueError("mask_ratio must contain at least one value")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError(f"mask_ratio should be non-negative, got {ratios}")

    if len(ratios) == 1:
        ratio = ratios[0]
        if ratio <= 0.0 or ratio >= 1.0:
            return [1.0]
        return [ratio, 1.0 - ratio]

    positive_ratios = [ratio for ratio in ratios if ratio > 0.0]
    if not positive_ratios:
        return [1.0]

    total = sum(positive_ratios)
    return [ratio / total for ratio in positive_ratios]


def _compute_group_counts(ratios, seq_len, device, batch_size=None):
    ratios = torch.as_tensor(ratios, device=device, dtype=torch.float32)
    if ratios.ndim == 1:
        ratios = ratios.unsqueeze(0)
    if batch_size is not None and ratios.shape[0] == 1:
        ratios = ratios.expand(batch_size, -1)

    expected = ratios * seq_len
    counts = torch.floor(expected).long()
    remainders = seq_len - counts.sum(dim=1)

    for batch_idx, remainder in enumerate(remainders.tolist()):
        if remainder <= 0:
            continue
        fractional = expected[batch_idx] - counts[batch_idx].float()
        topk = torch.topk(fractional, k=remainder).indices
        counts[batch_idx, topk] += 1

    return counts


def _build_group_ids_from_counts(counts, seq_len, device):
    batch_size, num_groups = counts.shape
    perm = torch.rand((batch_size, seq_len), device=device).argsort(dim=1)
    ordered_group_ids = torch.empty((batch_size, seq_len), device=device, dtype=torch.long)
    rank = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
    start = torch.zeros((batch_size, 1), device=device, dtype=torch.long)

    for group_idx in range(num_groups):
        end = start + counts[:, group_idx:group_idx + 1]
        in_group = (rank >= start) & (rank < end)
        ordered_group_ids[in_group] = group_idx
        start = end

    group_ids = torch.empty_like(ordered_group_ids)
    group_ids.scatter_(1, perm, ordered_group_ids)
    return group_ids.unsqueeze(-1).unsqueeze(-1)


def _expand_group_scalars(group_values, seq_len):
    return [value.expand(value.shape[0], seq_len, 1, 1) for value in group_values]


def _mix_group_scalars(group_values, group_ids):
    mixed = group_values[0].clone()
    for group_idx in range(1, len(group_values)):
        mixed = torch.where(group_ids == group_idx, group_values[group_idx], mixed)
    return mixed


class SRALoss:
    def __init__(
        self,
        prediction="v",
        path_type="linear",
        weighting="uniform",
        latents_scale=None,
        latents_bias=None,
        use_align_loss=False,
        block_out_s=4,
        block_out_t=8,
        mask_ratio=0.5,
        full_sample_prob=0.0,
        loss_type="cos",
        mu=0.847,
        std=0.8,
        teacher_t="self_flow",
        teacher_mask=False,
        dual_time_scheduling=False,
    ):
        if full_sample_prob < 0.0 or full_sample_prob > 1.0:
            raise ValueError(f"full_sample_prob should be in [0, 1], got {full_sample_prob}")

        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.criterion = Simpleloss()
        self.block_out_s = block_out_s
        self.block_out_t = block_out_t
        self.loss_type = loss_type
        self.mask_ratio = _normalize_mask_ratios(mask_ratio)
        self.full_sample_prob = full_sample_prob
        self.use_align_loss = use_align_loss
        self.mu = mu
        self.std = std
        self.teacher_t = teacher_t
        self.teacher_mask = teacher_mask
        self.dual_time_scheduling = dual_time_scheduling
        self.num_mask_groups = len(self.mask_ratio)

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def _sample_timestep(self, bsz, device, dtype):
        if self.weighting == "uniform":
            return torch.rand((bsz, 1, 1, 1), device=device, dtype=dtype)

        if self.weighting == "lognormal":
            rnd_normal = self.mu + self.std * torch.randn((bsz, 1, 1, 1), device=device)
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                return sigma / (1 + sigma)
            if self.path_type == "cosine":
                return 2 / np.pi * torch.atan(sigma)
            raise ValueError(f"Unsupported path_type: {self.path_type}")

        raise ValueError(f"Unsupported weighting: {self.weighting}")

    def _sample_group_timesteps(self, bsz, device, dtype):
        first_timestep = self._sample_timestep(bsz, device, dtype)
        if self.num_mask_groups == 1 or not self.dual_time_scheduling:
            return [first_timestep for _ in range(self.num_mask_groups)]

        group_timesteps = [first_timestep]
        for _ in range(1, self.num_mask_groups):
            group_timesteps.append(self._sample_timestep(bsz, device, dtype))
        return group_timesteps

    def _build_mask(self, bsz, seq_len, device):
        if self.num_mask_groups == 1:
            return None

        if self.full_sample_prob > 0.0 and self.num_mask_groups != 2:
            raise ValueError("full_sample_prob only supports two-group masks")

        counts = _compute_group_counts(self.mask_ratio, seq_len, device, batch_size=bsz)
        mask = _build_group_ids_from_counts(counts, seq_len, device)

        if self.full_sample_prob > 0.0:
            use_full_sample = torch.rand((bsz,), device=device) < self.full_sample_prob
            if use_full_sample.any():
                mask[use_full_sample] = 1

        return mask

    def __call__(self, model, images, teacher, labels, seq_len):
        seq_len = int(seq_len)

        bsz = images.shape[0]
        device = images.device
        dtype = images.dtype

        mask = self._build_mask(bsz, seq_len, device)
        group_timesteps = self._sample_group_timesteps(bsz, device, dtype)
        group_time_inputs = _expand_group_scalars(group_timesteps, seq_len)
        noises = torch.randn_like(images)

        group_model_inputs = []
        group_targets = []
        for group_timestep in group_timesteps:
            alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(group_timestep)
            group_model_inputs.append(alpha_t * images + sigma_t * noises)
            if self.prediction == "v":
                group_targets.append(d_alpha_t * images + d_sigma_t * noises)
            else:
                raise NotImplementedError()

        if mask is None:
            model_output_gen, xr, gt_token = model(
                group_model_inputs[0],
                group_timesteps[0].flatten(),
                y=labels,
                ad=self.block_out_s,
                gt=group_targets[0],
                return_token=True,
            )

            if self.teacher_t == "self_flow":
                teacher_time_base = group_timesteps[0]
            elif self.teacher_t == "sra":
                interval = 0.2 * torch.rand_like(group_timesteps[0])
                teacher_time_base = torch.clamp(group_timesteps[0] - interval, min=0, max=1)
            else:
                raise NotImplementedError()

            alpha_teacher, sigma_teacher, _, _ = self.interpolant(teacher_time_base)
            teacher_input = alpha_teacher * images + sigma_teacher * noises
            time_input_teacher = teacher_time_base.expand(bsz, seq_len, 1, 1)
            mask_teacher = None
        else:
            time_input = _mix_group_scalars(group_time_inputs, mask)
            model_input = torch.cat(group_model_inputs, dim=0)
            gt_input = torch.cat(group_targets, dim=0)
            model_output_gen, xr, gt_token = model(
                model_input,
                time_input.squeeze(-1).squeeze(-1),
                y=labels,
                ad=self.block_out_s,
                mask=mask,
                gt=gt_input,
                return_token=True,
            )

            if self.teacher_mask:
                mask_teacher = mask
                if self.teacher_t == "self_flow":
                    teacher_time_base = group_timesteps[0]
                    for group_timestep in group_timesteps[1:]:
                        teacher_time_base = torch.minimum(teacher_time_base, group_timestep)
                    alpha_teacher, sigma_teacher, _, _ = self.interpolant(teacher_time_base)
                    teacher_branch = alpha_teacher * images + sigma_teacher * noises
                    teacher_groups = [teacher_branch] + [images for _ in range(self.num_mask_groups - 1)]
                    teacher_input = torch.cat(teacher_groups, dim=0)
                    time_input_teacher = teacher_time_base.expand(bsz, seq_len, 1, 1)
                elif self.teacher_t == "same":
                    teacher_input = model_input
                    time_input_teacher = time_input
                elif self.teacher_t == "sra":
                    teacher_group_timesteps = []
                    for group_timestep in group_timesteps:
                        interval = 0.2 * torch.rand_like(group_timestep)
                        teacher_group_timesteps.append(torch.clamp(group_timestep - interval, min=0, max=1))
                    teacher_time_input = _expand_group_scalars(teacher_group_timesteps, seq_len)
                    time_input_teacher = _mix_group_scalars(teacher_time_input, mask)
                    teacher_inputs = []
                    for teacher_group_timestep in teacher_group_timesteps:
                        alpha_teacher, sigma_teacher, _, _ = self.interpolant(teacher_group_timestep)
                        teacher_inputs.append(alpha_teacher * images + sigma_teacher * noises)
                    teacher_input = torch.cat(teacher_inputs, dim=0)
                else:
                    raise NotImplementedError()
            else:
                mask_teacher = None
                if self.teacher_t == "self_flow":
                    teacher_time_base = group_timesteps[0]
                    for group_timestep in group_timesteps[1:]:
                        teacher_time_base = torch.minimum(teacher_time_base, group_timestep)
                    alpha_teacher, sigma_teacher, _, _ = self.interpolant(teacher_time_base)
                    teacher_input = alpha_teacher * images + sigma_teacher * noises
                    time_input_teacher = teacher_time_base.expand(bsz, seq_len, 1, 1)
                else:
                    raise NotImplementedError()

        denoising_loss = mean_flat((model_output_gen - gt_token) ** 2)

        if self.use_align_loss:
            with torch.no_grad():
                xr_t = teacher(
                    teacher_input,
                    time_input_teacher.squeeze(-1).squeeze(-1),
                    y=labels,
                    ad=self.block_out_t,
                    mask=mask_teacher,
                )[1]
            align_loss = self.criterion(xr, xr_t, loss_type=self.loss_type)
            return denoising_loss, align_loss

        return denoising_loss, 0
