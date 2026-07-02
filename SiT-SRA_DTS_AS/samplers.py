import torch
import numpy as np




def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t ** 2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    return 2 * t_cur


def euler_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",  # not used, just for compatability
        mask_ratio=0.0,  # not used, just for compatability
        mask_type="random",  # not used, kept for a shared sampling interface
):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    _dtype = latents.dtype
    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device

    if mask_ratio > 0.0 and mask_ratio < 1.0:
        latent_size = latents.size(-1)
        seq_len = int(latent_size * latent_size / (2 ** 2))  # we use patch_size 2 as default
        rand_order = torch.rand((int(latents.size(0)), seq_len), device=device)
        perm = rand_order.argsort(dim=1)
        group_1 = int(seq_len * mask_ratio)
        mask = torch.zeros((int(latents.size(0)), seq_len), device=device, dtype=torch.bool)
        mask.scatter_(1, perm[:, :group_1], True)
        mask = mask.unsqueeze(-1).unsqueeze(-1)  # (bsz, seq_len, 1, 1)

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y

            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            if mask_ratio > 0.0 and mask_ratio < 1.0:
                kwargs = dict(y=y_cur, mask=mask)
                d_cur = model(torch.cat([model_input] * 2, dim=0).to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs)[0].to(torch.float64)
            else:
                kwargs = dict(y=y_cur)
                d_cur = model(
                    model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                )[0].to(torch.float64)
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
            x_next = x_cur + (t_next - t_cur) * d_cur
            if heun and (i < num_steps - 1):
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    model_input = torch.cat([x_next] * 2)
                    y_cur = torch.cat([y, y_null], dim=0)
                else:
                    model_input = x_next
                    y_cur = y
                time_input = torch.ones(model_input.size(0)).to(
                    device=model_input.device, dtype=torch.float64
                ) * t_next
                if mask_ratio > 0.0 and mask_ratio < 1.0:
                    kwargs = dict(y=y_cur, mask=mask)
                    d_prime = \
                    model(torch.cat([model_input] * 2, dim=0).to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs)[
                        0].to(torch.float64)
                else:
                    kwargs = dict(y=y_cur)
                    d_prime= model(
                        model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                    )[0].to(torch.float64)
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    d_prime_cond, d_prime_uncond = d_prime.chunk(2)
                    d_prime = d_prime_uncond + cfg_scale * (d_prime_cond - d_prime_uncond)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def euler_maruyama_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        mask_ratio=0.0,
        mask_type="random",
):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)

    _dtype = latents.dtype

    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    if (mask_ratio > 0.0 and mask_ratio < 1.0) and mask_type == "fix":
        latent_size = latents.size(-1)
        seq_len = int(latent_size * latent_size / (2 ** 2))  # we use patch_size 2 as default
        rand_order = torch.rand((int(latents.size(0)), seq_len), device=device)
        perm = rand_order.argsort(dim=1)
        group_1 = int(seq_len * mask_ratio)
        mask = torch.zeros((int(latents.size(0)), seq_len), device=device, dtype=torch.bool)
        mask.scatter_(1, perm[:, :group_1], True)
        mask = mask.unsqueeze(-1).unsqueeze(-1)  # (bsz, seq_len, 1, 1)

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next
            if (mask_ratio > 0.0 and mask_ratio < 1.0) and mask_type == "random":
                latent_size = latents.size(-1)
                seq_len = int(latent_size * latent_size / (2 ** 2))  # we use patch_size 2 as default
                rand_order = torch.rand((int(latents.size(0)), seq_len), device=device)
                perm = rand_order.argsort(dim=1)
                group_1 = int(seq_len * mask_ratio)
                mask = torch.zeros((int(latents.size(0)), seq_len), device=device, dtype=torch.bool)
                mask.scatter_(1, perm[:, :group_1], True)
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # (bsz, seq_len, 1, 1)
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            diffusion = compute_diffusion(t_cur)
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            # compute drift
            if mask_ratio > 0.0 and mask_ratio < 1.0:
                kwargs = dict(y=y_cur, mask=mask)
                v_cur = \
                model(torch.cat([model_input] * 2, dim=0).to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs)[
                    0].to(torch.float64)
            else:
                kwargs = dict(y=y_cur)
                v_cur = model(
                    model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                )[0].to(torch.float64)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

            x_next = x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

    # last step
    if (mask_ratio > 0.0 and mask_ratio < 1.0) and mask_type == "random":
        latent_size = latents.size(-1)
        seq_len = int(latent_size * latent_size / (2 ** 2))  # we use patch_size 2 as default
        rand_order = torch.rand((int(latents.size(0)), seq_len), device=device)
        perm = rand_order.argsort(dim=1)
        group_1 = int(seq_len * mask_ratio)
        mask = torch.zeros((int(latents.size(0)), seq_len), device=device, dtype=torch.bool)
        mask.scatter_(1, perm[:, :group_1], True)
        mask = mask.unsqueeze(-1).unsqueeze(-1)  # (bsz, seq_len, 1, 1)
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
    ) * t_cur

    # compute drift
    if mask_ratio > 0.0 and mask_ratio < 1.0:
        kwargs = dict(y=y_cur, mask=mask)
        v_cur = \
            model(torch.cat([model_input] * 2, dim=0).to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs)[
                0].to(torch.float64)
    else:
        kwargs = dict(y=y_cur)
        v_cur = model(
            model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
        )[0].to(torch.float64)
    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur

    return mean_x
