#data transforms
import torch
import torchvision.transforms as transforms
def get_transforms(res):
    return transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),inplace=True)
    ])
def encode_pixels(vae, x):  # raw pixels => raw latents
   # x = torch.cat([vae.encode(x)['latent_dist'].mean, vae.encode(x)['latent_dist'].std], dim=1)
    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
    return x
def sample_posterior(moments, latents_scale=1., latents_bias=0.):

    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z
def get_f(model,vae,image, res,device,label_id,layersout,time,baseline=False,legacy=True):
    transform = get_transforms(res)
    image = transform(image).unsqueeze(0).to(device)
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
    ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
    ).view(1, 4, 1, 1).to(device)
    with torch.no_grad():
        if baseline:
            f = encode_pixels(vae,image)
        else:
            x = torch.cat([vae.encode(image)['latent_dist'].mean, vae.encode(image)['latent_dist'].std], dim=1)
            f = sample_posterior(x, latents_scale, latents_bias)
        labels=torch.tensor([label_id],device=image.device)
        time=torch.tensor([time],device=image.device)
        noise = torch.randn_like(f)
        f = (1 - time) * f + time * noise
        # Because the SiT baseline do not use inverse t,so the time input should be (1-time), but in our early experiments, we mistakenly use time as the input.
        # We find this do not affect the final PCA observation a lot, but we still add a legacy option to help to reproduce the results in our paper.
        if baseline:
            if legacy:
                f = model(f, time, labels, layersout)[1]
            else:
                f = model(f, (1 - time), labels, layersout)[1]
        else:
            f = model(f, time, labels, layersout)[1]

    return f