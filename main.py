import torch
from torch.autograd import Variable

import numpy as np
from PIL import Image

from stylegan3 import dnnlib, legacy
from CLIP import clip


def deprocess_sd(img):
    img = img.squeeze(0)
    img = img.permute(1, 2, 0)
    img = img * 127.5 + 128
    img = img.clamp(0, 255)
    img = img.to(torch.uint8)
    return img


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

model, preprocess = clip.load("ViT-B/32", device=device)
text = clip.tokenize(["a black cat"]).to(device)

with dnnlib.util.open_url("temp/pkl/stylegan3-r-afhqv2-512x512.pkl") as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)
    z = torch.randn([1, G.z_dim]).to(device)
    for i in range(1000):
        z = Variable(z, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=0.01)
        optimizer.zero_grad()
        img = G(z, None)
        image = torch.nn.functional.interpolate(img, (224, 224))
        loss, _ = model(image, text)
        loss = 1 / loss
        loss = loss[0][0]
        print(loss.detach().cpu().numpy())
        loss.backward()
        optimizer.step()
        
        if i % 30 == 0 or i == 1000 - 1:
            img0 = deprocess_sd(img)
            img0 = img0.detach().cpu().numpy()
            img0 = Image.fromarray(img0, mode="RGB")
            img0.save(f"temp/image/{i}.jpg", "JPEG")
