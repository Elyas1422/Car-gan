
from fastapi import FastAPI,Response
import torch
from torch import nn
from fastapi import FastAPI
from io import BytesIO
from PIL import Image
import numpy as np


# Generator structure.
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = 128
        self.fc1 = nn.Linear(self.latent_dim, 128 * (self.image_size // 4) * (self.image_size // 4))
        self.relu1 = nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, latent):
        latent = latent.view(latent.size(0), -1)
        x = self.fc1(latent)
        x = self.relu1(x)
        x = x.view(x.size(0), 128, self.image_size // 4, self.image_size // 4)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.deconv2(x)
        x = self.tanh(x)
        return x

# Load the trained model to be used.
device = torch.device("cpu")
model = Generator(100).to(device)
model.load_state_dict(torch.load("generator.pth",map_location ='cpu'))


# API app
app = FastAPI()

@app.get("/")
async def generate_image():
    # Generate an image
    noise = torch.randn(1, 100).to(device)
    gen_img = model.forward(noise)[0]
    gen_img= gen_img.permute(1, 2, 0).cpu().detach().numpy() * 0.5 + 0.5
    gen_img = np.clip(gen_img * 255, 0, 255).astype(np.uint8)
    image = Image.fromarray(gen_img)

    # Save the generated image to a BytesIO buffer
    buf = BytesIO()
    image.save(buf, format='PNG')
    image_bytes = buf.getvalue()

    return Response(content=image_bytes, media_type="image/png")