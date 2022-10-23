import torch
import torch.nn as nn

#import matplotlib.pyplot as plt
#from tqdm.auto import tqdm
import numpy as np
import random
import requests

from PIL import Image
import io


def conv2d(grid, kernel):
    """Utility function to perform a simple per-channel depthwise 2d
    convolution of a filter over a (B, C, H, W) tensor"""
    ch = grid.shape[-3]
    return torch.conv2d(grid, kernel.to(grid.device).repeat(ch, 1, 1, 1), padding=1, groups=ch)

class NCA(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.
        self.sobel_y = self.sobel_x.T

        self.NCA_channels = ch
        self.main = nn.Sequential(
            nn.Conv2d(ch*3, ch*8, 1),
            nn.ReLU(),
            nn.Conv2d(ch*8, ch, 1, bias=True),
        )

        with torch.no_grad():
            for p in self.main[-1].parameters():
                p.zero_()
    
    def forward(self, grid):
        perception = torch.cat([
            grid, conv2d(grid, self.sobel_x), conv2d(grid, self.sobel_y)
        ], dim=-3)
        
        return self.main(perception)



def load_emoji(emoji):
    code = hex(ord(emoji))[2:].lower()
    r = requests.get(f'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u{code}.png?raw=true')
    img = Image.open(io.BytesIO(r.content))
    img.thumbnail((48, 48), Image.ANTIALIAS)
    return img


def to_pil(grid, vmax=1.0, mode="RGB"):
    if type(grid) is torch.Tensor:
        grid = grid.cpu().detach().numpy()

    grid = (grid / vmax * 255).clip(0,255).astype(np.uint8)
    if mode == "L":
        if len(grid.shape) == 2:
            return Image.fromarray(grid, mode=mode)
        if len(grid.shape) == 3 and grid.shape[0] == 1:
            return Image.fromarray(grid[0], mode=mode)
    if mode == "RGB":
        return Image.fromarray(grid[:3, :, :].transpose(1,2,0), mode=mode)
    if mode == "RGBA":
        return Image.fromarray(grid[:4, :, :].transpose(1,2,0), mode=mode)

    raise TypeError("invalid image or something")


def to_bytes(grid, size=256, format='jpeg', **kwargs):
    image = to_pil(grid, **kwargs)
    image = image.resize((size, size), resample=0)
    f = io.BytesIO()
    image.save(f, format)
    return f.getvalue()


class Environment:
    """Utility class to handle growing and training NCA model(s).

    It handles everything to do with inputs and outputs and how to run the
    models. It could be considered the "world" or "environment".

    This specific version implementes having inputs map directly to a known
    target output. For the sake of speed, a number of samples is precomputed
    at initialization time. For the original GrowingNCA, a single example
    is all that's needed.
    """
    def __init__(self, update_p=0.5, life_threshold=0.1, examples_n=1,
                 NCA_channels=16, size=64, device='cpu', batch_size=8,
                 loss_func=nn.MSELoss(reduction='mean'),
                 seed=None, target=None, # np.array (k, size, size) or an emoji
                 emoji_padding=8,
                 ):
        self.update_p = update_p
        self.life_threshold = life_threshold

        self.NCA_channels = NCA_channels
        self.size = size
        self.batch_size = batch_size
        self.loss_func = loss_func

        self.seed = seed
        if self.seed is None:
            self.seed = np.zeros((NCA_channels, size, size), dtype=np.float32)
            self.seed[:, size//2, size//2] = 1.
        self.target = target
        if type(self.target) is str:
            img = load_emoji(self.target)
            self.target = np.zeros((4, size, size), dtype=np.float32)
            self.target[:4, emoji_padding:size-emoji_padding, emoji_padding:size-emoji_padding] = np.array(img).transpose(2, 0, 1) / 255.
            self.target[:3] *= self.target[3]

        samples = []
        for _ in range(examples_n):
            samples.append(self.create_example())

        inputs, outputs = zip(*samples)
        self.input_samples = torch.from_numpy(np.array(inputs)).to(device)
        self.output_samples = torch.from_numpy(np.array(outputs)).to(device)

    def iterate_NCA(self, model, grid):
        assert model.NCA_channels == self.NCA_channels
        update = model(grid)
        if self.update_p < 1:
            update_mask = (torch.rand_like(grid[:, :1]) < self.update_p)
            update *= update_mask
        grid = grid + update

        if self.life_threshold > 0:
            alive_mask = (torch.max_pool2d(grid[:, 3:4], kernel_size=3,
                                           stride=1, padding=1) > self.life_threshold)
            grid = grid*alive_mask
        return grid

    def train_episode(self, model, optimizer, its):
        grids, targets = self.get_sample()

        optimizer.zero_grad()
        for _ in range(random.randint(*its)):
            grids = self.iterate_NCA(model, grids)
        
        loss = self.evaluate_results(grids, targets)
        loss.backward()
        optimizer.step()
        return grids, loss.item()

    def get_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        indices = np.random.choice(
                range(len(self.input_samples)), batch_size).tolist()
        batch_in = self.input_samples[indices]
        batch_out = self.output_samples[indices][:, :3]

        return batch_in, batch_out

    def evaluate_results(self, grids, targets):
        k = targets.shape[1]
        return self.loss_func(grids[:, :k], targets).mean()

    def create_example(self):
        # return (input, output)
        #   input: np array (NCA_channels, size, size) float32
        #   output: np array (k, size, size) float32
        if self.target is None:
            raise TypeError("target cannot be None")
        return self.seed, self.target

    def make_gif(self, model, duration, size):
        grids, _ = self.get_sample(1)
        frames = []
        for _ in range(duration):
            grids = self.iterate_NCA(model, grids).detach()
            image = to_pil(grids[0, :4], vmax=1.0, mode="RGB")
            image = image.resize((size, size), resample=0)
            frames.append(image)

        f = io.BytesIO()
        frames[0].save(f, format='png', append_images=frames[1:],
                       save_all=True, duration=30, loop=0)

        return f.getvalue()




#class TargetEnvironment(Environment):
#    """Environment with a well-defined input mapped to a complete target output."""
#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)
#
#    def evaluate_results(self, grids, targets):
#        raise NotImplementedError()
#        #return self.loss_func(grids, targets).mean()
#
#    def create_example(self):
#        # return (input, output)
#        #   input: np array (NCA_channels, size, size) float32
#        #   output: np array (k, size, size) float32
#        raise NotImplementedError()
#
#
#class ElectrodeEnvironment(Environment):
#    """Environment which implements external application of I/O."""
#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)


