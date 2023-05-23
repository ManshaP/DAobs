r"""Neural networks"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import *
from zuko.nn import LayerNorm


class ResidualBlock(nn.Sequential):
    r"""Creates a residual block."""

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class ContextResidualBlock(nn.Module):
    r"""Creates a residual block with context."""

    def __init__(self, project: nn.Module, residue: nn.Module):
        super().__init__()

        self.project = project
        self.residue = residue

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x + self.residue(x + self.project(y))


class ResMLP(nn.Sequential):
    r"""Creates a residual multi-layer perceptron (ResMLP).

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hidden_features: The number of hidden features.
        activation: The activation function constructor.
        kwargs: Keyword arguments passed to :class:`nn.Linear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = nn.ReLU,
        **kwargs,
    ):
        blocks = []

        for before, after in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            if after != before:
                blocks.append(nn.Linear(before, after, **kwargs))

            blocks.append(
                ResidualBlock(
                    LayerNorm(),
                    nn.Linear(after, after, **kwargs),
                    activation(),
                    nn.Linear(after, after, **kwargs),
                )
            )

        super().__init__(*blocks)

        self.in_features = in_features
        self.out_features = out_features


class UNet(nn.Module):
    r"""Creates a U-Net.

    References:
        | U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
        | https://arxiv.org/abs/1505.04597

    Arguments:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        context: The number of context features.
        hidden_channels: The number of hidden channels.
        hidden_blocks: The number of hidden blocks at each depth.
        kernel_size: The size of the convolution kernels.
        stride: The stride of the downsampling convolutions.
        activation: The activation function constructor.
        spatial: The number of spatial dimensions. Can be either 1, 2 or 3.
        kwargs: Keyword arguments passed to :class:`nn.Conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context: int,
        hidden_channels: Sequence[int] = (32, 64, 128),
        hidden_blocks: Sequence[int] = (2, 3, 5),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        activation: Callable[[], nn.Module] = nn.ReLU,
        spatial: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial = spatial

        # Components
        convolution = {
            1: nn.Conv1d,
            2: nn.Conv2d,
            3: nn.Conv3d,
        }.get(spatial)

        if type(kernel_size) is int:
            kernel_size = [kernel_size] * spatial

        if type(stride) is int:
            stride = [stride] * spatial

        kwargs.update(
            kernel_size=kernel_size,
            padding=[k // 2 for k in kernel_size],
        )

        block = lambda channels: ContextResidualBlock(
            project=nn.Sequential(
                nn.Linear(context, channels),
                nn.Unflatten(-1, (-1,) + (1,) * spatial),
            ),
            residue=nn.Sequential(
                LayerNorm(-(spatial + 1)),
                convolution(channels, channels, **kwargs),
                activation(),
                convolution(channels, channels, **kwargs),
            ),
        )

        # Layers
        heads, tails = [], []
        descent, ascent = [], []

        for i, blocks in enumerate(hidden_blocks):
            if i > 0:
                heads.append(
                    convolution(
                        hidden_channels[i - 1],
                        hidden_channels[i],
                        stride=stride,
                        **kwargs,
                    )
                )

                tails.append(
                    nn.Sequential(
                        LayerNorm(-(spatial + 1)),
                        nn.Upsample(scale_factor=tuple(stride), mode='nearest'),
                        convolution(
                            hidden_channels[i],
                            hidden_channels[i - 1],
                            **kwargs,
                        ),
                    )
                )
            else:
                heads.append(convolution(in_channels, hidden_channels[i], **kwargs))
                tails.append(convolution(hidden_channels[i], out_channels, **kwargs))

            descent.append(nn.ModuleList(block(hidden_channels[i]) for _ in range(blocks)))
            ascent.append(nn.ModuleList(block(hidden_channels[i]) for _ in range(blocks)))

        self.heads = nn.ModuleList(heads)
        self.tails = nn.ModuleList(reversed(tails))
        self.descent = nn.ModuleList(descent)
        self.ascent = nn.ModuleList(reversed(ascent))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        memory = []

        for head, blocks in zip(self.heads, self.descent):
            x = head(x)

            for block in blocks:
                x = block(x, y)

            memory.append(x)

        memory.pop()

        for blocks, tail in zip(self.ascent, self.tails):
            for block in blocks:
                x = block(x, y)

            if memory:
                x = tail(x) + memory.pop()
            else:
                x = tail(x)

        return x


class S4DLayer(nn.Module):
    r"""Creates a diagonal structured state space sequence (S4D) layer.

    References:
        | On the Parameterization and Initialization of Diagonal State Space Models (Gu et al., 2022)
        | https://arxiv.org/abs/2206.11893

    Arguments:
        channels: The number of channels.
        space: The size of the state space.
    """

    def __init__(self, channels: int, space: int = 16):
        super().__init__()

        self.log_dt = nn.Parameter(torch.empty(channels).uniform_(-7.0, -3.0))
        self.a_real = nn.Parameter(torch.full((channels, space), 0.5).log())
        self.a_imag = nn.Parameter(torch.pi * torch.arange(space).expand(channels, -1))
        self.c = nn.Parameter(torch.randn(channels, space, 2))

    def extra_repr(self) -> str:
        channels, space = self.a_real.shape
        return f'{channels}, space={space}'

    def kernel(self, length: int) -> Tensor:
        dt = self.log_dt.exp()
        a = torch.complex(-self.a_real.exp(), self.a_imag)
        c = torch.view_as_complex(self.c)

        a_dt = a * dt[..., None]
        b_c = c * (a_dt.exp() - 1) / a

        power = torch.arange(length, device=a.device)
        vandermonde = (a_dt[..., None] * power).exp()

        return 2 * torch.einsum('...i,...ij', b_c, vandermonde).real

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[-1]
        k = self.kernel(length)

        k = torch.fft.rfft(k, n=2 * length)
        x = torch.fft.rfft(x, n=2 * length)
        y = torch.fft.irfft(k * x)[..., :length]

        return y


class S4DBlock(nn.Module):
    r"""Creates a S4D bidirectional block.

    Arguments:
        channels: The number of channels.
        kwargs: Keyword arguments passed to :class:`S4DLayer`.
    """

    def __init__(self, channels: int, **kwargs):
        super().__init__()

        self.l2r = S4DLayer(channels, **kwargs)
        self.r2l = S4DLayer(channels, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat((
            self.l2r(x),
            self.r2l(x.flip(-1)).flip(-1),
        ), dim=-2)