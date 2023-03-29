r"""Markov chains"""

import abc
import jax
import jax.numpy as jnp
import jax.random as rng
import math
import random
import torch

try:
    import jax_cfd.base as cfd
except:
    pass

from functools import partial
from torch import Tensor, Size
from torch.distributions import MultivariateNormal
from typing import *


class MarkovChain(abc.ABC):
    r"""Abstract first-order time-invariant Markov chain class

    Wikipedia:
        https://wikipedia.org/wiki/Markov_chain
        https://wikipedia.org/wiki/Time-invariant_system
    """

    @abc.abstractmethod
    def prior(self, shape: Size = ()) -> Tensor:
        r""" x_0 ~ p(x_0) """

        pass

    @abc.abstractmethod
    def transition(self, x: Tensor) -> Tensor:
        r""" x_i ~ p(x_i | x_{i-1}) """

        pass

    def trajectory(self, x: Tensor, length: int, last: bool = False) -> Tensor:
        r""" (x_1, ..., x_n) ~ \prod_i p(x_i | x_{i-1}) """

        if last:
            for _ in range(length):
                x = self.transition(x)

            return x
        else:
            X = []

            for _ in range(length):
                x = self.transition(x)
                X.append(x)

            return torch.stack(X)


class DampedSpring(MarkovChain):
    r"""Linearized dynamics of a mass attached to a spring, subject to wind and drag."""

    def __init__(self, dt: float = 0.01):
        super().__init__()

        self.mu_0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.Sigma_0 = torch.tensor([1.0, 1.0, 1.0, 1.0]).diag()

        self.A = torch.tensor([
            [1.0, dt, dt**2 / 2, 0.0],
            [0.0, 1.0, dt, 0.0],
            [-0.5, -0.1, 0.0, 0.2],
            [0.0, 0.0, 0.0, 0.99],
        ])
        self.b = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.Sigma_x = torch.tensor([0.1, 0.1, 0.1, 1.0]).diag() * dt

    def prior(self, shape: Size = ()) -> Tensor:
        return MultivariateNormal(self.mu_0, self.Sigma_0).sample(shape)

    def transition(self, x: Tensor) -> Tensor:
        return MultivariateNormal(x @ self.A.T + self.b, self.Sigma_x).sample()


class DiscreteODE(MarkovChain):
    r"""Discretized ordinary differential equation (ODE)

    Wikipedia:
        https://wikipedia.org/wiki/Ordinary_differential_equation
    """

    def __init__(self, dt: float = 0.01, steps: int = 1):
        super().__init__()

        self.dt, self.steps = dt, steps

    @staticmethod
    def rk4(f: Callable[[Tensor], Tensor], x: Tensor, dt: float) -> Tensor:
        r"""Performs a step of the fourth-order Runge-Kutta integration scheme.

        Wikipedia:
            https://wikipedia.org/wiki/Runge-Kutta_methods
        """

        k1 = f(x)
        k2 = f(x + dt * k1 / 2)
        k3 = f(x + dt * k2 / 2)
        k4 = f(x + dt * k3)

        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    @abc.abstractmethod
    def f(self, x: Tensor) -> Tensor:
        r""" f(x) = \frac{dx}{dt} """

        pass

    def transition(self, x: Tensor) -> Tensor:
        for _ in range(self.steps):
            x = self.rk4(self.f, x, self.dt / self.steps)

        return x


class Lorenz63(DiscreteODE):
    r"""Lorenz 1963 dynamics

    Wikipedia:
        https://wikipedia.org/wiki/Lorenz_system
    """

    def __init__(
        self,
        sigma: float = 10.0,  # [9, 13]
        rho: float = 28.0,  # [28, 40]
        beta: float = 8 / 3,  # [1, 3]
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sigma, self.rho, self.beta = sigma, rho, beta

    def prior(self, shape: Size = ()) -> Tensor:
        mu = torch.tensor([0.0, 0.0, 25.0])
        sigma = torch.tensor([
            [64.0, 50.0,  0.0],
            [50.0, 81.0,  0.0],
            [ 0.0,  0.0, 75.0],
        ])

        return MultivariateNormal(mu, sigma).sample(shape)

    def f(self, x: Tensor) -> Tensor:
        return torch.stack((
            self.sigma * (x[..., 1] - x[..., 0]),
            x[..., 0] * (self.rho - x[..., 2]) - x[..., 1],
            x[..., 0] * x[..., 1] - self.beta * x[..., 2],
        ), dim=-1)

    @staticmethod
    def preprocess(x: Tensor) -> Tensor:
        mu = x.new_tensor([0.0, 0.0, 25.0])
        sigma = x.new_tensor([8.0, 9.0, 8.6])

        return (x - mu) / sigma

    @staticmethod
    def postprocess(x: Tensor) -> Tensor:
        mu = x.new_tensor([0.0, 0.0, 25.0])
        sigma = x.new_tensor([8.0, 9.0, 8.6])

        return mu + sigma * x


class NoisyLorenz63(Lorenz63):
    r"""Noisy Lorenz 1963 dynamics"""

    def transition(self, x: Tensor) -> Tensor:
        x = super().transition(x)
        z = 1e-2 * self.dt ** 0.5 * self.f(x) * torch.randn_like(x)

        return x + z


class Lorenz96(DiscreteODE):
    r"""Lorenz 1996 dynamics

    Wikipedia:
        https://wikipedia.org/wiki/Lorenz_96_model
    """

    def __init__(
        self,
        n: int = 32,
        F: float = 16.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n, self.F = n, F

    def prior(self, shape: Size = ()) -> Tensor:
        return torch.randn(*shape, self.n)

    def f(self, x: Tensor) -> Tensor:
        x1, x2, x3 = [torch.roll(x, i, dims=-1) for i in (1, -2, -1)]

        return (x1 - x2) * x3 - x + self.F


class LotkaVolterra(DiscreteODE):
    r"""Lotka-Volterra dynamics

    Wikipedia:
        https://wikipedia.org/wiki/Lotka-Volterra_equations
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        delta: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.alpha, self.beta = alpha, beta
        self.delta, self.gamma = delta, gamma

    def prior(self, shape: Size = ()) -> Tensor:
        return torch.rand(*shape, 2)

    def f(self, x: Tensor) -> Tensor:
        return torch.stack((
            self.alpha - self.beta * x[..., 1].exp(),
            self.delta * x[..., 0].exp() - self.gamma,
        ), dim=-1)


class KolmogorovFlow(MarkovChain):
    r"""2-D fluid dynamics with Kolmogorov forcing

    Wikipedia:
        https://wikipedia.org/wiki/Navier-Stokes_equations
    """

    def __init__(
        self,
        size: int = 256,
        dt: float = 0.01,
        reynolds: int = 1e3,
    ):
        super().__init__()

        grid = cfd.grids.Grid(
            shape=(size, size),
            domain=((0, 2 * math.pi), (0, 2 * math.pi)),
        )

        bc = cfd.boundaries.periodic_boundary_conditions(2)

        forcing = cfd.forcings.simple_turbulence_forcing(
            grid=grid,
            constant_magnitude=1.0,
            constant_wavenumber=4.0,
            linear_coefficient=-0.1,
            forcing_type='kolmogorov',
        )

        dt_min = cfd.equations.stable_time_step(
            grid=grid,
            max_velocity=5.0,
            max_courant_number=0.5,
            viscosity=1 / reynolds,
        )

        if dt_min > dt:
            steps = 1
        else:
            steps = math.ceil(dt / dt_min)

        step = cfd.funcutils.repeated(
            f=cfd.equations.semi_implicit_navier_stokes(
                grid=grid,
                forcing=forcing,
                dt=dt / steps,
                density=1.0,
                viscosity=1 / reynolds,
            ),
            steps=steps,
        )

        def wrap(uv: jax.Array):
            return cfd.initial_conditions.wrap_variables(
                var=tuple(uv),
                grid=grid,
                bcs=(bc, bc),
            )

        @partial(jnp.vectorize, signature='(K)->(C,H,W)')
        def prior(key: rng.PRNGKey) -> jax.Array:
            u, v = cfd.initial_conditions.filtered_velocity_field(
                key,
                grid=grid,
                maximum_velocity=3.0,
                peak_wavenumber=4.0,
            )

            return jnp.stack((u.data, v.data))

        @partial(jnp.vectorize, signature='(C,H,W)->(C,H,W)')
        def transition(uv: jax.Array) -> jax.Array:
            u, v = wrap(uv)
            u, v = step((u, v))

            return jnp.stack((u.data, v.data))

        @partial(jnp.vectorize, signature='(C,H,W)->(L,C,H,W)', excluded=[1])
        def trajectory(uv: jax.Array, length: int) -> jax.Array:
            u, v = wrap(uv)
            _, (u, v) = cfd.funcutils.trajectory(step, length)((u, v))

            return jnp.stack((u.data, v.data), axis=1)

        self._prior = jax.jit(prior)
        self._transition = jax.jit(transition)
        self._trajectory = jax.jit(trajectory, static_argnums=[1])

    def prior(self, shape: Size = ()) -> jax.Array:
        seed = random.randrange(2**32)

        key = rng.PRNGKey(seed)
        keys = rng.split(key, Size(shape).numel())
        keys = keys.reshape(*shape, -1)

        return self._prior(keys)

    def transition(self, x: jax.Array) -> jax.Array:
        return self._transition(x)

    def trajectory(self, x: jax.Array, length: int, last: bool = False) -> jax.Array:
        X = self._trajectory(x, length)

        if last:
            return X[-1]
        else:
            return X

    @staticmethod
    def coarsen(x: jax.Array, r: int = 2) -> jax.Array:
        *batch, h, w = x.shape

        x = x.reshape(*batch, h // r, r, w // r, r)
        x = x.mean(axis=(-3, -1))

        return x

    @staticmethod
    def vorticity(x: Tensor) -> Tensor:
        *batch, _, h, w = x.shape

        y = x.reshape(-1, 2, h, w)
        y = torch.nn.functional.pad(y, (1, 1, 1, 1), mode='circular')

        du, = torch.gradient(y[:, 0], dim=-1)
        dv, = torch.gradient(y[:, 1], dim=-2)

        y = du - dv
        y = y[:, 1:-1, 1:-1]
        y = y.reshape(*batch, h, w)

        return y
