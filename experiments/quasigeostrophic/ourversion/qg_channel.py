"""
Operators in a channel with rigid walls

v = psi_x = 0

The streamfunction has dirichlet boundary conditions to north and south, and
periodic in x.

Here is how we can take derivatives

## Neumann boundary conditions

DCT-3 should be the dct

f_k = f_0 + 2 * sum(n=1, N-1) f_n * cos(pi (2k + 1) n / 2N)
    = f_0 + 2 * sum(n=1, N-1) f_n * cos(pi n x_k/Ly)
x_k = (2k + 1) / 2N * Ly, k = 0, ... N -1

f'_k = - 2 sum(n=1, N-1) f_n * sin(pi n x_k/Ly) * pi n / Ly
     = - 2 sum(n=1, N-1) f_n * sin(pi n (2k + 1)/2N) * pi n / Ly
     = - 2 sum(n=0, N-2) f_{n+1} * sin(pi (n + 1)(2k + 1)/2N) * pi (n + 1) / Ly
     = 2 sum(n=0, N-2) g_{n+1} * sin(pi (n + 1)(2k + 1)/2N)
     = DST-3(g)

g[n] = f[n+1] pi (n + 1) / Ly, n = 0, ... N - 2
g[N-1] = 0

## Dirichlet boundary conditions

DST-3 represents these boundary conditions

x_k = (2k + 1)/2N * L

y[k] = -1^k f[N-1] + 2 sum(n=0, N-2) f[n] sin(pi (n + 1)(2k + 1)/2N)
     = f[N-1] sin(N  pi  x_k / L) + 2 sum(n=0, N-2) f[n] sin(pi (n + 1) x_k / L)

Take the derivative:

y'(x[k]) = 2 sum(n=0, N-2) f[n] cos(pi (n + 1) x_k / L) pi (n + 1) / L
         = f[N-1] N pi / L cos(N  pi  x_k / L) 
            + 2 sum(n=0, N-2) f[n] cos(pi (n + 1) (2k + 1)/ 2N ) pi (n + 1) / L
         = 2 sum(n=1, N-1) f[n-1] cos(pi n (2k + 1)/ 2N ) pi n / L
         = DCT-3(g)

note the F[N-1] term vanishes

g[0] = 0
g[n] = f[n-1] pi n / L, n = 1, ... N - 1


y''(x[k]) = DST-3(f[n] * (n + 1) / pi / L)


"""

# %%
from collections import deque
import numpy as np
import numpy
import os
import dataclasses
import matplotlib.pyplot as plt
import time
import cupy
import torch.distributed as dist
import cupyx.scipy
import scipy.fft
import tqdm
import typer
import hashlib
import shutil
import logging
import torch
import einops
# from pdes.kuramoto_sivashinsky import initial_condition_image
# from training.domain import Plane


class DIMS:
    C = 0
    B = 1
    Y = 2
    X = 3


@dataclasses.dataclass
class Channel:
    Lx: float = 2 * np.pi
    Nx: int = 100
    Ly: float = np.pi
    Ny: int = 50
    device: str = "cpu"

    def __post_init__(self):
        if self.device == "cuda":
            self._fft = cupyx.scipy.fft
            self._np = cupy
        else:
            self._np = numpy
            self._fft = scipy.fft

        np = self._np

        X = np.arange(self.Nx) * self.Lx / self.Nx
        Y = (2 * np.arange(self.Ny) + 1) / 2 / self.Ny * self.Ly
        self.X, self.Y = np.meshgrid(X, Y)
        assert self.X.shape == self.Y.shape == (self.Ny, self.Nx)

        self.kx = self._fft.fftfreq(self.Nx) * self.Nx * 2 * np.pi / self.Lx
        n = np.arange(self.Ny)
        ky = np.zeros([self.Ny, self.Nx])
        ky[:, 0] = n * np.pi / self.Ly
        ky[:, 1:] = (n[:, None] + 1) * np.pi / self.Ly
        self.ky = ky

        # operators
        self.s2v = 1j * self.kx
        self.s2vort = -(self.kx**2 + self.ky**2)
        self.vort2s = np.where(self.s2vort == 0, 0, 1 / self.s2vort)
        self.ds_x = 1j * self.kx
        self.ds_y = self.ky
        self.du_x = 1j * self.kx
        self.du_y = -self.ky

    def streamfunction_to_spectral(self, f):
        """Transform a field like a streamfunction

        f has shape (Ny, Nx)
        """
        fh = self._fft.fft(f, axis=-1)
        # zonal mean component has Neumann conditions (u = - psi_y = 0)
        fhb = self._fft.idct(fh[..., 0:1], type=3, axis=-2)
        # perturbation has Dirichlet conditions (v = psi_x = 0)
        fh = self._fft.idst(fh[..., 1:], type=3, axis=-2)
        return fhb, fh

    def streamfunction_to_physical(self, fhb, fh):
        """Transform a field like a streamfunction

        f has shape (Ny, Nx)
        """
        fhb = self._fft.dct(fhb, axis=-2, type=3)
        fh = self._fft.dst(fh, axis=-2, type=3)
        fh = self._np.concatenate([fhb, fh], axis=-1)
        return self._fft.ifft(fh, axis=-1)

    def transform_streamfunction(self, f):
        return self._np.concatenate(self.streamfunction_to_spectral(f), axis=-1)

    def inverse_streamfunction(self, fh):
        fhb, fh = fh[..., 0:1], fh[..., 1:]
        return self.streamfunction_to_physical(fhb, fh)

    def inverse_u(self, fh):
        np = self._np

        fhb, fh = fh[..., 0:1], fh[..., 1:]
        # 1,2,3 -> 2,3,0
        fhb = self._fft.dst(-fhb[..., 1:, :], axis=-2, n=fhb.shape[-2], type=3)

        fh = np.roll(fh, 1, axis=-2)
        fh[..., 0, :] = 0
        fh = self._fft.dct(fh, axis=-2, type=3)

        fh = self._np.concatenate([fhb, fh], axis=-1)
        return self._fft.ifft(fh, axis=-1)

    def transform_u(self, u):
        np = self._np

        f = self._fft.fft(u, axis=-1)

        fhb, fh = f[..., 0:1], f[..., 1:]

        fhb = self._fft.idst(-fhb, axis=-2, type=3)
        # 2,3,0 -> 0, 2, 3
        fhb = np.roll(fhb, 1, axis=-2)
        fhb[..., 0, :] = 0

        fh = self._fft.dct(fh, axis=-2, type=3)
        fh = np.roll(fh, -1, axis=-2)
        fh[..., -1, :] = 0
        fh = np.concatenate([fhb, fh], axis=-1)
        return fh

    def del_y(self, f):
        """Compute the gradient in y

        f has shape (Ny, Nx)

        Returns a v-like variable in spectral space
        """
        return f * self.ky

    def del_x(self, f):
        return f * 1j * self.kx

    def jacobian(self, f, g):
        """Jacobian operator

        J(ψ, q) = ∂ψ/∂x * ∂q/∂y - ∂ψ/∂y * ∂q/∂x

        ψ had dirichlet boundary conditions in y and periodic in x

        """
        f_x = self.streamfunction_to_physical(self.del_x(f))
        g_x = self.streamfunction_to_physical(self.del_x(g))
        f_y = self.u_to_physical(self.del_y(f))
        g_y = self.u_to_physical(self.del_y(g))
        return self.streamfunction_to_spectral(f_x * g_y - f_y * g_x)

    def filter(self, f):
        k = scipy.fft.fftfreq(self.Nx)
        f[..., np.abs(k) >= 1 / 3] = 0
        f[..., -(self.Ny // 3) :, :] = 0


def barotropic():
    """Solve barotropic equation
    q_t + J(psi, q) + psi_x beta = 0
    q = Δ ψ
    """
    # KH instability
    domain = Channel(Nx=256, Ny=128, Lx=10, Ly=5, device="cpu")

    # kh instability
    beta = 0
    yp = domain.Y - domain.Ly / 2
    sigma = 0.1
    q = np.exp(-(yp**2) / sigma**2) / sigma * (1 + 0.1 * np.cos(2 * domain.X))

    # point vortices
    beta = 0

    def guassian(y0, sigma=0.10):
        return np.exp(
            -((domain.X - domain.Lx / 2) ** 2 + (domain.Y - y0) ** 2) / sigma**2
        )

    if False:
        q = 100 * (guassian(domain.Ly / 2 * 1.05) - guassian(domain.Ly / 2 * 0.95))

    # rossby wave
    beta = 1.0
    q = (
        0.5
        * np.sin(domain.X * 2 * np.pi / domain.Lx)
        * np.sin(domain.Y * np.pi / domain.Ly)
    )

    # done

    q = domain.streamfunction_to_spectral(q)
    # Time-stepping parameters
    dx = domain.Lx / domain.Nx
    #
    # CFL = U * dt / dx  # time step size
    U = 10.0
    dt = 0.5 * dx / U
    T = 10  # total time
    timesteps = int(T / dt)

    # Function to calculate the Jacobian J(f,g)

    sources = deque([], maxlen=3)

    # AB3 time-stepping loop
    for t in range(0, timesteps):
        # Output or analysis

        # Compute the nonlinear terms in the spectral space
        psi = q / (-domain.kx**2 - domain.ky**2)

        # domain.filter(q)
        K2 = domain.kx**2 + domain.ky**2
        K4 = K2 * K2
        nu = 0.25 / K4.max() / dt
        # domain.filter(q)
        v = domain.del_x(psi)
        f1 = -domain.jacobian(psi, q) - nu * K4 * q - beta * v

        if t % 100 == 0:
            # plt.imshow(np.real(ifft2(-psi1 * 1j * Ky)))
            # plt.contour(np.real(ifft2(q1)), colors='k')
            # plt.imshow(np.real(ifft2(psi2)))
            # u = -domain.u_to_physical(domain.del_y(psi))
            # plt.imshow(np.real(u))
            pv = domain.streamfunction_to_physical(q) + beta * domain.Y
            plt.contourf(
                domain.X, domain.Y, np.real(pv[1] - pv[0]), levels=np.arange(0, 6, 0.25)
            )
            # plt.imshow(np.real(domain.streamfunction_to_physical(f1)))
            plt.colorbar()
            plt.savefig(f"q1_{t//100:03d}.png")
            plt.close("all")
            print(f"Step {t}/{timesteps}")

        if len(sources) == 3:
            sources.popleft()

        sources.append(f1)

        if len(sources) < 3:
            f1 = sources[-1]
        else:
            # AB3 coefficients
            f1 = sources[0] * 5 / 12 + sources[1] * -16 / 12 + sources[2] * 23 / 12

        q = q + dt * f1

        if np.any(np.isnan(q)):
            raise ValueError("NaNs in q1 or q2")

    # Your output or return statements...


def pv_from_streamfunction(domain, psi, F1, F2):
    pv = domain.s2vort * psi
    pv[0] += F1 * (psi[1] - psi[0])
    pv[1] += F2 * (psi[0] - psi[1])
    return pv


def invert_pv(domain, q, F1=1, F2=1):
    # Compute the nonlinear terms in the spectral space
    # the pv inversion
    # The matrix
    # [a b][psi1] = [q1]
    # [c d][psi2]   [q2]
    a = domain.s2vort - F1
    b = F1
    c = F2
    d = domain.s2vort - F2
    # invert the matrix
    # [ (d, -b), (-c, a)] / (ad - bc)
    det = a * d - b * c
    det = np.where(det == 0, 1, det)

    q1, q2 = q
    psi1 = (d * q1 - b * q2) / det
    psi2 = (-c * q1 + a * q2) / det
    psi = np.stack([psi1, psi2], axis=DIMS.C)
    return psi


def qg():
    nplot = 1
    count = -1
    max_time = 400
    for t, psi, u, X, Y in run_simulation(batch_size=1):
        if t > max_time:
            break

        psi = psi[0]
        u = u[0]
        if t % nplot == 0:
            count += 1

            def plot_stream(f, ax=None):
                if ax is None:
                    ax = plt.gca()
                step = 0.5
                bins = np.arange(-5, 5 + step, step)
                im = ax.contourf(
                    X,
                    Y,
                    f,
                    cmap="RdBu_r",
                    levels=bins,
                )
                ax.set_ylim([-20, 20])

                plt.colorbar(im, ax=ax, orientation="horizontal")

            fig, (a, b, c) = plt.subplots(3, 1, figsize=(6, 8))
            # plt.savefig("J.png")
            fig.suptitle(f"t (nondim) = {t: .2f}\n 5 units ~ 1 day")
            plot_stream(psi[0] - psi[1], ax=a)
            a.set_title(r"$\psi_1 - \psi_2$")
            a.set_ylabel("y (nondim)")
            a.set_xlabel("x (nondim)")
            plot_stream(psi[1], ax=b)
            b.set_title(r"$\psi_2$")
            b.set_ylabel("y (nondim)")
            b.set_xlabel("x (nondim)")

            ub = u.mean(axis=-1)
            c.plot(Y.mean(axis=-1), ub[0], label="upper")
            c.plot(Y.mean(axis=-1), ub[1], label="lower")
            c.set_ylim([-1, 1.5])
            c.set_xlim([-20, 20])
            c.set_ylabel("zonal velocity (nondim)")
            c.set_xlabel("y (nondim)")
            c.legend()

            fig.savefig(f"q1_{count:03d}.png")
            plt.close("all")


def _batch_first(x):
    return x.transpose(DIMS.B, DIMS.C, DIMS.Y, DIMS.X)


def run_simulation(
    batch_size: int,
    device="cpu",
    verbose=True,
    check_inversion=True,
    raise_on_nan=False,
    dt=0.05,
    nu=0.01,
):
    r"""Solve barotropic equation
    $$
        \frac{\partial q_k}{\partial t} + J(\psi_k, q_k)
            = -\frac{1}{\tau_d} (-1)^k (\psi_1 - \psi_2 - \psi_R)
              - \frac{1}{\tau_f} \delta_{k2} \nabla^2\psi_k
              - \nu \nabla^4 q_k ,
    $$

    q1 = laplace psi1 + (psi2 - psi1) + beta y
    q2 = laplace psi2 + (psi1 - psi2) + beta y

    $$

    Yields:
    ------
    (time, psi, u, X, Y)

    - psi and u are (b, 2, nx, ny) tensors

    Notes:
    ------

    Performance on an A6000 ada generation

    With a batch size of 512, GPU is much faster

        In [24]: def run_many_steps(n, **kwargs):
    ...:     for k, (time, psi, u, X, Y) in enumerate(run_simulation(**kwargs)):
    ...:         if k  >= n:
    ...:             break
    ...:     torch.cuda.synchronize()
    ...:

        In [25]: %time run_many_steps(50, batch_size=512, device="cuda");
        Step 0 Time 0.00 CFL_advection 0.00 CFL_hyperdiffusion 0.01 Wall_time 0.0 Throughput (time units/second) 0.00
        CPU times: user 3.07 s, sys: 3.49 ms, total: 3.07 s
        Wall time: 3.07 s

        In [26]: %time run_many_steps(50, batch_size=512, device="cpu");
        /home/nbrenowitz/workspace/edm/pdes/qg_channel.py:108: RuntimeWarning: divide by zero encountered in divide
        self.vort2s = np.where(self.s2vort == 0, 0, 1 / self.s2vort)
        Step 0 Time 0.00 CFL_advection 0.00 CFL_hyperdiffusion 0.01 Wall_time 1.5 Throughput (time units/second) 0.00
        CPU times: user 1min 1s, sys: 19.5 s, total: 1min 20s
        Wall time: 1min 20s

    With a smaller batch size the CPU is faster::

        In [27]: %time run_many_steps(50, batch_size=1, device="cpu");
        /home/nbrenowitz/workspace/edm/pdes/qg_channel.py:108: RuntimeWarning: divide by zero encountered in divide
        self.vort2s = np.where(self.s2vort == 0, 0, 1 / self.s2vort)
        Step 0 Time 0.00 CFL_advection 0.00 CFL_hyperdiffusion 0.01 Wall_time 0.0 Throughput (time units/second) 0.00
        CPU times: user 86.7 ms, sys: 0 ns, total: 86.7 ms
        Wall time: 86.1 ms

        In [28]: %time run_many_steps(50, batch_size=1, device="cuda");
        Step 0 Time 0.00 CFL_advection 0.00 CFL_hyperdiffusion 0.01 Wall_time 0.0 Throughput (time units/second) 0.00
        CPU times: user 239 ms, sys: 3.84 ms, total: 243 ms
        Wall time: 242 ms

    The throughput on CPU for a single init is 30 time units / second. Which is about 1 year / minute::

        In [30]: %time run_many_steps(1000, batch_size=1, device="cpu");
        Step 0 Time 0.00 CFL_advection 0.00 CFL_hyperdiffusion 0.01 Wall_time 0.0 Throughput (time units/second) 0.00
        Step 500 Time 25.00 CFL_advection 0.01 CFL_hyperdiffusion 0.01 Wall_time 0.8 Throughput (time units/second) 29.59


        An equivalent volume of data can be read from disk in 0.5 seconds.
        Suggests we should save data separately

    """
    # TODO this routine is sometimes goes unstable for longer predictions around
    # t = 800-1000, should investigate
    if device == "cuda":
        import cupy as np
    else:
        import numpy as np

    tau_d = 100
    tau_f = 15
    beta = 0.196
    sigma = 3.5

    domain = Channel(Nx=64, Ny=128, Lx=46, Ly=68, device=device)

    F1 = F2 = 1.0

    # plane wave (useful for testing that nonlinear terms are small)
    q1 = np.sin(10 * domain.X * 2 * np.pi / domain.Lx) * np.sin(
        domain.Y * np.pi / domain.Ly
    )
    q2 = 0

    # random init
    q1 = 0.001 * np.random.random(size=(batch_size, *domain.X.shape))
    q2 = np.zeros_like(q1)

    yp = domain.Y - domain.Ly / 2
    orig = u = 1.0 / np.cosh(yp / sigma) ** 2
    uh = domain.transform_u(orig)
    u = domain.inverse_u(uh)
    np.testing.assert_allclose(u, orig, atol=1e-12)
    psir = -uh * domain.du_y * domain.vort2s

    # forcing
    d_sponge = np.minimum(domain.Y, domain.Ly - domain.Y)
    d_sponge = 1.0 * np.exp(-(d_sponge**2) / 2 / sigma**2)

    #  prepare inputs
    q = np.stack([q1, q2], axis=DIMS.C)
    q = domain.transform_streamfunction(q)

    # Function to calculate the Jacobian J(f,g)

    sources = deque([], maxlen=3)

    # AB3 time-stepping loop
    start_time = time.time()
    t = -1
    while True:
        t += 1
        # Output or analysis
        if raise_on_nan and np.any(np.isnan(q)):
            raise ValueError("NaNs in q1 or q2")

        # no domain mean vorticity
        q[..., 0, 0] = 0
        domain.filter(q)
        psi = invert_pv(domain, q, F1=F1, F2=F2)

        # check that PV inversion is correct
        q_expected = pv_from_streamfunction(domain, psi, F1=F1, F2=F2)
        if check_inversion:
            mae = np.mean(np.abs(q_expected - q))
            assert mae < 1e-10, mae

        K4 = domain.s2vort**2

        # advection
        u = domain.inverse_u(-psi * domain.ds_y)
        vh = psi * domain.ds_x
        v = domain.inverse_streamfunction(vh)
        q_x = domain.inverse_streamfunction(q * domain.ds_x)
        vort = domain.inverse_streamfunction(psi * domain.s2vort)
        q_y = domain.inverse_u(q * domain.ds_y)
        J = u * q_x + v * q_y

        yield t * dt, _batch_first(
            domain.inverse_streamfunction(psi).real
        ), _batch_first(vort).real, domain.X, domain.Y - domain.Ly / 2

        sponge = -d_sponge * vort

        Jh = domain.transform_streamfunction(J)
        sponge = domain.transform_streamfunction(sponge)

        i = np.arange(2)[:, None, None, None]
        f = (
            -Jh
            - beta * vh
            + sponge
            - domain.s2vort / tau_f * (i == 1) * psi
            - nu * K4 * domain.s2vort * psi
        )
        f[0] += +(psi[0] - psi[1] - psir) / tau_d
        f[1] += -(psi[0] - psi[1] - psir) / tau_d

        if len(sources) == 3:
            sources.popleft()

        sources.append(f)

        if len(sources) < 3:
            f = sources[-1]
        else:
            # AB3 coefficients
            f = sources[0] * 5 / 12 + sources[1] * -16 / 12 + sources[2] * 23 / 12

        q = q + dt * f

        if t % 500 == 0 and verbose:
            dx = domain.Lx / domain.Nx * 2 / 3
            current_time = time.time()
            print(
                f"Step {t}",
                f"Time {t * dt:.2f}",
                f"CFL_advection {np.max(np.abs(u)) * dt / dx:.2f}",
                f"CFL_hyperdiffusion {nu * dt / dx**4:.2f}",
                f"Wall_time {current_time - start_time:.1f}",
                f"Throughput (time units/second) {t  * dt / (current_time - start_time) * batch_size:.2f}",
            )

    # Your output or return statements...


logger = logging.getLogger(__name__)


class QGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sampling_rate: float = 1,
        num_samples: int = 512,
        spinup_samples: int = 200,
        ny: int = 64,
        batch_size: int = 256,
        base_dir: str = ".tmp",
    ):
        # make a unique path based on the mdf5 sum of the function arguments
        args = {
            "sampling_rate": sampling_rate,
            "num_samples": num_samples,
            "spinup_samples": spinup_samples,
            "ny": ny,
            "batch_size": batch_size,
        }
        args_str = str(args).encode()
        hash = hashlib.md5(args_str).hexdigest()

        path = os.path.join(base_dir, hash, "array.npy")
        os.makedirs(os.path.join(base_dir, hash), exist_ok=True)
        if not os.path.exists(path):
            logger.info(f"Cache miss for path: {path}")
            tmppath = os.path.join(base_dir, hash, ".tmp.npy")
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    write_dataset(tmppath, **args)
                    shutil.move(tmppath, path)
                dist.barrier()
            else:
                write_dataset(tmppath, **args)
        else:
            logger.info(f"Cache hit for path: {path}")
        memmap = np.load(path, mmap_mode="r")
        B, T, C, Y, X = memmap.shape

        self._data = memmap.reshape(B * T // 64, 64, C, Y, X)
        self.B, self.T, self.C, self.Y, self.X = memmap.shape

        # attributed needed to build the network
        assert self.Y == self.X
        self.space_dim = self.X * self.Y
        self.time_length = 64
        self.num_channels = 2
        self.label_dim = 0
        self.condition_channels = self.num_channels

    @property
    def domain(self):
        return Plane(nx=self.X, ny=self.Y)

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        out = torch.from_numpy(self._data[idx].copy())
        reshaped = einops.rearrange(out, "t c y x -> c t (y x)")
        condition = initial_condition_image(reshaped)
        labels = torch.zeros([0])
        return reshaped, labels, condition


def get_dataset(batch_size, kwargs):
    return QGDataset(**kwargs)


def write_dataset(
    path: str,
    sampling_rate: float = 1,
    num_samples: int = 1024,
    spinup_samples: int = 200,
    ny: int = 64,
    batch_size: int = 128,
):
    """Write a dataset to disk

    Parameters
    ----------
    path : str
        Path to write dataset
    sampling_rate : float, optional
        Sampling rate, by default 1.0
    num_samples :
        Number of samples to in time per initial condition
    spinup_samples :
        Number of samples to exclude from the beginning of each initial condition, by default 200
    ny :
        number of points in y direction (centered at Ly/2)
    batch_size : int, optional
        Batch size, by default 1
    """
    logger.info(f"Writing QG data with args: {locals()}")
    shape = (batch_size, num_samples, 2, ny, 64)
    mmap = np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=np.float32)
    k = -1
    yoffset = (128 - ny) // 2

    with tqdm.tqdm(total=spinup_samples + num_samples) as pbar:
        for t, psi, u, X, Y in run_simulation(
            batch_size=batch_size, verbose=False, device="cuda", check_inversion=False
        ):
            if t % sampling_rate == 0:
                k += 1

                if k >= spinup_samples + num_samples:
                    break

                pbar.update(1)
                if k >= spinup_samples:
                    mmap[:, k - spinup_samples] = psi[..., yoffset:-yoffset, :].get()


if __name__ == "__main__":
    # barotropic()
    # qg()
    # os.system(
    #     "ffmpeg -y -framerate 24 -i q1_%03d.png -c:v libx264 -pix_fmt yuv420p output.mp4"
    # )

    typer.run(write_dataset(path = 'test_data', batch_size=1))
