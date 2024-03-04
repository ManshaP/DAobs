r"""
Solve the 2D navier stokes in vorticity-streamfunction formulation

    q = Δ𝜓 
    q_t = J(𝜓, q)

    J(𝜓, q) = 𝜓_x q_y - 𝜓_y q_x

The velocities can be inferend from the stream function 𝜓 like this:

    (u, v) = (-𝜓_y, 𝜓_x)

The vorticity is defined as 

    q = v_x - u_y


Unless specified all variables are in spectral space.
"""
from collections import deque
import os


class Cases:
    KH = "kelvin_helmholtz"
    kolmogorov = "kolmogorov"


# Define the number of grid points, domain size, and other parameters
N = 256
use_cupy = True
n_inits = 1  # can increase for more simulataneous initial conditions
case = Cases.kolmogorov

if use_cupy:
    from cupyx.scipy.fft import fftfreq, fft2, ifft2, fftshift, ifftshift
    import cupy as np
else:
    import numpy as np
    from scipy.fft import fftfreq, fft2, ifft2, fftshift, ifftshift


Lx = Ly = 5 * 2 * np.pi
dx = Lx / N
dy = Ly / N
x = np.linspace(0, Lx, N, endpoint=False)
y = np.linspace(0, Ly, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Define wavenumbers for spectral method
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N, d=dy) * 2 * np.pi
Kx, Ky = np.meshgrid(kx, ky)
K2 = Kx**2 + Ky**2
K2[0, 0] = 1.0
K4 = K2**2
nu = 5e-6


def to_spectral(f):
    return fft2(f, axes=(-2, -1))


def to_space(f):
    return ifft2(f, axes=(-2, -1))


# Define the initial conditions for q1, q2, and other necessary fields
def guassian(y0):
    return np.exp(-((X - Lx / 2) ** 2 + (Y - y0) ** 2) / 0.25**2)


# kh
if case == Cases.KH:
    q = (
        np.exp(-((Y - Ly / 2) ** 2) / 0.10**2)
        / 0.10
        * (1 + 0.01 * np.random.randn(n_inits, N, N))
    )
elif case == Cases.kolmogorov:
    q = 0.01 * np.random.randn(n_inits, N, N) 
else:
    raise NotImplementedError(case)


q = to_spectral(q)
# AB3 coefficients
ab3_coeff = np.array([23, -16, 5]) / 12

# Time-stepping parameters
dt = 0.03  # time step size
T = 100  # total time
timesteps = int(T / dt)


# Function to calculate the Jacobian J(f,g)
def jacobian(f, g):
    f_x = 1j * Kx * f
    f_y = 1j * Ky * f

    g_x = 1j * Kx * g
    g_y = 1j * Ky * g

    # dealiasing
    N = f.shape[-1]
    n = (3 * N) // 2 + 1

    def pad_to(f, n):
        shifted = fftshift(f, axes=(-2, -1))
        assert f.ndim == 3
        m = f.shape[-1]
        assert f.shape[-2] == m
        assert n >= m

        min_m = m // 2
        max_m = (m - 1) // 2
        min_n = n // 2
        max_n = (n - 1) // 2

        pad = [(0, 0)] + [(min_n - min_m, max_n - max_m)] * 2

        padded = np.pad(shifted, pad, mode="constant")
        return ifftshift(padded, axes=(-2, -1))

    def unpad(f, m):
        shifted = fftshift(f, axes=(-2, -1))
        n = f.shape[-1]

        min_m = m // 2
        max_m = (m - 1) // 2
        min_n = n // 2
        max_n = (n - 1) // 2

        a = min_n - min_m
        b = max_n - max_m
        idx = slice(a, -b)
        return ifftshift(shifted[:, idx, idx], axes=(-2, -1))

    def inverse_transform(f):
        return to_space(pad_to(f, n))

    def forward_transform(x):
        return unpad(to_spectral(x), N)

    J = forward_transform(
        inverse_transform(f_x) * inverse_transform(g_y)
        - inverse_transform(f_y) * inverse_transform(g_x)
    )

    return -J


kx, ky = np.meshgrid(fftfreq(N), fftfreq(N))
mask_x = np.abs(kx) <= 2 / 3 / 2
mask_y = np.abs(ky) <= 2 / 3 / 2
M = mask_x & mask_y


sources = deque([], maxlen=3)

# AB3 time-stepping loop
frame = 0
for t in range(0, timesteps):
    # Output or analysis
    assert not np.any(np.isnan(q)), t

    # Compute the nonlinear terms in the spectral space
    psi = -q / K2

    if t % 50 == 0:
        import matplotlib.pyplot as plt

        # plt.imshow(np.real(ifft2(-psi1 * 1j * Ky)))
        # plt.contour(np.real(ifft2(q1)), colors='k')
        plotme = np.real(to_space(q)[0])
        if use_cupy:
            plotme = plotme.get()
        plt.imshow(plotme, vmin=-10, vmax=10, cmap="RdBu_r")
        plt.savefig(f"q1_{frame:06d}.png")
        frame += 1
        plt.close("all")
        print(f"Step {t}/{timesteps}")

    f = jacobian(psi, q) - to_spectral( 1/2 * np.sin(Y/2))

    if len(sources) == 3:
        sources.popleft()

    sources.append(f)

    if len(sources) < 3:
        f = sources[-1]
    else:
        # AB3 coefficients
        f = sources[0] * 5 / 12 + sources[1] * -16 / 12 + sources[2] * 23 / 12

    q = q + dt * f

    # hyperdiffusion (backward euler)
    # q[n+1] = q[n] - nu k^4 dt q[n+1]
    # q[n+1] = 1/(1 + nu k^4 dt) q[n]
    # nu = 0
    q = q / (1 + nu * dt * K4)


# Your output or return statements...

# combine the frames into a mp4
os.system(
    "ffmpeg -y -framerate 24 -i q1_%06d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4"
)
