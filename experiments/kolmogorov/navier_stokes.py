r"""
Solve the 2D navier stokes in vorticity-streamfunction formulation

    q = Δ𝜓 
    q_t = J(𝜓, q)

    J(𝜓, q) = 𝜓_x q_y - 𝜓_y q_x

The velocities can be inferend from the stream function 𝜓 like this:

    (u, v) = (-𝜓_y, 𝜓_x)

The vorticity is defined as 

    q = v_x - u_y
djf

Unless specified all variables are in spectral space.
"""
import subprocess
import sys

from collections import deque
import os
import numpy as np
savepath = '/lustre/fsw/portfolios/nvr/projects/nvr_earth2_e2/sda/kolmogorov/data/hf/'
os.mkdir(savepath)

# batchno = int(sys.argv[1])
for batchno in np.arange(10):
    class Cases:
        KH = "kelvin_helmholtz"
        kolmogorov = "kolmogorov"

    
    # Define the number of grid points, domain size, and other parameters
    N = 256
    use_cupy = True
    n_inits = 100  # can increase for more simulataneous initial conditions
    case = Cases.kolmogorov
    Re = 1000
    dt = 0.001  # time step size 
    # in rozet's code the output dt=0.2 which means that if we want to simulate at the same output we need nstep = 0.2//dt and need to reduce T to something like 64 + 64*0.2 = 64 + 12.8  = 72.8 ooor 25.6 if we use less initial time
    nstep = 0.2 // dt
    # nstep = 1 // dt
    # T = 128  # total time 
    T = 25.6
    # T = 256

    if use_cupy:
        from cupyx.scipy.fft import fftfreq, fft2, ifft2
        import cupy as np
    else:
        import numpy as np
        from scipy.fft import fftfreq, fft2, ifft2


    Lx = Ly = 2 * np.pi
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
            np.exp(-((Y - Ly / 2) ** 2) / 0.025**2)
            / 0.025
            * (1 + 0.01 * np.random.randn(n_inits, N, N))
        )
        source = np.zeros_like(q)
    elif case == Cases.kolmogorov:
        q = np.random.randn(n_inits, N, N)
        q -= q.mean()
        source = np.cos(4 * X) * 4
        source -= source.mean()
    else:
        raise NotImplementedError(case)


    q = to_spectral(q)
    source = to_spectral(source)
    # AB3 coefficients
    ab3_coeff = np.array([23, -16, 5]) / 12

    # Time-stepping parameters
    timesteps = int(T / dt)

    # 2/3 filter
    kx, ky = np.meshgrid(fftfreq(N), fftfreq(N))
    mask_x = np.abs(kx) < 1 / 3
    mask_y = np.abs(ky) < 1 / 3
    filter_ = mask_x & mask_y


    sources = deque([], maxlen=3)
    out = np.zeros((n_inits,1,2, N, N))

    # AB3 time-stepping loop
    frame = 0
    for t in range(0, timesteps):
        # Output or analysis
        if np.any(np.isnan(q)):
            print("NaN detected")
            break
        q = filter_ * q

        # Compute the nonlinear terms in the spectral space
        psi = -q / K2

        u = -1j * Ky * psi
        v = 1j * Kx * psi

        q_x = 1j * Kx * q
        q_y = 1j * Ky * q

        # .real is important to avoid accumulating errors in the complex component
        ug = to_space(u).real
        vg = to_space(v).real
        J = to_spectral(to_space(q_x).real * ug + to_space(q_y).real * vg)

        f = -J + source - q /10 #10
        if t % nstep == 0:
            # filter

            import matplotlib.pyplot as plt

            # plt.imshow(np.real(ifft2(-psi1 * 1j * Ky)))
            # plt.contour(np.real(ifft2(q1)), colors='k')
            plotme = np.real(to_space(q)[0])
            # if use_cupy:
            #     plotme = plotme.get()
            # plt.imshow(plotme, cmap="RdBu_r", vmin=-10, vmax=10)
            # plt.colorbar()
            # plt.savefig(f"data/q1_{frame:06d}.png")
            frame += 1
            plt.close("all")
            cfl = max(np.max(ug.real), np.max(vg.real)) * dt / dx
            enstrophy = np.sum(plotme**2) * dx * dy
            print(f"Step {t*dt}", "CFL", cfl, "enstrophy", enstrophy)
            if frame>64:#960: 
                thisframe = np.expand_dims(np.stack((ug, vg), axis =1 ), 1)
                np.save(savepath + f'uv_b{batchno:02d}_t{frame:06d}.npy', thisframe)

            # out = np.concatenate((out, thisframe), axis = 1)
            # print(out.shape)

        


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
        q = q / (1 + dt * K2 / Re)

    print(out.shape)

    # np.save('data/our_impl/uv.npy', out)
    # Your output or return statements...

    # # combine the frames into a mp4
    # os.system(
    #     "ffmpeg -y -framerate 24 -i q1_%06d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4"
    # )
