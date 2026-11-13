"""
Lid-Driven Cavity Flow: 2D Incompressible Navier-Stokes Solver
==============================================================

Algorithm : SIMPLE (pressure-velocity coupling)
Scheme    : Central Difference Scheme (CDS), forward Euler in time
Grid      : Semi-staggered; u, v at cell corners, p at cell centers
Pressure  : Red-black Gauss-Seidel, zero-gradient (Neumann) walls
Case      : Re = 100, unit square cavity, lid velocity u = 1
Benchmark : Ghia, Ghia & Shin (1982), centerline velocities at Re = 100
Figures   : streamlines, velocity magnitude with vectors,
            centerline comparison with Ghia, residual convergence
Outputs   : the four figures (PNG) and three data files (CSV), written to
            the results folder

Array convention: field[j, i], j = row (y-direction), i = column (x-direction).
Row 0 is the moving lid (y = 1); row n-1 is the bottom wall (y = 0).

Run:  python ldc_cds.py        or, after pip install:  ldc-cds

Python port of LDC_CDS.m
"""

import os

import numpy as np
import matplotlib.pyplot as plt


def main():
    # =========================================================================
    # 1. PARAMETERS
    # =========================================================================
    re       = 100              # Reynolds number
    n        = 101              # grid nodes in each direction
    l        = 1.0              # cavity side length
    h        = l / (n - 1)      # grid spacing
    dt       = 0.001            # time step
    max_iter = 100000           # maximum number of time steps
    max_gs   = 50               # Gauss-Seidel sweeps per time step
    alpha_p  = 0.7              # pressure under-relaxation factor
    tol      = 1e-6             # steady-state tolerance (max change per step)
    out_dir  = 'results'        # folder for the output figures and data files


    # =========================================================================
    # 2. FIELD ALLOCATION
    # =========================================================================
    # velocities at cell corners (n x n); pressure at cell centers with one
    # ghost layer on each side ((n+1) x (n+1))
    u       = np.zeros((n, n))
    v       = np.zeros((n, n))
    u_star  = np.zeros((n, n))
    v_star  = np.zeros((n, n))
    p       = np.zeros((n + 1, n + 1))
    p_prime = np.zeros((n + 1, n + 1))
    b       = np.zeros((n + 1, n + 1))

    # residual histories (trimmed to the steps actually run after the loop)
    res_u    = np.zeros(max_iter)
    res_v    = np.zeros(max_iter)
    res_cont = np.zeros(max_iter)


    # =========================================================================
    # 3. BOUNDARY CONDITIONS
    # =========================================================================
    # lid (row 0) moves with u = 1; left, right and bottom walls are no-slip.
    # the lid is set last so its two corner nodes also take u = 1.
    u[:, 0] = 0; u[:, -1] = 0; u[-1, :] = 0; u[0, :] = 1
    v[:, 0] = 0; v[:, -1] = 0; v[-1, :] = 0; v[0, :] = 0


    # =========================================================================
    # 4. SIMPLE ALGORITHM (time marching to steady state)
    # =========================================================================
    # red-black ordering for the Gauss-Seidel sweep: cells with (j + i) even are
    # red, the rest black. no two cells of the same color are neighbors, so each
    # color can be updated in a single array operation.
    jj, ii = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing='ij')
    red   = (jj + ii) % 2 == 0
    black = ~red

    for it in range(1, max_iter + 1):

        # keep the previous step for the convergence check; reset p'
        u_old = u.copy()
        v_old = v.copy()
        p_prime[:] = 0

        # ---------------------------------------------------------------------
        # 4.1 Momentum predictor: guessed velocities u*, v*
        # ---------------------------------------------------------------------
        # convective terms, central differences. the row index increases
        # downward, so d/dy = (row above - row below) / 2h
        dudx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * h)
        dudy = (u[:-2, 1:-1] - u[2:, 1:-1]) / (2 * h)
        dvdx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * h)
        dvdy = (v[:-2, 1:-1] - v[2:, 1:-1]) / (2 * h)

        # diffusive terms, 5-point Laplacian
        d2u = ((u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / h**2
               + (u[:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]) / h**2)
        d2v = ((v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / h**2
               + (v[:-2, 1:-1] - 2 * v[1:-1, 1:-1] + v[2:, 1:-1]) / h**2)

        # pressure gradient at velocity nodes (from the 4 surrounding cells)
        dpdx = (p[1:n-1, 2:n] + p[2:n, 2:n] - p[1:n-1, 1:n-1] - p[2:n, 1:n-1]) / (2 * h)
        dpdy = (p[1:n-1, 1:n-1] + p[1:n-1, 2:n] - p[2:n, 1:n-1] - p[2:n, 2:n]) / (2 * h)

        # explicit forward-Euler update
        u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (
            -(u[1:-1, 1:-1] * dudx + v[1:-1, 1:-1] * dudy) - dpdx + (1 / re) * d2u)
        v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (
            -(u[1:-1, 1:-1] * dvdx + v[1:-1, 1:-1] * dvdy) - dpdy + (1 / re) * d2v)

        # boundary conditions on the guessed velocities
        u_star[:, 0] = 0; u_star[:, -1] = 0; u_star[-1, :] = 0; u_star[0, :] = 1
        v_star[:, 0] = 0; v_star[:, -1] = 0; v_star[-1, :] = 0; v_star[0, :] = 0

        # ---------------------------------------------------------------------
        # 4.2 Pressure correction
        # ---------------------------------------------------------------------
        # (a) mass imbalance of the guessed field: b = div(u*) / dt
        dustar_dx = (u_star[:-1, 1:] + u_star[1:, 1:]
                     - u_star[:-1, :-1] - u_star[1:, :-1]) / (2 * h)   # right - left
        dvstar_dy = (v_star[:-1, :-1] + v_star[:-1, 1:]
                     - v_star[1:, :-1] - v_star[1:, 1:]) / (2 * h)     # top - bottom
        b[1:-1, 1:-1] = (dustar_dx + dvstar_dy) / dt

        # (b) solve lap(p') = b by red-black Gauss-Seidel
        for gs in range(max_gs):
            for mask in (red, black):              # red cells first, then black
                new = 0.25 * (p_prime[2:, 1:-1] + p_prime[:-2, 1:-1]
                              + p_prime[1:-1, 2:] + p_prime[1:-1, :-2]
                              - b[1:-1, 1:-1] * h**2)
                p_prime[1:-1, 1:-1] = np.where(mask, new, p_prime[1:-1, 1:-1])

            # zero-gradient (Neumann) walls via ghost cells
            p_prime[0, :]  = p_prime[1, :]
            p_prime[-1, :] = p_prime[-2, :]
            p_prime[:, 0]  = p_prime[:, 1]
            p_prime[:, -1] = p_prime[:, -2]

        # (c) under-relaxed pressure update
        p[1:-1, 1:-1] = p[1:-1, 1:-1] + alpha_p * p_prime[1:-1, 1:-1]

        # ---------------------------------------------------------------------
        # 4.3 Velocity correction: u = u* - dt * grad(p')
        # ---------------------------------------------------------------------
        dppdx = (p_prime[1:n-1, 2:n] + p_prime[2:n, 2:n]
                 - p_prime[1:n-1, 1:n-1] - p_prime[2:n, 1:n-1]) / (2 * h)
        dppdy = (p_prime[1:n-1, 1:n-1] + p_prime[1:n-1, 2:n]
                 - p_prime[2:n, 1:n-1] - p_prime[2:n, 2:n]) / (2 * h)

        u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * dppdx
        v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * dppdy

        # boundary conditions on the corrected velocities
        u[:, 0] = 0; u[:, -1] = 0; u[-1, :] = 0; u[0, :] = 1
        v[:, 0] = 0; v[:, -1] = 0; v[-1, :] = 0; v[0, :] = 0

        # ---------------------------------------------------------------------
        # 4.4 Steady-state check and residual recording
        # ---------------------------------------------------------------------
        du = np.abs(u - u_old).max()
        dv = np.abs(v - v_old).max()
        res_u[it - 1]    = du                  # it counts from 1, arrays from 0
        res_v[it - 1]    = dv
        res_cont[it - 1] = np.abs(b).max() * dt    # max |div u*|

        if it % 500 == 0:
            print('step %6d   max change = %.2e' % (it, max(du, dv)))
        if max(du, dv) < tol:
            print('Steady state reached at step %d  (t = %.3f)' % (it, it * dt))
            break

    # trim residual histories to the steps actually run
    res_u    = res_u[:it]
    res_v    = res_v[:it]
    res_cont = res_cont[:it]


    # =========================================================================
    # 5. POST-PROCESSING
    # =========================================================================
    # plotting grid
    x = np.linspace(0, l, n)
    y = np.linspace(0, l, n)
    X, Y = np.meshgrid(x, y)

    # flip so that row 0 is the bottom wall (y increasing), as in the MATLAB plots
    uf = np.flipud(u)
    vf = np.flipud(v)

    # -------------------------------------------------------------------------
    # 5.1 Streamlines
    # -------------------------------------------------------------------------
    # streamfunction psi = integral of u dy (trapezoidal rule, from the bottom wall)
    psi = np.zeros((n, n))
    psi[1:, :] = np.cumsum(0.5 * (uf[1:, :] + uf[:-1, :]) * h, axis=0)

    # contour levels in three groups, because psi spans very different scales:
    # the near-wall region (psi almost zero), the primary vortex, and the corner
    # eddies (psi positive, order 1e-7). the main group follows psi.min() so it
    # fills the vortex whatever Re and grid are used.
    n_main = 18
    outer  = [-1e-10, -1e-7, -1e-5, -1e-4, -5e-4, -1e-3, -2.5e-3, -5e-3]
    main_levels = list(np.linspace(-0.01, 0.985 * psi.min(), n_main))
    corner = [1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 1.5e-3]
    levels = sorted(outer + main_levels + corner)

    fig_stream = plt.figure()
    ax = plt.gca()
    cs = ax.contour(X, Y, psi, levels=levels, colors='k',
                    linewidths=0.7, linestyles='solid')

    # arrowheads on the streamlines: walk along each contour line, place a head
    # every arrow_spacing of arc length, pointing along the local velocity
    # (velocity is tangent to the line, so the head sits on the curve)
    arrow_spacing = 0.09
    head_size = 7.5

    for seg in [sg for lvl in cs.allsegs for sg in lvl]:
        if len(seg) < 3:
            continue
        step = np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1]))
        arc = np.concatenate([[0.0], np.cumsum(step)])    # distance along the line
        if arc[-1] < arrow_spacing:
            continue
        for t in np.arange(arrow_spacing * 0.5, arc[-1], arrow_spacing):
            px, py = seg[np.searchsorted(arc, t)]
            i = int(np.clip(round(px / h), 0, n - 1))     # nearest grid node
            j = int(np.clip(round(py / h), 0, n - 1))
            m = np.hypot(uf[j, i], vf[j, i])
            if m < 1e-12:
                continue
            dx, dy = uf[j, i] / m * 1e-3, vf[j, i] / m * 1e-3
            ax.annotate('', xy=(px + dx, py + dy), xytext=(px - dx, py - dy),
                        arrowprops=dict(arrowstyle='-|>', color='k', lw=0.5,
                                        mutation_scale=head_size,
                                        shrinkA=0, shrinkB=0))

    plt.axis('scaled'); plt.xlim(0, l); plt.ylim(0, l)
    plt.title('Streamlines  (Re = %d)' % re); plt.xlabel('x'); plt.ylabel('y')

    # -------------------------------------------------------------------------
    # 5.2 Velocity magnitude contour with velocity vectors
    # -------------------------------------------------------------------------
    vel_mag = np.sqrt(uf**2 + vf**2)

    # number of arrows across the cavity, and the node stride that gives it
    if re <= 100:
        n_vectors = 17
    elif re <= 250:
        n_vectors = 19
    else:
        n_vectors = 21

    s = max(1, int(np.ceil((n - 1) / (n_vectors - 1))))

    # sample every s-th node
    X_s = X[::s, ::s]
    Y_s = Y[::s, ::s]
    u_s = uf[::s, ::s]
    v_s = vf[::s, ::s]

    # arrow length proportional to |V|, one scale factor for every arrow: the
    # fastest arrow spans arrow_gain times the arrow spacing, the rest scale
    # down in proportion. raise arrow_gain for longer arrows.
    vmax = np.sqrt(u_s**2 + v_s**2).max()
    arrow_gain = 2.6

    fig_vel = plt.figure()
    plt.contourf(X, Y, vel_mag, 60, cmap='jet')
    plt.colorbar()
    plt.quiver(X_s, Y_s, u_s, v_s,
               angles='xy', scale_units='xy', scale=vmax / (arrow_gain * s * h),
               color='w', width=0.0032,
               headwidth=3.5, headlength=4.0, headaxislength=3.6)
    plt.axis('equal')
    plt.title('Velocity Magnitude + Vectors  (Re = %d)' % re)
    plt.xlabel('x'); plt.ylabel('y')

    # -------------------------------------------------------------------------
    # 5.3 Centerline velocities vs Ghia et al. (1982), Re = 100
    # -------------------------------------------------------------------------
    x_ghia = np.array([0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5,
                       0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0])
    y_ghia = np.array([0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5,
                       0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0])
    u_ghia = np.array([0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
                       -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717,
                       0.73722, 0.78871, 0.84123, 1.0])
    v_ghia = np.array([0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507,
                       0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313,
                       -0.08864, -0.07391, -0.05906, 0])

    ic = np.argmin(np.abs(x - 0.5))      # column at x = 0.5  (vertical line)
    jr = np.argmin(np.abs(y - 0.5))      # row    at y = 0.5  (horizontal line)

    fig_center = plt.figure(figsize=(11, 4.5))

    plt.subplot(1, 2, 1)
    plt.plot(uf[:, ic], y, 'b-', linewidth=1.5, label='Present')
    plt.plot(u_ghia, y_ghia, 'ro', label='Ghia et al.')
    plt.xlabel('u'); plt.ylabel('y'); plt.grid(True)
    plt.title('u on vertical centerline (x = 0.5)')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(x, vf[jr, :], 'b-', linewidth=1.5, label='Present')
    plt.plot(x_ghia, v_ghia, 'ro', label='Ghia et al.')
    plt.xlabel('x'); plt.ylabel('v'); plt.grid(True)
    plt.title('v on horizontal centerline (y = 0.5)')
    plt.legend(loc='best')

    plt.tight_layout()

    # quantitative check: interpolate the solution onto Ghia's sample points.
    # np.interp needs an increasing grid (unlike MATLAB's interp1), so the
    # flipped uf, vf are used
    u_here = np.interp(y_ghia, y, uf[:, ic])
    v_here = np.interp(x_ghia, x, vf[jr, :])
    print('\nCenterline check vs Ghia (Re = %d):' % re)
    print('  max |u_present - u_Ghia| = %.4f' % np.abs(u_here - u_ghia).max())
    print('  max |v_present - v_Ghia| = %.4f' % np.abs(v_here - v_ghia).max())

    # -------------------------------------------------------------------------
    # 5.4 Residual convergence history
    # -------------------------------------------------------------------------
    iters = np.arange(1, len(res_u) + 1)

    fig_res = plt.figure()
    plt.semilogy(iters, res_u,    'b-', linewidth=1.2, label='u')
    plt.semilogy(iters, res_v,    'r-', linewidth=1.2, label='v')
    plt.semilogy(iters, res_cont, 'k-', linewidth=1.2, label='continuity')
    plt.xlabel('iterations'); plt.ylabel('residual'); plt.grid(True)
    plt.title('Residual convergence')
    plt.legend(loc='upper right')


    # =========================================================================
    # 6. SAVE RESULTS (figures and data files in out_dir)
    # =========================================================================
    os.makedirs(out_dir, exist_ok=True)

    # the four figures, as PNG at 150 dpi
    figures = {'streamlines': fig_stream, 'velocity': fig_vel,
               'centerlines': fig_center, 'residuals': fig_res}
    for name, fig in figures.items():
        fig.savefig(os.path.join(out_dir, '%s_Re%d.png' % (name, re)),
                    dpi=150, bbox_inches='tight')

    # velocity at every grid node, y increasing (the same arrays as the plots)
    np.savetxt(os.path.join(out_dir, 'velocity_field_Re%d.csv' % re),
               np.column_stack([X.ravel(), Y.ravel(), uf.ravel(), vf.ravel(),
                                vel_mag.ravel()]),
               delimiter=',', fmt=['%.6f', '%.6f', '%.6e', '%.6e', '%.6e'],
               header='x,y,u,v,velocity_magnitude', comments='')

    # centerline velocities at Ghia's points, next to Ghia's values
    np.savetxt(os.path.join(out_dir, 'centerline_vs_ghia_Re%d.csv' % re),
               np.column_stack([y_ghia, u_here, u_ghia, x_ghia, v_here, v_ghia]),
               delimiter=',', fmt='%.5f',
               header='y,u_present,u_ghia,x,v_present,v_ghia', comments='')

    # residual history, one row per time step
    np.savetxt(os.path.join(out_dir, 'residuals_Re%d.csv' % re),
               np.column_stack([iters, res_u, res_v, res_cont]),
               delimiter=',', fmt=['%d', '%.6e', '%.6e', '%.6e'],
               header='iteration,res_u,res_v,res_continuity', comments='')

    print('\nFigures and data saved in: %s' % os.path.abspath(out_dir))

    # show all four figures (the program ends when the windows are closed)
    plt.show()


if __name__ == "__main__":
    main()
