# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "astropy==7.1.0",
#     "jax==0.7.1",
#     "marimo",
#     "matplotlib==3.10.6",
#     "numpy==2.2.6",
#     "plotly==6.3.0",
#     "scipy==1.16.2",
# ]
# ///

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    from astropy import constants as const

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.visualization import quantity_support
    import plotly.graph_objects as go

    quantity_support()
    return const, go, mo, np, plt, u


@app.cell
def _(mo):
    mo.md(
        r"""
    # Warped disk evolution
    In many cases astrophysical disks (from protostellar disks and black hole accretion disks to to galaxy scale disks, so from au to kpc) show a warped structure. 

    ## Tilted rings model
    By warped dtructure we mean that if we describe the disk consisting of a set circular rings with spherical radius $r$, then each normal vector per ring $\hat{l}(r)$ can have different orientation which evolves also over time so $\hat{l}(t,r)$. If we define the inclination angle $\theta(t,r)$ and position angle $\phi(t,r)$ then $\hat{l}(t,r)$ can be defined as

    $$
    \hat{l}= 
    \left( \begin{array}{c}
    \sin\theta \cos \phi \\
    \sin\theta \sin \phi \\
    \cos \theta \end{array} \right)
    $$

    Except for the orientation, rings have several other properties:

    1. Surface density $\Sigma(t,r)$ which corresponds to the mass a ring is "carrying".
    2. Angular velocity, $\Omega (r)$ which depends only on the gravitational potential

    With these we can define the angular momentum per unit area of each ring as

    $$
    \boldsymbol{L} (t,r)=\Sigma (t,r) \, r^2 \, \Omega(r) \, \hat{l}(t,r)
    $$

    For our example we can play with a sigmoid expression for the two angles and a gaussian expression for the surface density. For the angular velocity we use a typical Herniquist potential together with a SMBH Keplerian potential. (see details in the next sections)

    /// details | Kinematics details

    We assume the disk is in a sperical stellar gravity potential described by the Hernquist distribution. 
    Hernquist model depends on the total stellar mass of the bulge ($M_*$) and a scale length ($r_a$) and has a stellar density of 

    \begin{equation*}
        \rho(r)=\frac{M_*}{2 \pi} \frac{r_a}{r} \frac{1}{(r+r_a)^{3}}
    \end{equation*}

    which provide a potential 

    \begin{equation*}
        \Phi(r)=-\frac{G M_*}{r+r_a}
    \end{equation*}

    The rotational velocity is found from solving the centripetal acceleration $\frac{\mathrm{v}^2(r)}{r}=|\vec{F}|$ as

    \begin{equation*}
       \mathrm{v}_\mathrm{Bulge}(r;M_*,r_a) =\sqrt{r\frac{d}{dr} \Phi(r)} =\frac{\sqrt{G M_* r}}{r+r_a}
    \end{equation*}

    In addition to the stellar component as the rings can reach the vicinity of the SMBH we add and a Keplerian component 

    \begin{equation*}
        \mathrm{v}_\mathrm{SMBH}(r) =\sqrt{\frac{G~M_\mathrm{SMBH}}{r}}
    \end{equation*}

    so

    \begin{equation*}
        \mathrm{V}(r;M_*,r_a,M_\mathrm{SMBH})=\sqrt{\mathrm{v}^2_\mathrm{SMBH}(r)+\mathrm{v}^2_\mathrm{Bulge}(r;M_*,r_a)}
    \end{equation*}

    ///
    """
    )
    return


@app.cell
def _(mo):
    theta_io = mo.ui.range_slider(
        start=0, stop=80, step=5, value=[15, 60], label=r"$\theta$ range"
    )
    theta_center_io = mo.ui.slider(
        start=0.5, stop=5, step=0.25, value=3, label=r"$\theta$ center"
    )
    theta_scale_io = mo.ui.slider(
        start=0.5, stop=5, step=0.25, value=2, label=r"$\theta$ scale"
    )

    phi_io = mo.ui.range_slider(
        start=0, stop=80, step=5, value=[20, 40], label=r"$\phi$ range"
    )
    phi_center_io = mo.ui.slider(
        start=0.5, stop=5, step=0.25, value=3, label=r"$\phi$ center"
    )
    phi_scale_io = mo.ui.slider(
        start=0.5, stop=5, step=0.25, value=2, label=r"$\phi$ scale"
    )


    sigma_bck_io = mo.ui.slider(
        start=0.5,
        stop=10,
        step=0.5,
        value=3,
        label=r"Surface density baseline ($\mathrm{M_\odot/kpc}$)",
    )
    sigma_gs_io = mo.ui.slider(
        start=0,
        stop=10,
        step=0.5,
        value=2,
        label=r"Surface density gaussian scale ($\mathrm{M_\odot/kpc}$)",
    )
    sigma_gc_io = mo.ui.slider(
        start=0.5,
        stop=10,
        step=0.5,
        value=4,
        label=r"Surface density gaussian center ($\mathrm{kpc}$)",
    )
    sigma_gd_io = mo.ui.slider(
        start=0.5,
        stop=10,
        step=0.5,
        value=1,
        label=r"Surface density gaussian dispersion ($\mathrm{kpc}$)",
    )

    mo.accordion(
        {
            "Initial Conditions parameters (expand to play)": mo.vstack(
                [
                    mo.hstack([theta_io, theta_center_io, theta_scale_io]),
                    mo.hstack([phi_io, phi_center_io, phi_scale_io]),
                    mo.hstack(
                        [sigma_bck_io, sigma_gs_io, sigma_gc_io, sigma_gd_io]
                    ),
                ]
            )
        }
    )
    return (
        phi_center_io,
        phi_io,
        phi_scale_io,
        sigma_bck_io,
        sigma_gc_io,
        sigma_gd_io,
        sigma_gs_io,
        theta_center_io,
        theta_io,
        theta_scale_io,
    )


@app.cell
def _(
    np,
    phi_center_io,
    phi_io,
    phi_scale_io,
    sigma_bck_io,
    sigma_gc_io,
    sigma_gd_io,
    sigma_gs_io,
    theta_center_io,
    theta_io,
    theta_scale_io,
    u,
):
    rr = np.linspace(0.1, 10.0, 100) * u.kpc


    def theta0(r):
        return (
            theta_io.value[0] * u.deg
            + (theta_io.value[1] - theta_io.value[0])
            / (
                1
                + np.exp(
                    -theta_scale_io.value
                    * (r - theta_center_io.value * u.kpc)
                    / u.kpc
                )
            )
            * u.deg
        )


    theta_expr = f"$\\theta_0(r) = {theta_io.value[0]}+\\frac{{ {theta_io.value[1] - theta_io.value[0]} }} {{1+e^{{-{theta_scale_io.value}(r-{theta_center_io.value})}} }}$"


    def phi0(r):
        return (
            phi_io.value[0] * u.deg
            + (phi_io.value[1] - phi_io.value[0])
            / (
                1
                + np.exp(
                    -phi_scale_io.value * (r - phi_center_io.value * u.kpc) / u.kpc
                )
            )
            * u.deg
        )


    phi_expr = f"$\\phi_0(r) = {phi_io.value[0]}+\\frac{{ {phi_io.value[1] - phi_io.value[0]} }} {{1+e^{{-{phi_scale_io.value}(r-{phi_center_io.value})}} }}$"


    def Sigma0(r):
        return (
            sigma_bck_io.value * u.solMass / u.kpc**2
            + sigma_gs_io.value
            * np.exp(
                -((sigma_gc_io.value - r.value) ** 2) / (2 * sigma_gd_io.value**2)
            )
            * u.solMass
            / u.kpc**2
        )


    # mo.hstack(
    #     [
    #         mo.mpl.interactive(plt.gcf()),
    #     ]
    # )
    # mo.accordion(
    #     {
    #         "Show Initial condition plots": fig  # mo.mpl.interactive(plt.gcf())
    #     },
    #     lazy=True,
    # )
    return Sigma0, phi0, phi_expr, rr, theta0, theta_expr


@app.cell
def _(Sigma0, go, mo, np, phi0, phi_expr, plt, rr, theta0, theta_expr):
    fig, ax = plt.subplot_mosaic(
        """
        AS
        """,
        figsize=(12, 5),
    )
    ax["A"].plot(rr, theta0(rr), label=theta_expr)
    ax["A"].plot(rr, phi0(rr), label=phi_expr)

    ax["A"].legend()
    ax["S"].plot(rr, Sigma0(rr))
    ax["A"].set(title="Angles")
    ax["S"].set(title="Surface Density")


    θ = np.deg2rad(theta0(rr).value)
    φ = np.deg2rad(
        phi0(rr).value
    )  # np.arctan2(-lx0(rr) / np.sin(θ), (rr.value) / np.sin(θ))
    s = Sigma0(rr).value

    # Define the number of points for each circle
    num_points = 100

    # Create a 3D plot with circles for each radius
    fig3d = go.Figure()

    # sigma_min, sigma_max = sigma_values.min(), sigma_values.max()
    # normalized_sigma = (sigma_values - sigma_min) / (sigma_max - sigma_min)

    # Loop over each radius to create a circle
    for i, r in enumerate(rr.value):
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros_like(x)  # All points lie in the xy-plane

        cosθ, sinθ = np.cos(θ[i]), np.sin(θ[i])
        cosφ, sinφ = np.cos(φ[i]), np.sin(φ[i])
        x_rot = x * cosφ + y * sinφ
        y_rot = -x * sinθ * sinφ + y * sinθ * cosφ + z * cosθ
        z_rot = x * cosθ * sinφ - y * cosθ * cosφ + z * sinθ
        sigma_value = Sigma0(rr[i]).value / Sigma0(rr).value.max()
        color_data = np.full(num_points, Sigma0(rr[i]).value)

        # Add the circle to the plot
        fig3d.add_trace(
            go.Scatter3d(
                x=x_rot,
                y=y_rot,
                z=z_rot,
                mode="lines",
                line=dict(
                    width=3,
                    color=color_data,
                    cmin=0.1,
                    cmax=Sigma0(rr).value.max(),
                    colorscale="Greys",
                ),  # , showscale=True),
                showlegend=False,
            )
        )

    # Update layout for better visualization
    fig3d.update_layout(
        scene=dict(
            xaxis_title="X [kpc]",
            yaxis_title="Y [kpc]",
            zaxis_title="Z [kpc]",
        ),
        title="3D Circles for Each Radius",
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Display the plot

    mo.accordion(
        {
            "Plots": fig,
            "Show 3D plot": mo.ui.plotly(fig3d),  # mo.mpl.interactive(plt.gcf())
        },
        #lazy=True,
    )
    return


@app.cell
def _(mo):
    Mhern_io = mo.ui.slider(
        start=0.5,
        stop=15,
        step=0.5,
        value=1,
        label=r"Hernquist (stellar) mass ($\times10^{11}\,\mathrm{M_\odot}$)",
    )
    a_hern_io = mo.ui.slider(
        start=0.5, stop=5.0, step=0.5, value=2, label=r"Hernquist scale (kpc)"
    )
    Mbh_io = mo.ui.slider(
        start=0.5,
        stop=15,
        step=0.5,
        value=4,
        label=r"SMBH mass ($\times10^{8}\,\mathrm{M_\odot}$)",
    )


    sigma_gas_io = mo.ui.slider(
        start=2,
        stop=80,
        step=1,
        value=12,
        label=r"Gas velocity dispersion ($\mathrm{km/s}$) -constant-",
    )

    pars_kinematics = mo.vstack(
        [
            mo.hstack([Mhern_io, a_hern_io]),
            mo.hstack([Mbh_io]),
            # mo.hstack([theta_io, theta_center_io, theta_scale_io]),
            # mo.hstack([phi_io, phi_center_io, phi_scale_io]),
            mo.hstack([sigma_gas_io]),
        ]
    )
    mo.accordion(items={"Kinematics Parameters (expand to play)": pars_kinematics})
    return Mbh_io, Mhern_io, a_hern_io, sigma_gas_io


@app.cell
def _(Mbh_io, Mhern_io, a_hern_io, const, mo, np, plt, rr, u):
    G = const.G.to(u.kpc / u.solMass * (u.km / u.s) ** 2)
    eo = 2
    Mbh = Mbh_io.value * u.solMass
    Mstar = Mhern_io.value * 1e11 * u.solMass
    a_hern = a_hern_io.value * u.kpc
    Vcirc = lambda r: np.sqrt(G * Mbh / r + (G * Mstar * r) / (r + a_hern) ** 2)
    W = lambda r: Vcirc(r) / r
    dW = lambda r: np.gradient(W(r), r, edge_order=eo)
    dr2W = lambda r: np.gradient(r**2 * W(r), r, edge_order=eo)
    dr3W = lambda r: np.gradient(r**3 * W(r), r, edge_order=eo)
    k = lambda r: np.sqrt(
        2 * W(r) / r * np.gradient(r**2 * W(r), r, edge_order=eo)
    )

    figV, axV = plt.subplot_mosaic([["Vc", "W", "k"]], figsize=(16, 5))
    axV["Vc"].plot(rr, Vcirc(rr))
    axV["W"].plot(rr, W(rr))
    axV["k"].plot(rr, k(rr))

    axV["Vc"].set(title="Circular Velocity")
    axV["W"].set(title="Angular Velocity")
    axV["k"].set(title="Epicyclic frequency")

    # mo.mpl.interactive(plt.gcf())
    mo.accordion({"Velocity Plots": figV})
    return W, dW, dr2W, eo, k


@app.cell
def _():
    # θ = np.arccos(lz0(rr))
    # φ = np.arctan2(-lx0(rr) / np.sin(θ), (rr.value) / np.sin(θ))
    # s = Sigma0(rr).value

    # # Define the number of points for each circle
    # num_points = 100

    # # Create a 3D plot with circles for each radius
    # fig3d = go.Figure()

    # # sigma_min, sigma_max = sigma_values.min(), sigma_values.max()
    # # normalized_sigma = (sigma_values - sigma_min) / (sigma_max - sigma_min)

    # # Loop over each radius to create a circle
    # for i, r in enumerate(rr.value):
    #     theta = np.linspace(0, 2 * np.pi, num_points)
    #     x = r * np.cos(theta)
    #     y = r * np.sin(theta)
    #     z = np.zeros_like(x)  # All points lie in the xy-plane

    #     cosθ, sinθ = np.cos(θ[i]), np.sin(θ[i])
    #     cosφ, sinφ = np.cos(φ[i]), np.sin(φ[i])
    #     x_rot = x * cosφ + y * sinφ
    #     y_rot = -x * sinθ * sinφ + y * sinθ * cosφ + z * cosθ
    #     z_rot = x * cosθ * sinφ - y * cosθ * cosφ + z * sinθ
    #     sigma_value = Sigma0(rr[i]).value / Sigma0(rr).value.max()

    #     # Add the circle to the plot
    #     fig3d.add_trace(
    #         go.Scatter3d(
    #             x=x_rot,
    #             y=y_rot,
    #             z=z_rot,
    #             mode="lines",
    #             line=dict(width=2, color=sigma_value, colorscale="Magma"),
    #             showlegend=False,
    #         )
    #     )

    # # Update layout for better visualization
    # fig3d.update_layout(
    #     scene=dict(
    #         xaxis_title="X",
    #         yaxis_title="Y",
    #         zaxis_title="Z",
    #     ),
    #     title="3D Circles for Each Radius",
    #     margin=dict(l=0, r=0, b=0, t=40),
    # )

    # # Display the plot
    # mo.ui.plotly(fig3d)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Disk Kinematic Viscosity
    The kinematic viscosity $\nu$ is related to the dynamic viscosity coefficient $\eta$ and the gas density $\rho$.

    /// Details | Shakura & Sunyaev accretion disk approach 

    ### Shakura & Sunyaev accretion disk approach 
    The most known kinematic viscosity approxiamtion is the Shakura \& Sunyaev accretion disk where we consider a locally isothermal disk with aspect ratio $H / R$, where $H$ is the disk scale height.

    Angular momentum transport in an accretion disk is driven by turbulent eddies with a maximum size $H$, and maximum speed the sound speed, $c_{\mathrm{s}}=H \Omega$. 

    The azimuthal shear viscosity has the standard form

    \begin{equation*}
    \nu_1 (r)=\alpha_1\left(\frac{H}{r}\right)^2 r^2 \Omega
    \end{equation*}

    \begin{equation*}
    \nu_2 (r)=\alpha_2\left(\frac{H}{r}\right)^2 r^2 \Omega,
    \end{equation*}

    (Papaloizou & Pringle 1983), where $\alpha_2 \simeq 1 /(2 \alpha)$ in the linear approximation.

    ### Note:
    **This approximation is used in accretion disks.** As we will note later, on the equations section, the literature is focused 99.9% on accretion and protostellar disks. This makes hard for us to build up and provide a framework for a large-galaxy scale warp. Of course there are **reasons** for this, galaxies are more complicated (they have )

    ///

    /// Details | Cloud fluid approximation

      ### Cloud fluid approximation for a galactic gas disk
    For large scale disks we follow the approxiamtion of Steiman Cameron 1988 approximation.
    We assume that the molecular clouds are close to hard spheres of radius $R_c$ and mass $m$ their dynamic viscosity coefficient is (Reif+1965)

    $$
    \mu= 0.319 \frac{m\sigma}{\sigma _{0}}
    $$

    where $\sigma_{0} =4\pi c R_{c}^2$ and $c<1$ is a factor of the "closeness" to a hard sphere with typical value of $c\sim0.2$ (Scalo+1982).
    Assuming Maxwellian velocity distribution, we get for the kinematic viscocity

    $$
    \nu=0.451 l \sigma
    $$

    where $l =(n \sigma)^{-1}$ is the mean free path of the clouds. 

    The mean path of the cloud can be estimated using the radial extent over a cloud can travel using the epicyclin amplitude $k$.

    Together with the assumption of isotropic velocity dispersion the kinematic viscosity is

    $$
    \nu = 0.26 \frac{\sigma^2}{k}
    $$

    with typical values at 1 kpc

    $$
     \left[ \frac{\nu}{\mathrm{kpc^2\,Gyr^{-1}}} \right]= 0.15 \left[ \frac{\sigma}{\mathrm{20\,km\,s^{-1}}} \right]^{2} \left[ \frac{k}{\mathrm{700\,km\,s^{-1}\,kpc^{-1}}} \right]^{-1}
    $$

    using 

    $$
    \kappa^{2} = \frac{2 \Omega}{R} \frac{d}{d R}\left(R^{2} \Omega\right) 
    $$

    the epicyclic frequency of the disk.

    ///
    """
    )
    return


@app.cell
def _(k, mo, plt, rr, sigma_gas_io, u):
    sigma_gas = sigma_gas_io.value * u.km / u.s
    nu1 = lambda r: (sigma_gas**2 / k(r)).to(u.kpc**2 / u.Gyr)
    nu2 = lambda r: (sigma_gas**2 / k(r)).to(u.kpc**2 / u.Gyr)

    fignu, axnu = plt.subplot_mosaic([["nu1", "nu2", "nu3"]], figsize=(16, 5))
    axnu["nu1"].plot(rr, nu1(rr))
    axnu["nu2"].plot(rr, nu2(rr))

    axnu["nu1"].set(title="nu1")
    axnu["nu2"].set(title="nu2 (=nu1)")
    # mo.mpl.interactive(plt.gcf())
    mo.accordion({"Viscosity plots": fignu})
    return nu1, nu2


@app.cell
def _(Sigma0, W, dW, eo, mo, np, nu1, nu2, phi0, plt, rr, theta0):
    lx0 = lambda r: np.sin(theta0(r)) * np.cos(phi0(r))
    ly0 = lambda r: np.sin(theta0(r)) * np.sin(phi0(r))
    lz0 = lambda r: np.cos(theta0(r))

    dlx0 = lambda r: np.gradient(lx0(r), r, edge_order=eo)
    dly0 = lambda r: np.gradient(ly0(r), r, edge_order=eo)
    dlz0 = lambda r: np.gradient(lz0(r), r, edge_order=eo)

    l0 = lambda r: np.array([lx0(r), ly0(r), lz0(r)])
    dl0 = lambda r: np.array([dlx0(r), dly0(r), dlz0(r)])
    Lx0 = lambda r: Sigma0(r) * r**2 * W(r) * lx0(r)
    Ly0 = lambda r: Sigma0(r) * r**2 * W(r) * ly0(r)
    Lz0 = lambda r: Sigma0(r) * r**2 * W(r) * lz0(r)
    L0 = lambda r: np.sqrt(Lx0(r) ** 2 + Ly0(r) ** 2 + Lz0(r) ** 2)

    G1x0 = lambda r: 2 * np.pi * r**3 * nu1(r) * Sigma0(r) * dW(r) * lx0(r)
    G1y0 = lambda r: 2 * np.pi * r**3 * nu1(r) * Sigma0(r) * dW(r) * ly0(r)
    G1z0 = lambda r: 2 * np.pi * r**3 * nu1(r) * Sigma0(r) * dW(r) * lz0(r)

    G2x0 = lambda r: np.pi * r**3 * nu2(r) * Sigma0(r) * W(r) * dlx0(r)
    G2y0 = lambda r: np.pi * r**3 * nu2(r) * Sigma0(r) * W(r) * dly0(r)
    G2z0 = lambda r: np.pi * r**3 * nu2(r) * Sigma0(r) * W(r) * dlz0(r)

    G3x0 = (
        lambda r: np.pi
        * r**3
        * nu2(r)
        * Sigma0(r)
        * W(r)
        * (ly0(r) * dlz0(r) - lz0(r) * dly0(r))
    )
    G3y0 = (
        lambda r: np.pi
        * r**3
        * nu2(r)
        * Sigma0(r)
        * W(r)
        * (-lx0(r) * dlz0(r) + lz0(r) * dlx0(r))
    )
    G3z0 = (
        lambda r: np.pi
        * r**3
        * nu2(r)
        * Sigma0(r)
        * W(r)
        * (lx0(r) * dly0(r) - ly0(r) * dlx0(r))
    )

    G10 = lambda r: np.sqrt(G1x0(r) ** 2 + G1y0(r) ** 2 + G1z0(r) ** 2)
    G20 = lambda r: np.sqrt(G2x0(r) ** 2 + G2y0(r) ** 2 + G2z0(r) ** 2)
    G30 = lambda r: np.sqrt(G3x0(r) ** 2 + G3y0(r) ** 2 + G3z0(r) ** 2)

    figG, axG = plt.subplot_mosaic([["G1", "G2", "G3"]], figsize=(16, 5))
    axG["G1"].plot(rr, G10(rr))
    axG["G2"].plot(rr, G20(rr))
    axG["G3"].plot(rr, G30(rr))

    axG["G1"].set(title="|G1|", yscale="log")
    axG["G2"].set(title="|G2|", yscale="log")
    axG["G3"].set(title="|G3|", yscale="log")
    # mo.mpl.interactive(plt.gcf())
    mo.accordion(items={"G plots": figG})
    return G10, G20, G30, L0


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Evolution
    We can describe the evolution in time of the rings using **conservation of mass**

    $$
    \frac{\partial \Sigma}{\partial t}+\frac{1}{r} \frac{\partial}{\partial r}\left(r \Sigma V_r\right)=0
    $$

    and **conservation of angular momentum**

    $$
    \frac{\partial \boldsymbol{L}}{\partial t}+\frac{1}{r} \frac{\partial}{\partial r}\left(r V_r \boldsymbol{L}\right)=\frac{1}{r} \frac{\partial \boldsymbol{G}}{\partial r}+\boldsymbol{T}
    $$


    /// Details | radial velocity $V_R$

    Using $L=\dot{\mathbf{L}}\cdot{\hat{l}} = r^2\Omega \dot{\Sigma}$ we can take the angular momentum equation along $\hat{l}$ and multiple mass conservation with $r^2\Omega$ to find the angular velocity

    \begin{align*}
    V_r &=\frac{1}{r} \frac{\frac{\partial G}{\partial r}+T}{\Sigma (r^2\Omega)'}
    \end{align*}

    ///

    /// Details | Internal torques $\boldsymbol{G}$
    ### Internal torques
    We can describe the "internal" torques $\boldsymbol{G}$ due to interaction between neighboring rings due to their relative "misalignments". 

    \begin{align*}
    \boldsymbol{G}_1 &= 2\pi r^3 \nu_1 \Sigma \Omega'\hat{l} && \text{torque tending to spin up (or down) the ring}\\
    \boldsymbol{G}_2 &=\pi r^3 \nu_2 \Sigma \Omega \hat{l}' && \text{torque tending to align the ring with its neighbours, which acts to flatten the disc}\\
    \boldsymbol{G}_3 &=2\pi r^3 \nu_3 \Sigma \Omega \hat{l}\times \hat{l}' && \text{torque tending to make the ring precess if it is misaligned with its neighbours; this leads to the dispersive wave-like propagation of the warp}
    \end{align*}

    $$
    \boldsymbol{G} = \boldsymbol{G}_1 + \boldsymbol{G}_2 + \boldsymbol{G}_3 = 2\pi r^3 \nu_1 \Sigma \Omega'\hat{l} + \pi r^3 \nu_2 \Sigma \Omega \hat{l}' +2\pi r^3 \nu_3 \Sigma \Omega \hat{l}\times \hat{l}'
    $$

    In the literature these torques are using different kinenamtic viscosities (or just coefficients in Ogilvie 1999) but all the literature are for accretion disks. For now we are going to use the same $\nu$ defined previously for large scale (kpc) disks, so $\nu_1=\nu_2=\nu_3=\nu (r)$

    ///

    /// Details | External torques $\boldsymbol{T}$
    ### External torques [assumed zero for now]
    We can estimate the external torques due to asphericity of the stellar potential 

    $$
    \boldsymbol{T}(t,r) = T_{\phi}\begin{pmatrix}
      \cos\phi \\
      -\sin \phi\\
      0
     \end{pmatrix}+ 
     T_{\theta}\begin{pmatrix}
      -\cos \theta \sin\phi \\
      \cos \theta \cos\phi \\
      -\sin \theta
     \end{pmatrix}
    $$

    and 

    $$
    \begin{align}
    T_{\phi}(r,\theta,\phi)=\frac{ \partial  \left\langle\Phi_1\right\rangle}{ \partial \theta }(r,\theta,\phi) && T_{\theta}(r,\theta,\phi)=\frac{ \partial  \left\langle\Phi_1\right\rangle}{ \partial \phi }(r,\theta,\phi)
    \end{align}
    $$

    and 

    $$
    \begin{align}
    &\frac{ \partial  \left\langle\Phi_1\right\rangle}{ \partial \theta } (r,\theta,\phi)= -3 C_{20} \frac{G M r_e^2}{2 r^3}  \sin \theta \cos \theta - 6 C_{22} \frac{G M r_e^2}{2 r^3} \sin \theta \cos \theta \cos(2 \phi)  \\
    &\frac{ \partial  \left\langle\Phi_1\right\rangle}{ \partial \phi } (r,\theta,\phi)= 6 C_{22}\frac{G M r_e^2}{2 r^3} \sin^2\theta \sin(2 \phi)
    \end{align}
    $$

    so,

    $$
    \boldsymbol{T}(t,r) = \frac{3G M r_e^2}{2 r^3}\sin \theta\left(-\left[C_{20} +2 C_{22}\cos(2 \phi) \right] \cos \theta
    \begin{pmatrix}
      \cos\phi \\
      -\sin \phi\\
      0
     \end{pmatrix}+ 
    2 C_{22}\sin\theta \sin(2 \phi)\begin{pmatrix}
      -\cos \theta \sin\phi \\
      \cos \theta \cos\phi \\
      -\sin \theta
     \end{pmatrix}
     \right)
    $$

    ///
    """
    )
    return


@app.cell
def _(G10, G20, G30, L0, Sigma0, dr2W, eo, np, plt, rr, u):
    Vr0 = lambda r: ((DL_dt0(r)) / (Sigma0(r) * dr2W(r))).to(u.km / u.s)

    DL_dt0 = (
        lambda r: 1 / r * np.gradient(G10(r) + G20(r) + G30(r), r, edge_order=eo)
    )  # +T(r)

    t_source = L0(rr) / DL_dt0(rr)
    t_adv = rr / Vr0(rr)
    figL, axL = plt.subplot_mosaic(
        [["L", "Vr", "T"], ["DL", "t", "."]], figsize=(16, 11)
    )
    axL["L"].plot(rr, L0(rr))
    axL["DL"].plot(rr, DL_dt0(rr))
    axL["Vr"].plot(rr, Vr0(rr))

    axL["t"].plot(rr, t_source, label="Source term")
    axL["t"].plot(rr, t_adv, label="Advection term")

    axL["L"].set(title="|L|", yscale="log")
    axL["DL"].set(title=r"|DL/Dt|=|dG/dt/r +T|", yscale="log")
    axL["Vr"].set(title="Vr")
    axL["T"].set(title="T")
    axL["t"].set(title="Timescale (~L/DL/Dt)")
    axL["t"].legend()
    # mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    /// Details | What other people do?
    ## Note: What other people do?

    As we said earlier, investigation of these equation are dominated by the protostellar and BH accretion disks. In these works (such as [Martin 2019]([https://arxiv.org/abs/2109.12035](https://arxiv.org/abs/1902.11073)), [Dullemond 2021](https://arxiv.org/abs/2109.12035)) they separate the equations in two different regimes

    ### Wave-like, where the $\frac{\partial \Sigma}{\partial t}\sim 0$
    In the Shakura & Sunyaev accretion disk parameterization $\alpha \lt H/r$. In this regime the solutions are wave like and described by the equations (from Martin 2019 in a Keplerian potential)

    \begin{align*}
    \frac{\partial G}{\partial t}+\omega l \times G+\alpha \Omega G=\frac{\Sigma H^2 R^3 \Omega^3}{4} \frac{\partial l}{\partial R}
    && \,\text{and}\, &&
    \Sigma R^2 \Omega \frac{\partial l}{\partial t}=\frac{1}{R} \frac{\partial G}{\partial R}+T
    \end{align*}

    where $\omega=\frac{\Omega^2-\kappa^2}{2 \Omega}$

    [So the G evolves also over time? Independently? Why?]


    ### Viscous limit, $\alpha \gt H/r$
    In this limit the equation gets to

    $$
    \begin{align*}
    \frac{\partial L}{\partial t}= & \frac{3}{R} \frac{\partial}{\partial R}\left[\frac{R^{1 / 2}}{\Sigma} \frac{\partial}{\partial R}\left(\nu _1 \Sigma R^{1 / 2}\right) L\right] \\
    & +\frac{1}{R} \frac{\partial}{\partial R}\left[\left(\nu_2 R^2\left|\frac{\partial l}{\partial R}\right|^2-\frac{3}{2} \nu _1\right) L\right] \\
    & +\frac{1}{R} \frac{\partial}{\partial R}\left[\frac{1}{2} \nu_2 R|L| \frac{\partial l}{\partial R}\right]+T
    \end{align*}
    $$

    ### Generalized Equations
    In Martin 2019 they construct a generalized equation just I do by solving $V_r$ and using it inside the other two equations [does not seem so "genious" idea]

    $$
    \begin{aligned}
    &\frac{\partial \Sigma}{\partial t}=-\frac{2}{R} \frac{\partial}{\partial R}\left[\frac{(\partial G / \partial R \cdot l)}{R \Omega}\right]\\
    &\text { and }\\
    &\frac{\partial L}{\partial t}=-\frac{2}{R} \frac{\partial}{\partial R}\left[\left(\frac{(\partial G / \partial R \cdot l)}{\Sigma R \Omega}\right) L\right]+\frac{1}{R} \frac{\partial G}{\partial R}+T .
    \end{aligned}
    $$

    and then they use a new dimensionless parameter $\beta$ to stitch together the two regimes

    $$
    \begin{aligned}
    & \frac{\partial G}{\partial t}+\omega l \times G+\alpha \Omega G+\beta \Omega(G \cdot l) l= \\
    & \quad \frac{\Sigma H^2 R^3 \Omega^3}{4} \frac{\partial l}{\partial R}-\frac{3}{2}(\alpha+\beta) v_1 \Sigma R^2 \Omega^2 l .
    \end{aligned}
    $$
    ///
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Our next steps: Solve the equation
    The next step is to solve numerically the equations 

    $$
    \frac{\partial \Sigma}{\partial t}+\frac{1}{r} \frac{\partial}{\partial r}\left(r \Sigma V_r\right)=0
    $$

    and **conservation of angular momentum**

    $$
    \frac{\partial \boldsymbol{L}}{\partial t}+\frac{1}{r} \frac{\partial}{\partial r}\left(r V_r \boldsymbol{L}\right)=\frac{1}{r} \frac{\partial \boldsymbol{G}}{\partial r}+\boldsymbol{T}
    $$

    using an intermediate step to calculate the radial velocity

    \begin{align*}
    V_r &=\frac{1}{r} \frac{\frac{\partial G}{\partial r}+T}{\Sigma (r^2\Omega)'}
    \end{align*}

    for the approximated values of $\boldsymbol{G}$ for the large scale galactic disks and without external Torques. The equations are stiff and need an implicit/explicit scheme and adaptive timescales. I tried the finite-volume-method, which is the state of the art on conservation laws with no luck. 

    Simpler schemes such as high order spatial finite difference methods and adaptime time integration through >4th order Runge Kutta I had better results but without forcing boundary conditions, which add extreme numerical instabilities.

    If I get it to work, then we need to use the external torques which involve faster timescales towards the center, making the numerical solver even more difficult to solve.

    ### Jax library
    I am using Jax, a python library which gives 3 major advantages

    1. The code gets compiled and highly optimized, so you can write custom code with C or Fortran like efficiency
    2. The compiled code it's GPU ready (so we can get even faster than C or Fortran)
    3. The final model (=resulting orientations, surface densities at specific radii and times given the initial conditions) is auto-differentiable. This means that if for example we have assumed an initial condition like this $\theta _0(r;a,b)=ax+b$ we can get as a result the $\frac{\partial \mathrm{model}(a,b)}{\partial a,b}$ etc. This is extremely useful if we have data, as we solve much much easier the inverse problem. Due to this we can also use generic (for example a neural network) "curves" (for example a generic unknown $\nu (r)$) with hundrends of parameters and given good data get the final result. Also, we can use bayesian inference of the needed parameters as we can understand the parameter space much muich faster by having the derivatives at this space
    """
    )
    return


@app.cell
def _():
    # def d_dr(r, y):
    #     return np.gradient(y, r, edge_order=2)


    # def dYdt(t, Y, r):
    #     S, Lx, Ly, Lz = Y.reshape(4, -1)
    #     r = r * u.kpc
    #     S = S * u.solMass / u.kpc**2
    #     Lx = Lx * u.km / u.s * u.solMass / u.kpc
    #     Ly = Ly * u.km / u.s * u.solMass / u.kpc
    #     Lz = Lz * u.km / u.s * u.solMass / u.kpc

    #     L = np.sqrt(Lx**2 + Ly**2 + Lz**2)

    #     lx = Lx / L
    #     ly = Ly / L
    #     lz = Lz / L

    #     G1x = 2 * np.pi * r**3 * nu1(r) * S * dW(r) * lx
    #     G1y = 2 * np.pi * r**3 * nu1(r) * S * dW(r) * ly
    #     G1z = 2 * np.pi * r**3 * nu1(r) * S * dW(r) * lz

    #     dlx = d_dr(r, lx)
    #     dly = d_dr(r, ly)
    #     dlz = d_dr(r, lz)

    #     G2x = np.pi * r**3 * nu2(r) * S * W(r) * dlx
    #     G2y = np.pi * r**3 * nu2(r) * S * W(r) * dly
    #     G2z = np.pi * r**3 * nu2(r) * S * W(r) * dlz

    #     Gx = G1x + G2x
    #     Gy = G1y + G2y
    #     Gz = G1z + G2z

    #     G_ = np.sqrt(Gx**2 + Gy**2 + Gz**2)
    #     Vr = d_dr(r, G_) / (r * S * d_dr(r, r**2 * W(r)))

    #     dS_dt = -d_dr(r, r * S * Vr) / r
    #     dLx_dt = -d_dr(r, r * Lx * Vr) / r + d_dr(r, Gx) / r
    #     dLy_dt = -d_dr(r, r * Ly * Vr) / r + d_dr(r, Gy) / r
    #     dLz_dt = -d_dr(r, r * Lz * Vr) / r + d_dr(r, Gz) / r

    #     derivatives = np.array([dS_dt, dLx_dt, dLy_dt, dLz_dt])
    #     return derivatives.ravel()
    return


@app.cell
def _():
    # Y0 = np.array([Sigma0(rr), Lx0(rr), Ly0(rr), Lz0(rr)])
    return


@app.cell
def _():
    # from scipy.integrate import solve_ivp
    return


@app.cell
def _():
    # sol = solve_ivp(dYdt,[0,0.2],Y0.ravel(),args=(rr.value,))
    return


@app.cell
def _():
    return


@app.cell
def _():
    # import jax
    # import jax.numpy as jnp
    return


@app.cell
def _():
    # Gj = jnp.array(G.value)
    # rrj = jnp.array(rr)
    # Mbhj = jnp.array(Mbh.value)
    # Mstarj = jnp.array(Mstar.value)
    # a_hernj = jnp.array(a_hern.value)
    # Vcirc_jax = lambda r: jnp.sqrt(Gj * Mbhj / r + (Gj * Mstarj * r) / (r + a_hernj) ** 2)

    # np.testing.assert_almost_equal(Vcirc_jax(rrj), Vcirc(rr).value,decimal=4)
    return


@app.cell
def _():
    # W_jax = lambda r: Vcirc_jax(r) / r
    # dW_jax = lambda r: jnp.gradient(W_jax(r), r)
    # dr2W_jax = lambda r: jnp.gradient(r**2 * W_jax(r), r)
    # dr3W_jax = lambda r: jnp.gradient(r**3 * W_jax(r), r)
    # k_jax = lambda r: jnp.sqrt(
    #     2 * W_jax(r) / r * jnp.gradient(r**2 * W_jax(r), r)
    # )

    # np.testing.assert_almost_equal(k_jax(rrj)[1:-2], k(rr).value[1:-2],decimal=2)
    return


@app.cell
def _():
    # sigma_gasj = jnp.array(sigma_gas_io.value)
    # nu_transform = jnp.array((sigma_gas**2/k(rr)).unit.to(u.kpc**2 / u.Gyr))
    # nu1_jax = lambda r: (sigma_gasj**2 / k_jax(r))*nu_transform
    # nu2_jax = lambda r: (sigma_gasj**2 / k_jax(r))*nu_transform

    # np.testing.assert_almost_equal(nu1_jax(rrj)[1:-2], nu1(rr).value[1:-2],decimal=3)
    # np.testing.assert_almost_equal(nu2_jax(rrj)[1:-2], nu2(rr).value[1:-2],decimal=3)
    return


@app.cell
def _():
    # @jax.jit
    # def jnp_gradient_edge_order4(f, x):
    #     dx=x[6]-x[5]
    #     df = jnp.zeros_like(f)

    #     # Interior points: 4th-order central difference
    #     df = df.at[2:-2].set((f[:-4] - 8*f[1:-3] + 8*f[3:-1] - f[4:]) / (12 * dx))

    #     # Edges: 1st, 2nd, N-2, N-1 use lower-order accurate formulas
    #     df = df.at[0].set((-25*f[0] + 48*f[1] - 36*f[2] + 16*f[3] - 3*f[4]) / (12 * dx))
    #     df = df.at[1].set((-3*f[0] - 10*f[1] + 18*f[2] - 6*f[3] + f[4]) / (12 * dx))
    #     df = df.at[-2].set((-f[-5] + 6*f[-4] - 18*f[-3] + 10*f[-2] + 3*f[-1]) / (12 * dx))
    #     df = df.at[-1].set((3*f[-5] - 16*f[-4] + 36*f[-3] - 48*f[-2] + 25*f[-1]) / (12 * dx))

    #     return df

    # def d_dr_jax(r, y):
    #     return jnp_gradient_edge_order4(y,r)#jnp.gradient(y, r)

    # def dYdt_jax(t, Y, r):
    #     S, Lx, Ly, Lz = Y.reshape(4, -1)

    #     L = jnp.sqrt(Lx**2 + Ly**2 + Lz**2)

    #     lx = Lx / L
    #     ly = Ly / L
    #     lz = Lz / L

    #     G1x = 2 * jnp.pi * r**3 * nu1_jax(r) * S * dW_jax(r) * lx
    #     G1y = 2 * jnp.pi * r**3 * nu1_jax(r) * S * dW_jax(r) * ly
    #     G1z = 2 * jnp.pi * r**3 * nu1_jax(r) * S * dW_jax(r) * lz

    #     dlx = d_dr_jax(r, lx)
    #     dly = d_dr_jax(r, ly)
    #     dlz = d_dr_jax(r, lz)

    #     G2x = jnp.pi * r**3 * nu2_jax(r) * S * W_jax(r) * dlx
    #     G2y = jnp.pi * r**3 * nu2_jax(r) * S * W_jax(r) * dly
    #     G2z = jnp.pi * r**3 * nu2_jax(r) * S * W_jax(r) * dlz

    #     Gx = G1x + G2x
    #     Gy = G1y + G2y
    #     Gz = G1z + G2z

    #     G_ = jnp.sqrt(Gx**2 + Gy**2 + Gz**2)
    #     Vr = d_dr_jax(r, G_) / (r * S * d_dr_jax(r, r**2 * W_jax(r)))

    #     dS_dt = -d_dr_jax(r, r * S * Vr) / r
    #     dLx_dt = -d_dr_jax(r, r * Lx * Vr) / r + d_dr_jax(r, Gx) / r
    #     dLy_dt = -d_dr_jax(r, r * Ly * Vr) / r + d_dr_jax(r, Gy) / r
    #     dLz_dt = -d_dr_jax(r, r * Lz * Vr) / r + d_dr_jax(r, Gz) / r

    #     derivatives = jnp.array([dS_dt, dLx_dt, dLy_dt, dLz_dt])
    #     return derivatives.ravel()

    # @jax.jit
    # def dY_dt_jax_wrapper(t,Y):
    #     return dYdt_jax(t, Y, rrj)
    return


@app.cell
def _():
    # Y0j = jnp.array([Sigma0(rr), Lx0(rr), Ly0(rr), Lz0(rr)])

    # np.testing.assert_almost_equal(Y0j.ravel(), Y0.ravel(),decimal=3)
    return


@app.cell
def _():
    # dY_dt_jax_wrapper(0.0, Y0j.ravel())
    return


@app.cell
def _():
    # sol_jax = solve_ivp(dY_dt_jax_wrapper,[0,0.2],Y0.ravel(),method="LSODA",atol=1e-3,rtol=1e-2,dense_output=True)
    return


@app.cell
def _():
    # sol_jax
    return


@app.cell
def _():
    # ttj =jnp.linspace(0,0.1,10)
    return


@app.cell
def _():
    # S_, Lx_, Ly_, Lz_ = sol_jax.sol(ttj)[:,1].reshape(4, -1)
    # plt.plot(rrj,S_)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
