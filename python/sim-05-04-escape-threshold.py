import numpy as np
from plot_style import FIGURE_DIR, setup_matplotlib

plt = setup_matplotlib()

output_dir = FIGURE_DIR
output_dir.mkdir(exist_ok=True)

def rk4_escape(mu, R, alpha, a0, t_burn, h0, v0, tmax, dt):
    t = np.arange(0.0, tmax + dt, dt)
    y = np.zeros((len(t), 2))
    y[0] = np.array([h0, v0])

    h_esc = alpha * R

    def thrust_accel(time):
        if 0.0 <= time <= t_burn:
            return a0
        return 0.0

    def f(time, Y):
        h, v = Y
        radius = R + h
        gravity = mu / radius**2
        return np.array([v, thrust_accel(time) - gravity])

    for i in range(len(t) - 1):
        h_step = t[i + 1] - t[i]

        k_1 = f(t[i], y[i])
        k_2 = f(t[i] + 0.5 * h_step, y[i] + 0.5 * h_step * k_1)
        k_3 = f(t[i] + 0.5 * h_step, y[i] + 0.5 * h_step * k_2)
        k_4 = f(t[i] + h_step, y[i] + h_step * k_3)

        y[i + 1] = y[i] + h_step * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6.0

        if y[i + 1, 0] <= -0.99 * R:
            y[i + 1:, :] = y[i + 1]
            break

    h = y[:, 0]
    v = y[:, 1]

    tau_esc = np.inf
    for i in range(len(t)):
        if np.all(h[i:] >= h_esc):
            tau_esc = t[i]
            break

    crossed = np.any(h >= h_esc)
    escaped = np.isfinite(tau_esc)

    if escaped:
        status = "threshold escape"
    elif crossed:
        status = "temporary crossing"
    else:
        status = "failed escape"

    energy = 0.5 * v**2 - mu / (R + h)

    tau_energy = np.inf
    for i in range(len(t)):
        if energy[i] >= 0.0 and v[i] > 0.0:
            tau_energy = t[i]
            break

    return {
        "t": t,
        "h": h,
        "v": v,
        "energy": energy,
        "h_esc": h_esc,
        "tau_esc": tau_esc,
        "tau_energy": tau_energy,
        "status": status,
    }

def plot_escape(result, name):
    t = result["t"]
    h = result["h"]
    h_esc = result["h_esc"]

    plt.figure(figsize=(7, 4))
    plt.plot(t, h / 1000.0, label=r"$h(t)$")
    plt.axhline(h_esc / 1000.0, linestyle="--", label=r"$h_{\mathrm{esc}}=\alpha R$")
    plt.xlabel(r"$t$ (s)")
    plt.ylabel(r"$h(t)$ (km)")
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"fig-05-04-{name.replace('_', '-')}.png", dpi=300)
    plt.savefig(output_dir / f"fig-05-04-{name.replace('_', '-')}.pdf")
    plt.close()

# Example parameters: replace these by the planet data you want.
mu = 3.986e14
R = 6.371e6

alpha = 0.1
h0 = 0.0
v0 = 0.0
tmax = 5000.0
dt = 0.5

cases = [
    ("escape_strong_burn", 20.0, 500.0),
    ("temporary_crossing", 14.0, 250.0),
    ("failed_escape", 10.0, 100.0),
]

for name, a0, t_burn in cases:
    result = rk4_escape(
        mu=mu,
        R=R,
        alpha=alpha,
        a0=a0,
        t_burn=t_burn,
        h0=h0,
        v0=v0,
        tmax=tmax,
        dt=dt,
    )

    plot_escape(result, name)

    print(name)
    print("status =", result["status"])
    print("h_esc =", result["h_esc"])
    print("tau_esc =", result["tau_esc"])
    print("tau_energy =", result["tau_energy"])
    print()
