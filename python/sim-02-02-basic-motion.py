import numpy as np
from plot_style import FIGURE_DIR, setup_matplotlib

plt = setup_matplotlib()

g = 9.81
h_d = 10.0
k1 = 2.0
k2 = 2.5
c = 0.3

t0 = 0.0
tf = 20.0
dt = 0.01
t = np.arange(t0, tf + dt, dt)
output_dir = FIGURE_DIR
output_dir.mkdir(exist_ok=True)

def dynamics_pd(state):
    h, v = state
    e = h - h_d
    u = g - k1 * e - k2 * v
    return np.array([v, u - g - c * v])

def dynamics_p_only(state):
    h, v = state
    e = h - h_d
    u = g - k1 * e
    return np.array([v, u - g - c * v])

def rk4(f, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(len(t) - 1):
        step = t[i + 1] - t[i]
        k_1 = f(y[i])
        k_2 = f(y[i] + 0.5 * step * k_1)
        k_3 = f(y[i] + 0.5 * step * k_2)
        k_4 = f(y[i] + step * k_3)
        y[i + 1] = y[i] + step * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

    return y

case_1 = rk4(dynamics_pd, np.array([4.0, 0.0]), t)
case_2 = rk4(dynamics_pd, np.array([16.0, 0.0]), t)
case_3 = rk4(dynamics_p_only, np.array([4.0, 0.0]), t)

plt.figure(figsize=(8, 5))
plt.plot(t, case_1[:, 0], label=r"$h(t)$")
plt.axhline(h_d, linestyle="--", label=r"$h_d$")
plt.xlabel(r"$t$")
plt.ylabel(r"$h(t)$")
plt.title(r"Case 1: Initial height below $h_d$")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "fig-02-02-case-01-below-hd.png", dpi=300)
plt.savefig(output_dir / "fig-02-02-case-01-below-hd.pdf")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(t, case_2[:, 0], label=r"$h(t)$")
plt.axhline(h_d, linestyle="--", label=r"$h_d$")
plt.xlabel(r"$t$")
plt.ylabel(r"$h(t)$")
plt.title(r"Case 2: Initial height above $h_d$")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "fig-02-02-case-02-above-hd.png", dpi=300)
plt.savefig(output_dir / "fig-02-02-case-02-above-hd.pdf")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(t, case_3[:, 0], label=r"$h(t)$")
plt.axhline(h_d, linestyle="--", label=r"$h_d$")
plt.xlabel(r"$t$")
plt.ylabel(r"$h(t)$")
plt.title(r"Case 3: No velocity feedback")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "fig-02-02-case-03-no-velocity-feedback.png", dpi=300)
plt.savefig(output_dir / "fig-02-02-case-03-no-velocity-feedback.pdf")
plt.close()
