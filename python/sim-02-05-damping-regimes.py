import numpy as np
from plot_style import FIGURE_DIR, setup_matplotlib


plt = setup_matplotlib()
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11.5,
})


def rk4_step(e, w, dt, k1, gamma):
    def f(y):
        e_val, w_val = y
        return np.array([w_val, -gamma * w_val - k1 * e_val])

    y = np.array([e, w])
    k_1 = f(y)
    k_2 = f(y + 0.5 * dt * k_1)
    k_3 = f(y + 0.5 * dt * k_2)
    k_4 = f(y + dt * k_3)
    return y + (dt / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)


def solve_error(k1, gamma, e0=3.0, w0=0.0, tf=8.0, dt=0.005):
    t = np.arange(0.0, tf + dt, dt)
    e = np.zeros_like(t)
    w = np.zeros_like(t)
    e[0] = e0
    w[0] = w0

    for i in range(len(t) - 1):
        e[i + 1], w[i + 1] = rk4_step(e[i], w[i], dt, k1, gamma)

    return t, e


k1 = 4.0
cases = [
    ("Overdamped", 5.0, "fig-02-05-overdamped"),
    ("Critically damped", 4.0, "fig-02-05-critical"),
    ("Underdamped", 1.2, "fig-02-05-underdamped"),
]


def draw_single(title, gamma, filename):
    t, e = solve_error(k1, gamma)
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(t, e, linewidth=2.2, label=rf"$\gamma={gamma:g}$")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$e(t)$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{filename}.png", dpi=300)
    plt.savefig(FIGURE_DIR / f"{filename}.pdf")
    plt.close()


for title, gamma, filename in cases:
    draw_single(title, gamma, filename)


fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
for ax, (title, gamma, _) in zip(axes, cases):
    t, e = solve_error(k1, gamma)
    ax.plot(t, e, linewidth=2.2, label=rf"$\gamma={gamma:g}$")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(r"$t$")
    ax.legend()

axes[0].set_ylabel(r"$e(t)$")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "fig-02-05-three-regimes-side-by-side.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "fig-02-05-three-regimes-side-by-side.pdf", bbox_inches="tight")


plt.figure(figsize=(7.0, 4.4))
for title, gamma, _ in cases:
    t, e = solve_error(k1, gamma)
    plt.plot(t, e, linewidth=2.2, label=rf"{title}, $\gamma={gamma:g}$")

plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
plt.xlabel(r"$t$")
plt.ylabel(r"$e(t)$")
plt.title("Three damping regimes")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fig-02-05-three-regimes.png", dpi=300)
plt.savefig(FIGURE_DIR / "fig-02-05-three-regimes.pdf")
