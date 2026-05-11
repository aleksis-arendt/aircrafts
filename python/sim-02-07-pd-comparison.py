import numpy as np
from plot_style import FIGURE_DIR, setup_matplotlib

plt = setup_matplotlib()

h_d = 10.0
t0 = 0.0
tf = 20.0
dt = 0.01
t = np.arange(t0, tf + dt, dt)

output_dir = FIGURE_DIR
output_dir.mkdir(exist_ok=True)

e0 = 3.0
w0 = 0.0

q = 1.0
p = 1.0
r = 0.2
c = 0.3

k1_var = np.sqrt(q / r)
k2_var = np.sqrt(p / r + c**2 + 2 * np.sqrt(q / r)) - c

def rk4(k1, k2, c, e0, w0, t):
    gamma = k2 + c
    y = np.zeros((len(t), 2))
    y[0] = np.array([e0, w0])

    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]

        def f(Y):
            e, w = Y
            return np.array([w, -gamma * w - k1 * e])

        k_1 = f(y[i])
        k_2 = f(y[i] + 0.5 * h * k_1)
        k_3 = f(y[i] + 0.5 * h * k_2)
        k_4 = f(y[i] + h * k_3)

        y[i + 1] = y[i] + h * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6.0

    return y[:, 0], y[:, 1]

cases = [
    ("variational_induced", rf"Induced by $p,q,r$: $(k_1,k_2)=({k1_var:.3f},{k2_var:.3f})$", k1_var, k2_var),
    ("pd_4_1", r"PD feedback: $(k_1,k_2)=(4,1)$", 4.0, 1.0),
    ("pd_4_4", r"PD feedback: $(k_1,k_2)=(4,4)$", 4.0, 4.0),
    ("pd_4_8", r"PD feedback: $(k_1,k_2)=(4,8)$", 4.0, 8.0),
]

results = []

for filename, title, k1, k2 in cases:
    e, w = rk4(k1, k2, c, e0, w0, t)
    h_values = h_d + e
    results.append((filename, title, k1, k2, e, w, h_values))

for filename, title, k1, k2, e, w, h_values in results:
    plt.figure(figsize=(6, 4))
    plt.plot(t, h_values, label=r"$h(t)$")
    plt.axhline(h_d, linestyle="--", label=r"$h_d$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$h(t)$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"fig-02-07-{filename.replace('_', '-')}.png", dpi=300)
    plt.savefig(output_dir / f"fig-02-07-{filename.replace('_', '-')}.pdf")
    plt.close()

plt.figure(figsize=(8, 5))
for filename, title, k1, k2, e, w, h_values in results:
    plt.plot(t, h_values, label=title)

plt.axhline(h_d, linestyle="--", label=r"$h_d$")
plt.xlabel(r"$t$")
plt.ylabel(r"$h(t)$")
plt.title("Variationally induced PD feedback and manually chosen PD laws")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "fig-02-07-variational-induced-vs-pd.png", dpi=300)
plt.savefig(output_dir / "fig-02-07-variational-induced-vs-pd.pdf")
plt.close()

fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)

for ax, (_, title, k1, k2, e, w, h_values) in zip(axes, results):
    ax.plot(t, h_values)
    ax.axhline(h_d, linestyle="--", color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(r"$t$")

axes[0].set_ylabel(r"$h(t)$")
fig.suptitle("Comparison of variationally induced and manually chosen PD feedback laws", y=1.05)
fig.tight_layout()
fig.savefig(output_dir / "fig-02-07-side-by-side.png", dpi=300, bbox_inches="tight")
fig.savefig(output_dir / "fig-02-07-side-by-side.pdf", bbox_inches="tight")
plt.close(fig)

print("Variationally induced gains:")
print("k1 =", k1_var)
print("k2 =", k2_var)
print("gamma =", k2_var + c)
