import numpy as np
from scipy.optimize import root
from plot_style import FIGURE_DIR, setup_matplotlib

plt = setup_matplotlib()

T = 5.0
dt = 0.01
t = np.arange(0.0, T + dt, dt)

h_d = 10.0
e0 = 3.0
w0 = 0.0

q = 1.0
p = 1.0
r = 0.2
c = 0.3

def rk4_general(f, x0, t):
    x = np.zeros((len(t), len(x0)))
    x[0] = x0

    for i in range(len(t) - 1):
        step = t[i + 1] - t[i]
        k_1 = f(x[i])
        k_2 = f(x[i] + 0.5 * step * k_1)
        k_3 = f(x[i] + 0.5 * step * k_2)
        k_4 = f(x[i] + step * k_3)
        x[i + 1] = x[i] + step * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

    return x

def optimal_system(x):
    e, w, e2, e3 = x
    return np.array([
        w,
        e2,
        e3,
        ((p + r * c**2) / r) * e2 - (q / r) * e
    ])

def optimal_residual(z):
    e2_0, e3_0 = z
    x0 = np.array([e0, w0, e2_0, e3_0])
    x = rk4_general(optimal_system, x0, t)

    e_T = x[-1, 0]
    w_T = x[-1, 1]

    return np.array([e_T, w_T])

solution = root(optimal_residual, np.array([0.0, 0.0]))

e2_0, e3_0 = solution.x
x0_optimal = np.array([e0, w0, e2_0, e3_0])
x_optimal = rk4_general(optimal_system, x0_optimal, t)

e_optimal = x_optimal[:, 0]
h_optimal = h_d + e_optimal

def pd_system(k1, k2):
    def f(x):
        e, w = x
        return np.array([
            w,
            -(k2 + c) * w - k1 * e
        ])
    return f

pd_params = [
    (4.0, 1.0, "PD: k1=4, k2=1"),
    (4.0, 4.0, "PD: k1=4, k2=4"),
    (4.0, 8.0, "PD: k1=4, k2=8"),
]

plt.figure(figsize=(9, 5.5))

plt.plot(
    t,
    h_optimal,
    linewidth=3,
    label="Variational trajectory"
)

for k1, k2, label in pd_params:
    x_pd = rk4_general(pd_system(k1, k2), np.array([e0, w0]), t)
    h_pd = h_d + x_pd[:, 0]
    plt.plot(t, h_pd, label=label)

plt.axhline(h_d, linestyle="--", label=r"$h_d$")
plt.xlabel("t")
plt.ylabel(r"$h(t)$")
plt.title("Comparison of height trajectories")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fig-02-07-variational-shooting.pdf")
plt.close()

print("Optimal shooting values:")
print("e''(0) =", e2_0)
print("e'''(0) =", e3_0)
print("Terminal error e(T), e'(T) =", x_optimal[-1, 0], x_optimal[-1, 1])
