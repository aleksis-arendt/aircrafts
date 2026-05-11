import numpy as np
import csv
from plot_style import FIGURE_DIR, setup_matplotlib

plt = setup_matplotlib()

output_dir = FIGURE_DIR
output_dir.mkdir(exist_ok=True)

def find_window_escape_time(t, h, h_esc, window_length):
    dt = t[1] - t[0]
    window_steps = int(window_length / dt)

    for i in range(len(t) - window_steps):
        if np.all(h[i:i + window_steps + 1] >= h_esc):
            return t[i]

    return np.inf

def simulate_rocket(mu, R, alpha, m_dry, m_prop0, ve, rho, burn_fraction, h0, v0, tmax, dt, window_length):
    t = np.arange(0.0, tmax + dt, dt)
    y = np.zeros((len(t), 3))

    m0 = m_dry + m_prop0
    m_used_target = burn_fraction * m_prop0
    m_final = m0 - m_used_target
    t_burn = m_used_target / rho if rho > 0 else 0.0
    h_esc = alpha * R

    y[0] = np.array([h0, v0, m0])

    def burning(time, mass):
        return time < t_burn and mass > m_final

    def f(time, Y):
        h, v, m = Y

        if h <= 0.0 and time > 0.0 and v <= 0.0:
            return np.array([0.0, 0.0, 0.0])

        radius = R + max(h, 0.0)
        gravity = mu / radius**2

        if burning(time, m):
            dm = -rho
            thrust_accel = ve * rho / m
        else:
            dm = 0.0
            thrust_accel = 0.0

        return np.array([v, thrust_accel - gravity, dm])

    impact = False

    for i in range(len(t) - 1):
        h_step = t[i + 1] - t[i]

        k_1 = f(t[i], y[i])
        k_2 = f(t[i] + 0.5 * h_step, y[i] + 0.5 * h_step * k_1)
        k_3 = f(t[i] + 0.5 * h_step, y[i] + 0.5 * h_step * k_2)
        k_4 = f(t[i] + h_step, y[i] + h_step * k_3)

        y[i + 1] = y[i] + h_step * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6.0

        if y[i + 1, 2] < m_final:
            y[i + 1, 2] = m_final

        if y[i + 1, 0] < 0.0 and t[i + 1] > 1.0:
            y[i + 1, 0] = 0.0
            y[i + 1, 1] = 0.0
            y[i + 1:, :] = y[i + 1]
            impact = True
            break

    h = y[:, 0]
    v = y[:, 1]
    m = y[:, 2]

    radius = R + np.maximum(h, 0.0)
    energy = 0.5 * v**2 - mu / radius

    tau_threshold = np.inf
    v_threshold = np.nan

    for i in range(len(t)):
        if h[i] >= h_esc:
            tau_threshold = t[i]
            v_threshold = v[i]
            break

    tau_window = find_window_escape_time(t, h, h_esc, window_length)

    tau_energy = np.inf
    for i in range(len(t)):
        if t[i] >= t_burn and energy[i] >= 0.0 and v[i] > 0.0:
            tau_energy = t[i]
            break

    crossed = np.any(h >= h_esc)

    if np.isfinite(tau_energy):
        status = "energy escape"
    elif np.isfinite(tau_window):
        status = "sustained threshold crossing"
    elif crossed:
        status = "temporary crossing"
    elif impact:
        status = "impact"
    else:
        status = "failed escape"

    return {
        "t": t,
        "h": h,
        "v": v,
        "m": m,
        "energy": energy,
        "h_esc": h_esc,
        "burn_fraction": burn_fraction,
        "m_used": m_used_target,
        "t_burn": t_burn,
        "tau_threshold": tau_threshold,
        "tau_window": tau_window,
        "tau_energy": tau_energy,
        "v_threshold": v_threshold,
        "status": status,
        "impact": impact,
        "window_length": window_length,
        "m_final": m_final,
    }

def succeeds(result):
    return result["status"] in ["energy escape", "sustained threshold crossing"]

def find_min_burn_fraction(mu, R, alpha, m_dry, m_prop0, ve, rho, h0, v0, tmax, dt, window_length, tol=1e-4):
    low = 0.0
    high = 1.0

    high_result = simulate_rocket(
        mu, R, alpha, m_dry, m_prop0, ve, rho, high, h0, v0, tmax, dt, window_length
    )

    if not succeeds(high_result):
        return None, high_result

    best_result = high_result

    for _ in range(60):
        mid = 0.5 * (low + high)

        result = simulate_rocket(
            mu, R, alpha, m_dry, m_prop0, ve, rho, mid, h0, v0, tmax, dt, window_length
        )

        if succeeds(result):
            high = mid
            best_result = result
        else:
            low = mid

        if high - low < tol:
            break

    return high, best_result

def plot_result(result, name):
    t = result["t"]
    h = result["h"]
    v = result["v"]
    m = result["m"]
    energy = result["energy"]
    h_esc = result["h_esc"]

    plt.figure(figsize=(7, 4))
    plt.plot(t, h / 1000.0, label=r"$h(t)$")
    plt.axhline(h_esc / 1000.0, linestyle="--", label=r"$h_{\mathrm{esc}}=\alpha R$")
    plt.xlabel(r"$t$ (s)")
    plt.ylabel(r"$h(t)$ (km)")
    plt.title(name.replace("_", " "))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"fig-05-06-{name.replace('_', '-')}-height.png", dpi=300)
    plt.savefig(output_dir / f"fig-05-06-{name.replace('_', '-')}-height.pdf")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(t, v / 1000.0, label=r"$v(t)$")
    plt.axhline(0.0, linestyle="--", label=r"$v=0$")
    plt.xlabel(r"$t$ (s)")
    plt.ylabel(r"$v(t)$ (km/s)")
    plt.title(name.replace("_", " "))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"fig-05-06-{name.replace('_', '-')}-velocity.png", dpi=300)
    plt.savefig(output_dir / f"fig-05-06-{name.replace('_', '-')}-velocity.pdf")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(t, m, label=r"$m(t)$")
    plt.xlabel(r"$t$ (s)")
    plt.ylabel(r"$m(t)$ (kg)")
    plt.title(name.replace("_", " "))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"fig-05-06-{name.replace('_', '-')}-mass.png", dpi=300)
    plt.savefig(output_dir / f"fig-05-06-{name.replace('_', '-')}-mass.pdf")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(t, energy / 1e6, label=r"$\mathcal{E}(t)$")
    plt.axhline(0.0, linestyle="--", label=r"$\mathcal{E}=0$")
    plt.xlabel(r"$t$ (s)")
    plt.ylabel(r"$\mathcal{E}(t)$ $(10^6\ \mathrm{J/kg})$")
    plt.title(name.replace("_", " "))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"fig-05-06-{name.replace('_', '-')}-energy.png", dpi=300)
    plt.savefig(output_dir / f"fig-05-06-{name.replace('_', '-')}-energy.pdf")
    plt.close()

def save_summary_csv(rows, filename):
    with open(output_dir / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

mu = 3.986e14
R = 6.371e6

alpha = 0.1
window_length = 1000.0

m_dry = 1000.0
m_prop0 = 4000.0
ve = 4500.0
rho = 20.0

h0 = 0.0
v0 = 0.0

tmax = 20000.0
dt = 1.0

burn_fraction, result = find_min_burn_fraction(
    mu=mu,
    R=R,
    alpha=alpha,
    m_dry=m_dry,
    m_prop0=m_prop0,
    ve=ve,
    rho=rho,
    h0=h0,
    v0=v0,
    tmax=tmax,
    dt=dt,
    window_length=window_length,
)

if burn_fraction is None:
    print("No sustained threshold crossing detected even after burning all propellant.")
else:
    print("Minimum burn fraction =", burn_fraction)
    print("Minimum burn percent =", 100 * burn_fraction)

print("status =", result["status"])
print("h_esc =", result["h_esc"])
print("m_used =", result["m_used"])
print("t_burn =", result["t_burn"])
print("tau_threshold =", result["tau_threshold"])
print("tau_window =", result["tau_window"])
print("window_length =", result["window_length"])
print("tau_energy =", result["tau_energy"])
print("v_threshold =", result["v_threshold"])
print("final mass =", result["m"][-1])
print("max height =", np.max(result["h"]))
print("max velocity =", np.max(result["v"]))
print("max energy =", np.max(result["energy"]))

plot_result(result, "minimum_window_escape")

rows = [
    {
        "status": result["status"],
        "alpha": alpha,
        "h_esc_m": result["h_esc"],
        "window_length_s": result["window_length"],
        "burn_fraction": result["burn_fraction"],
        "burn_percent": 100 * result["burn_fraction"],
        "m_used_kg": result["m_used"],
        "t_burn_s": result["t_burn"],
        "tau_threshold_s": result["tau_threshold"],
        "tau_window_s": result["tau_window"],
        "tau_energy_s": result["tau_energy"],
        "v_threshold_m_s": result["v_threshold"],
        "final_mass_kg": result["m"][-1],
        "max_height_m": np.max(result["h"]),
        "max_velocity_m_s": np.max(result["v"]),
        "max_energy_J_kg": np.max(result["energy"]),
    }
]

save_summary_csv(rows, "escape_summary.csv")
