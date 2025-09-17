import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import weibull_min
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D

# -------------------- Parameters --------------------
L = 6.0  # Cell length (µm)
v_cell = 2.0  # Cell speed (µm/s)
dt = 0.1
T_total = 300
steps = int(T_total / dt)

# SprB Parameters
N_tracks = 3
N_per_track = 15
N_total = N_tracks * N_per_track
sprb_speed = 3.0  # Translocation speed (µm/s)
track_length = L
rear_zone = 0.5
front_zone = 0.5

# Adhesion Kinetics (Weibull)
shape_param = 1.8
scale_param = 3.5
adhesion_thresh = 14
jam_time_thresh = 4.5

# Soft Turn Angle Distribution
soft_angle_bins = np.arange(15, 150, 15)
soft_angle_probs = np.array([390, 230, 175, 190, 185, 180, 220, 200, 250], dtype=float)
soft_angle_probs /= soft_angle_probs.sum()

def sample_soft_turn_angle():
    return np.random.choice(soft_angle_bins, p=soft_angle_probs)

# -------------------- SprB Class --------------------
class SprB:
    def __init__(self):
        self.position = np.random.uniform(0, track_length)
        self.bound = False
        self.timer = 0.0
        self.bound_duration = 0.0

    def update(self):
        if not self.bound:
            self.position = (self.position + sprb_speed * dt) % track_length
            if self.try_bind():
                self.bound = True
                self.bound_duration = 0.0
        else:
            self.bound_duration += dt
            if self.try_unbind():
                self.bound = False

    def try_bind(self):
        return (self.position >= (track_length - rear_zone)) and (random.random() < 0.02)

    def try_unbind(self):
        self.timer += dt
        unbind_prob = weibull_min.cdf(self.timer, shape_param, scale=scale_param)
        if random.random() < unbind_prob:
            self.timer = 0.0
            return True
        return False

# -------------------- Front Check --------------------
def front_sprb_ready(sprbs, front_zone=0.5):
    for s in sprbs:
        if not s.bound and s.position <= front_zone:
            return True
    return False

# -------------------- Initialization --------------------
sprbs = [SprB() for _ in range(N_total)]
x, y = [0], [0]
heading = 0
positions = [(0, 0)]
turn_log = []

# -------------------- Simulation Loop --------------------
for step in range(steps):
    rear_stuck_durations = []
    n_bound = 0
    for s in sprbs:
        s.update()
        if s.bound:
            n_bound += 1
            if s.position >= (track_length - rear_zone):
                rear_stuck_durations.append(s.bound_duration)

    rear_jammed = any(t > jam_time_thresh for t in rear_stuck_durations)
    no_front_ready = not front_sprb_ready(sprbs)

    if rear_jammed and no_front_ready:
        angle = 180
        heading += np.deg2rad(angle)
        turn_type = "sharp"
        turn_log.append((x[-1], y[-1], angle, step * dt, turn_type))
    elif n_bound < adhesion_thresh and random.random() < 0.015:
        angle = sample_soft_turn_angle()
        heading += np.deg2rad(angle) * (1 if random.random() < 0.5 else -1)
        turn_type = "soft"
        turn_log.append((x[-1], y[-1], angle, step * dt, turn_type))

    dx = v_cell * dt * np.cos(heading)
    dy = v_cell * dt * np.sin(heading)
    x.append(x[-1] + dx)
    y.append(y[-1] + dy)
    positions.append((x[-1], y[-1]))

# -------------------- Turn Stats --------------------
positions = np.array(positions)
if turn_log:
    turn_x, turn_y, turn_angle, turn_time, turn_type = zip(*turn_log)
    dists = np.insert(np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1)), 0, 0)
    turn_indices = [int(t / dt) for t in turn_time]
    turn_dists = [dists[i] for i in turn_indices]
else:
    turn_x, turn_y, turn_angle, turn_time, turn_dists, turn_type = [], [], [], [], [], []

distance_traveled = v_cell * T_total
body_lengths = distance_traveled / L
turns_per_bl = len(turn_log) / body_lengths

print(f"Total turns: {len(turn_log)}")
print(f"Turns per body length traveled: {turns_per_bl:.2f}")

# -------------------- Trajectory Plot --------------------
times = np.arange(len(positions)) * dt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))
norm = plt.Normalize(times.min(), times.max())
cmap = plt.get_cmap('viridis')
colors = cmap(norm(times))

for i in range(1, len(x)):
    ax1.plot(x[i-1:i+1], y[i-1:i+1], color=colors[i])

for i, (tx, ty, tt) in enumerate(zip(turn_x, turn_y, turn_type)):
    color = "magenta" if tt == "soft" else "red"
    ax1.scatter(tx, ty, color=color, s=25, label=tt if i == 0 else "")

ax1.set_title("Trajectory with Adhesin-Driven Turns (WT)")
ax1.set_xlabel("X (µm)")
ax1.set_ylabel("Y (µm)")
ax1.legend()
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax1).set_label("Time (s)")

ax2.plot(turn_dists, turn_angle, '-o', color='black')
ax2.set_title("Turn Angle vs Distance")
ax2.set_xlabel("Distance (µm)")
ax2.set_ylabel("Angle (°)")
ax2.set_ylim([0, 180])
ax2.grid(True)

plt.tight_layout()
plt.show()

# -------------------- Correct 3D Helical Track Plot --------------------
cell_radius = 0.5    # µm
n_turns = 3
colors = ['blue', 'green', 'red']
theta_vals = np.linspace(0, 2 * np.pi * n_turns, 500)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for i in range(N_tracks):
    offset = (2 * np.pi * i) / N_tracks
    theta = theta_vals + offset
    x_helix = (theta_vals / (2 * np.pi * n_turns)) * L - (L / 2)
    y_helix = cell_radius * np.cos(theta)
    z_helix = cell_radius * np.sin(theta)
    ax.plot(x_helix, y_helix, z_helix, color=colors[i], linewidth=2.5, label=f"Track {i+1}")
    ax.scatter(x_helix, y_helix, z_helix, color=colors[i], s=12, alpha=0.7)

ax.set_xlim([-L/2, L/2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])
ax.set_xlabel("X (cell axis, µm)")
ax.set_ylabel("Y (µm)")
ax.set_zlabel("Z (µm)")
ax.set_title("Correct 3D Helical SprB Tracks")
ax.legend()
plt.tight_layout()
plt.show()
