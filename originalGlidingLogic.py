import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import weibull_min
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D

# -------------------- Parameters --------------------
L = 6.0
v_cell = 2.0
dt = 0.1
T_total = 300
steps = int(T_total / dt)

N_tracks = 3
N_per_track = 15
N_total = N_tracks * N_per_track
sprb_speed = 3.0
track_length = L
rear_zone = 0.5
front_zone = 0.5
bind_prob = 0.007

shape_param = 1.8
scale_param = 12.0
adhesion_thresh = 10
jam_time_thresh = 4.5

cell_radius = 0.5
n_turns = 3
helical_pitch = track_length / n_turns
theta_per_um = 2 * np.pi / helical_pitch

def sample_turn_angle():
    while True:
        if np.random.rand() < 0.7:
            angle = np.random.normal(loc=25, scale=30)  # regular turn
        else:
            angle = np.random.normal(loc=175, scale=30)  # sharp turn
        
        if 0 <= angle <= 180 :
            return angle


# -------------------- SprB Class --------------------
class SprB:
    def __init__(self, track_id):
        # Assign the SprB to one of the three helical tracks on the cell surface
        self.track_id = track_id

        # Random initial position along the x-axis (cell body axis), ranging from -L/2 to +L/2
        self.x = np.random.uniform(-track_length / 2, track_length / 2)

        # Random initial phase (theta) plus an offset based on the track ID
        # Ensures SprBs in different tracks are offset in angular position
        self.theta = (2 * np.pi * np.random.rand()) + (2 * np.pi * track_id) / N_tracks

        # Initial binding state: 50% chance of being bound at t=0
        self.bound = random.random() < 0.5

        # Timer used for modeling unbinding probability (Weibull-based)
        self.timer = 0.0

        # If initially bound, assign a random bound duration between 0–2 seconds
        self.bound_duration = 0.0 if not self.bound else random.uniform(0, 2.0)

        # Compute initial helical coordinates (y, z) from theta
        self.update_helical_coords()

    def update_helical_coords(self):
        # Compute the radial position on the helix from the current theta
        self.y = cell_radius * np.cos(self.theta)
        self.z = cell_radius * np.sin(self.theta)

    def update(self):
        # Only unbound SprBs can move along the helical track
        if not self.bound:
            dx = sprb_speed * dt              # Linear axial motion per timestep
            dtheta = theta_per_um * dx        # Corresponding angular change (helical pitch)

            self.x += dx                      # Move forward along x (cell axis)
            self.theta += dtheta              # Rotate around the cell (helical motion)

            # Wrap around if SprB reaches rear of the cell (reappears at front)
            # This models a continuous helical track (like a looped conveyor belt)
            if self.x > track_length / 2:
                self.x -= track_length

            # Recompute position on helix after moving
            self.update_helical_coords()

            # Try binding to the surface if SprB is near the front of the track
            if self.try_bind():
                self.bound = True
                self.bound_duration = 0.0  # Reset timer on binding

        else:
            # If SprB is bound, increase its bound duration counter
            self.bound_duration += dt

            # Try unbinding based on a Weibull-distributed probability
            if self.try_unbind():
                self.bound = False  # Once unbound, it can move again

    def try_bind(self):
        # SprBs can only bind when near the front zone of the cell
        # Binding is probabilistic with a small probability per timestep
        return (self.x + track_length / 2 <= front_zone) and (random.random() < bind_prob)

    def try_unbind(self):
        # Unbinding probability increases over time, following a Weibull CDF
        self.timer += dt
        unbind_prob = weibull_min.cdf(self.timer, shape_param, scale=scale_param)

        # If random number falls below the cumulative unbinding probability, unbind
        if random.random() < unbind_prob:
            self.timer = 0.0  # Reset unbinding timer
            return True
        return False

    def get_helical_position(self):
        # Return full 3D coordinates of SprB on the helical track
        return self.x, self.y, self.z

    @property
    def linear_pos(self):
        # Returns the linearized track position in range [0, L]
        # Used to determine if SprB is near the rear end (for jamming logic)
        return self.x + track_length / 2


# -------------------- SprB Initialization --------------------
sprbs = []
for track_id in range(N_tracks):
    for _ in range(N_per_track):
        s = SprB(track_id)
        sprbs.append(s)

# -------------------- Simulation Setup --------------------
x, y = [0], [0]
positions = [(0, 0)]
heading = 0
turn_log = []
n_bound_prev = 0

# -------------------- Main Simulation Loop --------------------
for step in range(steps):
    rear_stuck_durations = []
    n_bound = 0

    for s in sprbs:
        s.update()
        if s.bound:
            n_bound += 1
            if s.linear_pos >= (track_length - rear_zone):
                rear_stuck_durations.append(s.bound_duration)

    rear_jammed = any(t > jam_time_thresh for t in rear_stuck_durations)

    rear_dominated = False
    if n_bound > 5:
        rear_bound = sum(1 for s in sprbs if s.bound and s.linear_pos > (track_length / 2))
        if rear_bound / n_bound > 0.8:
            rear_dominated = True

    coordination_failure = False
    if step > 0 and (n_bound_prev - n_bound) >= 5:
        coordination_failure = True

    if rear_jammed or rear_dominated or coordination_failure:
        angle = sample_turn_angle()
        heading += np.deg2rad(angle)
        turn_type = "sharp"
        turn_log.append((x[-1], y[-1], angle, step * dt, turn_type))

    elif n_bound < adhesion_thresh and random.random() < 0.015:
        angle = sample_turn_angle()
        heading += np.deg2rad(angle) * (1 if random.random() < 0.5 else -1)
        turn_type = "sharp" if angle >= 150 else "normal"
        turn_log.append((x[-1], y[-1], angle, step * dt, turn_type))

    dx = v_cell * dt * np.cos(heading)
    dy = v_cell * dt * np.sin(heading)
    x.append(x[-1] + dx)
    y.append(y[-1] + dy)
    positions.append((x[-1], y[-1]))
    n_bound_prev = n_bound

# -------------------- Plotting --------------------
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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))
times = np.arange(len(positions)) * dt
norm = plt.Normalize(times.min(), times.max())
cmap = plt.get_cmap('viridis')
colors = cmap(norm(times))

for i in range(1, len(x)):
    ax1.plot(x[i-1:i+1], y[i-1:i+1], color=colors[i])
legend_added = {"normal": False, "sharp": False}
for tx, ty, tt in zip(turn_x, turn_y, turn_type):
    color = "blue" if tt == "normal" else "red"
    label = tt if not legend_added[tt] else None
    ax1.scatter(tx, ty, color=color, s=25, label=label)
    legend_added[tt] = True

ax1.set_title("Trajectory with Adhesin-Driven Turns (Helical Simulation)")
ax1.set_xlabel("X (μm)")
ax1.set_ylabel("Y (μm)")
ax1.legend()
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax1).set_label("Time (s)")

ax2.plot(turn_dists, turn_angle, '-o', color='black')
ax2.set_title("Turn Angle vs Distance")
ax2.set_xlabel("Distance (μm)")
ax2.set_ylabel("Angle (°)")
ax2.set_ylim([0, 180])
ax2.grid(True)
plt.tight_layout()
plt.show()

# # -------------------- Clean Helical SprB Track Visualization --------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
colors = ['blue', 'green', 'red']

for s in sprbs:
    if s.bound:
        xh, yh, zh = s.get_helical_position()
        ax.scatter(xh, yh, zh, color=colors[s.track_id], s=25, alpha=0.8)

theta_vals = np.linspace(0, 2 * np.pi * n_turns, 200)
x_vals = np.linspace(-L/2, L/2, 200)

for i in range(N_tracks):
    offset = (2 * np.pi * i) / N_tracks
    theta = theta_vals + offset
    y_helix = cell_radius * np.cos(theta)
    z_helix = cell_radius * np.sin(theta)
    x_helix = x_vals
    ax.plot(x_helix, y_helix, z_helix, color=colors[i], linewidth=3, alpha=0.5, label=f"Track {i+1}")

ax.set_xlim([-track_length / 2, track_length / 2])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel("X (cell axis, μm)")
ax.set_ylabel("Y (μm)")
ax.set_zlabel("Z (μm)")
ax.set_title("SprB Helical Positions (Bound Only)")
ax.legend()
plt.tight_layout()
plt.show()

# -------------------- Summary Print --------------------
print("\n--- SprB System Parameters ---")
print(f"Total SprB molecules: {N_total} ({N_per_track} per track × {N_tracks} tracks)")
print(f"SprB track length: {track_length} μm")
print(f"SprB translocation speed: {sprb_speed} μm/s")
print(f"Rear unbinding zone: last {rear_zone} μm of the track")
print(f"Front binding zone: first {front_zone} μm of the track")

print("\n--- Adhesion Kinetics ---")
print(f"Weibull shape parameter (m): {shape_param}")
print(f"Weibull scale parameter (τ): {scale_param} s")
print(f"Adhesion threshold for soft turns: {adhesion_thresh} bound SprBs")
print(f"Rear SprB jam time threshold (flip trigger): {jam_time_thresh} s")

print("\n--- Simulation Parameters ---")
print(f"Cell speed: {v_cell} μm/s")
print(f"Total simulation time: {T_total} s")
print(f"Turns per body length traveled: {turns_per_bl:.2f}")
