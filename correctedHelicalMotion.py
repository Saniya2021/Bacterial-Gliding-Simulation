import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------- Step 1: Re-initialize 1 random SprB from each track ---------
selected_sprbs = []
for track_id in range(N_tracks):
    candidates = [s for s in sprbs if s.track_id == track_id]
    chosen = random.choice(candidates)
    chosen.__init__(track_id=track_id)  # reset to t=0
    selected_sprbs.append(chosen)

# --------- Step 2: Track their real-time 3D positions ---------
sprb_trajs = {i: {'x': [], 'y': [], 'z': []} for i in range(N_tracks)}

for step in range(steps):
    for i, s in enumerate(selected_sprbs):
        s.update()
        xh, yh, zh = s.get_helical_position()
        sprb_trajs[i]['x'].append(xh)
        sprb_trajs[i]['y'].append(yh)
        sprb_trajs[i]['z'].append(zh)

# --------- Step 3: Plot tracks + trajectories from multiple views ---------
view_angles = [
    (30, 45),    # Isometric
    (90, 0),     # Top-down
    (0, 0),      # Side view
    (45, 135),   # Diagonal
    (60, 270),   # Rear-diagonal
]

colors = ['blue', 'green', 'red']
theta_vals = np.linspace(0, 2 * np.pi * n_turns, 500)

for elev, azim in view_angles:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3 helical tracks
    for i in range(N_tracks):
        offset = (2 * np.pi * i) / N_tracks
        theta = theta_vals + offset
        x_helix = (theta_vals / (2 * np.pi * n_turns)) * track_length - (track_length / 2)
        y_helix = cell_radius * np.cos(theta)
        z_helix = cell_radius * np.sin(theta)
        ax.plot(x_helix, y_helix, z_helix, color=colors[i], linewidth=2.5, alpha=0.4, label=f"Track {i+1}")

    # Plot each selected SprB's real trajectory
    for i in range(N_tracks):
        ax.plot(
            sprb_trajs[i]['x'],
            sprb_trajs[i]['y'],
            sprb_trajs[i]['z'],
            color=colors[i],
            linewidth=2.0,
            label=f"SprB {i+1}"
        )

    # Set view and axes
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim([-track_length / 2, track_length / 2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("X (cell axis, μm)")
    ax.set_ylabel("Y (μm)")
    ax.set_zlabel("Z (μm)")
    ax.set_title(f"SprB Helical Motion (elev={elev}°, azim={azim}°)")
    ax.legend()
    plt.tight_layout()
    plt.show()
