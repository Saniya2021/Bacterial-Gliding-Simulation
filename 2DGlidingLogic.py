import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv

# ---------------- Parameters ----------------
xBegin = 0
xEnd = 6000  # nm
totalPoints = 1000
amplitude = 250
vShift = 250
trackLen = 6101.53  # arc length in nm
# speed = 100  # nm/sec (used for track progression)
speedMicron = 3000  # nm/sec 
dt = 1  # sec per time step
adhesion_strengths = [0.05, 0.2] #track1, track2
num_cycles = 2


N_per_track = 16

# ---------------- Sine wave tracks ----------------
xVal = np.linspace(xBegin, xEnd, totalPoints)
yVal1 = amplitude * np.sin((2 * np.pi / xEnd) * xVal) + vShift        
yVal2 = -amplitude * np.sin((2 * np.pi / xEnd) * xVal) + vShift        

# Arc lengths
dx = np.diff(xVal)
dy1 = np.diff(yVal1)
dy2 = np.diff(yVal2)

arcLen1 = np.concatenate(([0], np.cumsum(np.sqrt(dx**2 + dy1**2))))
arcLen2 = np.concatenate(([0], np.cumsum(np.sqrt(dx**2 + dy2**2))))

# Interpolators for movement (arc-length -> coords)
xTrackVal1 = interp1d(arcLen1, xVal, fill_value="extrapolate")
yTrackVal1 = interp1d(arcLen1, yVal1, fill_value="extrapolate")
xTrackVal2 = interp1d(arcLen2, xVal, fill_value="extrapolate")
yTrackVal2 = interp1d(arcLen2, yVal2, fill_value="extrapolate")

# ---------------- One unified simulation loop ----------------
# Record positions for plotting (aggregate across all SprBs)
sprb1_xs, sprb1_ys, sprb1_stuck = [], [], []
sprb2_xs, sprb2_ys, sprb2_stuck = [], [], []

trajectory = [0.0]
time_series = [0.0]
bound_counts = [] 

# Initialize N SprBs per track, evenly spaced by arc-length
# Track 1 goes 0 -> trackLen; Track 2 goes trackLen -> 0
sprb1_distances = np.linspace(0, trackLen * (1 - 1 / N_per_track), N_per_track)
sprb2_distances = np.linspace(trackLen, trackLen / N_per_track, N_per_track)

cycle = 0
t = 0.0
sprbCount = 0

while cycle < num_cycles:
    while sprbCount < N_per_track:
        s1x = xTrackVal1(sprb1_distances)
        s1y = yTrackVal1(sprb1_distances)
        s2x = xTrackVal2(sprb2_distances)
        s2y = yTrackVal2(sprb2_distances)

    # Record for track plot
        s1_bound_mask = (s1y <= 100)
        s2_bound_mask = (s2y <= 100)

        sprb1_xs.extend(s1x.tolist()); sprb1_ys.extend(s1y.tolist());
        sprb1_stuck.extend(s1_bound_mask.tolist())
        sprb2_xs.extend(s2x.tolist()); sprb2_ys.extend(s2y.tolist());
        sprb2_stuck.extend(s2_bound_mask.tolist())

        bound1 = np.count_nonzero(s1_bound_mask)
        bound2 = np.count_nonzero(s2_bound_mask)

        total_bound = bound1 + bound2
        bound_counts.append((t, total_bound))

        v_net = 0.0
        if bound1 > 0 or bound2 > 0:
            # Sum contribution = (speed * strength * count_on_track1) + (−speed * strength * count_on_track2)
            v_A = +speedMicron * adhesion_strengths[0] * bound1
            v_B = -speedMicron * adhesion_strengths[1] * bound2
            v_net = v_A + v_B  # (nm/s)

    # Keep your original sign convention
        cell_disp = v_net * dt 

    # Update cell displacement and time
        trajectory.append(trajectory[-1] + cell_disp)
        t += dt
        time_series.append(t)

    # Advance all SprBs along their tracks
        sprb1_distances = sprb1_distances + speedMicron * dt    # L -> R
        sprb2_distances = sprb2_distances - speedMicron * dt    # R -> L

    # Cycle/reset when the "lead" SprB on each track completes (index 0 starts at 0 / trackLen)
        if sprb1_distances[0] >= trackLen and sprb2_distances[0] <= 0:
            sprb1_distances = np.linspace(0, trackLen * (1 - 1 / N_per_track), N_per_track)
            sprb2_distances = np.linspace(trackLen, trackLen / N_per_track, N_per_track)
            sprbCount += 2
            cycle += 1
 

# ---------------- Plot: SprB Count ----------------
with open("bound_sprbs.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time (s)", "Bound SprBs"])   # header
    for t, count in bound_counts:
        writer.writerow([f"{t:.0f}", count])

print("SprB count per second saved in bound_sprbs.csv")


# ---------------- Plot: SprB tracks ----------------
plt.figure(figsize=(10, 4))
plt.plot(xVal, yVal1, label='Track 1 (SprB1)', alpha=0.5)
plt.plot(xVal, yVal2, label='Track 2 (SprB2)', alpha=0.5)

sprb1_xs = np.array(sprb1_xs); sprb1_ys = np.array(sprb1_ys); sprb1_stuck = np.array(sprb1_stuck)
sprb2_xs = np.array(sprb2_xs); sprb2_ys = np.array(sprb2_ys); sprb2_stuck = np.array(sprb2_stuck)

plt.scatter(sprb1_xs[~sprb1_stuck], sprb1_ys[~sprb1_stuck], s=8, label='Track1 SprBs (unstuck)')
plt.scatter(sprb2_xs[~sprb2_stuck], sprb2_ys[~sprb2_stuck], s=8, label='Track2 SprBs (unstuck)')


plt.title("SprBs on Mirrored Sine Tracks (multiple per track)")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.ylim(0, 500)
plt.xlim(0, 6000)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ---------------- Plot: Cell displacement over time ----------------
plt.figure(figsize=(10, 4))
plt.plot(time_series, trajectory, marker='o')
plt.title("Cell Displacement Over Time (Tug-of-War with multiple SprBs)")
plt.xlabel("Time (s)")
plt.ylabel("Cell Displacement (nm)")
plt.grid(True)
plt.tight_layout()
plt.show()


# ---------------- Plot: Cell displacement over XY ----------------

x_xy = np.array(trajectory)                
x_xy_wrapped = np.mod(x_xy, xEnd)
y_xy = np.full_like(x_xy_wrapped, vShift)   # constant centerline

plt.figure(figsize=(10, 4))
# draw the tracks faintly for context

# draw the cell path in XY
plt.plot(x_xy_wrapped, y_xy, linewidth=2, label="Cell trajectory (XY)")


plt.title("Cell Trajectory in XY")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.ylim(0, 500)
plt.xlim(0, 6000)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

for i in range(len(trajectory)):
    print(i, trajectory[i])
