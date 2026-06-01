import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. Setup 2D Geometry (X-Z Plane) ---
u = np.linspace(-1, 2.5 * np.pi, 200)
# Modified curve: Steepened the slope (1.5 * sin) so gravity projection is larger
x = u
z = 0.5 * u + 1.5 * np.sin(u) 

# --- 2. Select Analysis Point and Calculate Vectors ---
# Changed index to 25 (steeper part of the first hill)
idx_target = 50
P = np.array([x[idx_target], z[idx_target]])

# Calculate Tangent (using central difference)
dx = x[idx_target+1] - x[idx_target-1]
dz = z[idx_target+1] - z[idx_target-1]
T_raw = np.array([dx, dz])
# Unit Tangent Vector (t)
T = T_raw / np.linalg.norm(T_raw)

# Define Global Gravity Vector (g)
# Increased magnitude slightly for visibility
g_magnitude = 2.5 
G = np.array([0, -g_magnitude])

# Calculate Effective Gravity (Projection)
g_xi_scalar = np.dot(T, G)
G_eff = g_xi_scalar * T

# --- 3. Create the 2D Plot ---
fig, ax = plt.subplots(figsize=(10, 7))

# Plot the Channel Path surface
ax.plot(x, z, 'b-', linewidth=3, alpha=0.4, label='Micro-Channel Path')
ax.fill_between(x, z-0.2, z+0.2, color='blue', alpha=0.1) 
ax.scatter(P[0], P[1], color='k', s=60, zorder=10, label=r'Evaluation Node, $j$')

# --- Vector Visualization ---
origin = P
base_arrow_params = dict(angles='xy', scale_units='xy', scale=1, zorder=5)

# 1. Unit Tangent Vector (Green)
T_viz_length = 2.5
ax.quiver(*origin, *(T * T_viz_length), color='green', width=0.008, 
          label=r'Unit Tangent ($\mathbf{t}$)', **base_arrow_params)

# Draw backward extension
ax.plot([P[0]-T[0], P[0]+T[0]*3.0], [P[1]-T[1], P[1]+T[1]*3.0], 'g--', alpha=0.3)

# 2. Global Gravity Vector (Black)
ax.quiver(*origin, *G, color='black', width=0.008, 
          label=r'Global Gravity ($\mathbf{g}$)', **base_arrow_params)

# 3. Effective Gravity Component (Red) - NOW VISIBLE
# Thicker width (0.015) and brighter color
ax.quiver(*origin, *G_eff, color='#D00000', width=0.008, 
          label=r'Effective Gravity ($g^j_\xi$)', **base_arrow_params)

# --- Visual Aids for Projection ---
G_tip = origin + G
G_eff_tip = origin + G_eff
projection_line = patches.ConnectionPatch(xyA=G_tip, xyB=G_eff_tip, coordsA='data', coordsB='data', 
                                          axesA=ax, axesB=ax, color='black', linestyle='--', alpha=0.6)
ax.add_patch(projection_line)

# Add a right-angle marker
perp_T = np.array([-T[1], T[0]]) * 0.2
marker_corner = G_eff_tip - T*0.2 * np.sign(g_xi_scalar)
marker_path = [marker_corner, marker_corner + perp_T, G_eff_tip + perp_T]
right_angle = plt.Polygon(marker_path, fill=False, edgecolor='black', linewidth=1)
# ax.add_patch(right_angle)

# --- Aesthetics ---
# ax.set_aspect('equal') 
ax.set_xlabel(r'Horizontal Position ($x$)', fontsize=16)
ax.set_ylabel(r'Vertical Elevation ($z$)', fontsize=16)
# ax.set_title(r'2D Projection of Gravity onto Channel Path: $g_\xi = \langle \mathbf{t}, \mathbf{g} \rangle$')
ax.legend(loc='upper left', frameon=True, fontsize=16)
# ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 5)
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig("gravity_visual.pdf", dpi=300, transparent=True)
plt.show()