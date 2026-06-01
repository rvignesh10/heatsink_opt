import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bezier_curve(p0, p1, dx, n_points=100):
    """
    Generates a cubic Bezier curve between p0 and p1.
    Control handles are placed along the X-axis to simulate 
    smooth entry/exit tangent to the flow.
    """
    t = np.linspace(0, 1, n_points)
    
    # Heuristic: Place control handles 50% of the way along dx
    # This creates a smooth S-curve
    p_ctrl_1 = p0 + np.array([dx * 0.5, 0, 0])
    p_ctrl_2 = p1 - np.array([dx * 0.5, 0, 0])
    
    # Bezier Formula: (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)t^2*P2 + t^3*P3
    curve = (np.outer((1-t)**3, p0) + 
             np.outer(3*(1-t)**2 * t, p_ctrl_1) + 
             np.outer(3*(1-t) * t**2, p_ctrl_2) + 
             np.outer(t**3, p1))
    
    return curve[:, 0], curve[:, 1], curve[:, 2]

def plot_cone_constraint():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Parameters
    dx = 2.0  # Distance between planes
    angle_deg = 45
    radius = dx * np.tan(np.radians(angle_deg))

    # Origin Point (c_i)
    ci = np.array([0, 0, 0])

    # --- Draw the Planes ---
    # Plane 1 at x=0
    y = np.linspace(-3, 3, 10)
    z = np.linspace(-3, 3, 10)
    Y, Z = np.meshgrid(y, z)
    X0 = np.zeros_like(Y)
    ax.plot_surface(X0, Y, Z, alpha=0.1, color='blue')
    # Removed (x=0)
    ax.text(0, 0, 3.2, "Plane j", ha='center', fontsize=16)

    # Plane 2 at x=dx
    X1 = np.full_like(Y, dx)
    ax.plot_surface(X1, Y, Z, alpha=0.1, color='blue')
    # Removed (x=2)
    ax.text(dx, 0, 3.2, "Plane j+1", ha='center', fontsize=16)

    # --- Draw the Cone ---
    u = np.linspace(0, dx, 50)
    v = np.linspace(0, 2*np.pi, 50)
    U, V = np.meshgrid(u, v)
    
    X_cone = U
    Y_cone = U * np.tan(np.radians(angle_deg)) * np.cos(V)
    Z_cone = U * np.tan(np.radians(angle_deg)) * np.sin(V)
    
    ax.plot_surface(X_cone, Y_cone + ci[1], Z_cone + ci[2], 
                    alpha=0.15, color='orange', edgecolor='none')
    
    # Draw cone edges (generators)
    ax.plot([0, dx], [0, radius], [0, 0], 'k:', lw=1, alpha=0.4)
    ax.plot([0, dx], [0, -radius], [0, 0], 'k:', lw=1, alpha=0.4)
    ax.plot([0, dx], [0, 0], [0, radius], 'k:', lw=1, alpha=0.4)
    ax.plot([0, dx], [0, 0], [0, -radius], 'k:', lw=1, alpha=0.4)

    # --- Draw the Valid Region Circle on Plane j ---
    theta_circ = np.linspace(0, 2*np.pi, 100)
    y_circ = radius * np.cos(theta_circ) + ci[1]
    z_circ = radius * np.sin(theta_circ) + ci[2]
    x_circ = np.full_like(y_circ, dx)
    
    ax.plot(x_circ, y_circ, z_circ, color='orange', linewidth=3, label='Constraint Boundary')

    # --- Plot Points & Curves ---
    
    # 1. Origin
    ax.scatter(ci[0], ci[1], ci[2], color='black', s=100, label='$c_i$ (Origin)', depthshade=False)

    # 2. Valid Case
    cj_valid = np.array([dx, 0.5, 0.5])
    
    # Interpolated Curve (Valid)
    vx, vy, vz = bezier_curve(ci, cj_valid, dx)
    ax.plot(vx, vy, vz, 'g-', lw=3, label='Valid Path')
    
    # Straight Line (Control Polygon)
    ax.plot([ci[0], cj_valid[0]], [ci[1], cj_valid[1]], [ci[2], cj_valid[2]], 'g--', lw=1, alpha=0.5)
    
    # Endpoint
    ax.scatter(cj_valid[0], cj_valid[1], cj_valid[2], color='green', s=120, label='$c_j$ (Valid)', depthshade=False)


    # 3. Invalid Case
    cj_invalid = np.array([dx, 1.8, 1.2]) 
    
    # Interpolated Curve (Invalid)
    ivx, ivy, ivz = bezier_curve(ci, cj_invalid, dx)
    ax.plot(ivx, ivy, ivz, 'r-', lw=3, label='Invalid Path')
    
    # Straight Line (Control Polygon)
    ax.plot([ci[0], cj_invalid[0]], [ci[1], cj_invalid[1]], [ci[2], cj_invalid[2]], 'r--', lw=1, alpha=0.5)
    
    # Endpoint
    ax.scatter(cj_invalid[0], cj_invalid[1], cj_invalid[2], color='red', s=120, label='$c_j$ (Invalid)', depthshade=False)

    # Annotations
    ax.set_xlabel('X [m]', fontsize=16)
    ax.set_ylabel('Y [m]', fontsize=16)
    ax.set_zlabel('Z [m]', fontsize=16)
    # ax.set_title(f'Cone Projection Constraint')
    
    ax.set_xlim(-0.5, dx + 0.5)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    
    ax.view_init(elev=20, azim=-60)
    
    # plt.legend(loc='upper left', fontsize=16)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig('cone_constraint_interpolated.pdf', dpi=300, transparent=True)
    plt.show()

if __name__ == "__main__":
    plot_cone_constraint()