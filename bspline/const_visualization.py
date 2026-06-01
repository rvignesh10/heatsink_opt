import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Global Configuration
NCP = 8  # Increased number of control points
LONG_MAX = 40  # Maximum longitudinal distance
X_LONG_CP = np.linspace(0, LONG_MAX, NCP)  # Longitudinal locations of CPs
X_DENSE = np.linspace(0, LONG_MAX, 300)    # Dense grid for smooth spline plotting
fs = 16 # font size

def setup_plot(title=None):
    """Helper to setup the rotated coordinate system (Longitudinal on Y-axis)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    # ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel('Longitudinal Distance: X', fontsize=fs)
    ax.set_xlabel('Traverse Distance: Y', fontsize=fs)
    ax.grid(True, linestyle=':', alpha=0.6)
    # Set limits to make it look like a road Channel
    ax.set_xlim(-5, 8) 
    ax.set_ylim(-2, LONG_MAX + 2)
    return fig, ax

def plot_normal_entry_exit():
    """1. Normal Entry & Exit: Visualizes y[1] == y[0] constraint."""
    fig, ax = setup_plot("1. Normal Entry & Exit Constraint")

    # Define Control Point Lateral Coordinates
    # Good: First two and last two are equal (zero derivative w.r.t longitudinal) 
    y_lat_good = np.array([0, 0, 0.5, 2, 2, 0.5, 0, 0])
    
    
    # Bad: First two and last two differ (sharp angle entry/exit)
    y_lat_bad = np.array([-2, 0, 0.5, 2, 2, 0.5, 0, -2])

    # Generate Splines
    # Note: make_interp_spline takes (x, y), where x is the independent variable (Longitudinal)
    spline_good = make_interp_spline(X_LONG_CP, y_lat_good, k=3)(X_DENSE)
    
    spline_bad = make_interp_spline(X_LONG_CP, y_lat_bad, k=3)(X_DENSE)

    # PLOT: swap args to put Lateral on X-axis, Longitudinal on Y-axis
    ax.plot(y_lat_good, X_LONG_CP, 'go', label='_nolegend_', alpha=0.6)
    ax.plot(spline_good, X_DENSE, 'g-', linewidth=2.5, label='Normal Entry')

    ax.plot(y_lat_bad, X_LONG_CP, 'rx', label='_nolegend_', alpha=0.6)
    ax.plot(spline_bad, X_DENSE, 'r--', linewidth=2.5, label='Angled Entry')

    # Annotations
    ax.annotate('Normal Entry\n($d^2_{y, e} = d^1_{y, e}$)', 
                xy=(y_lat_good[0], X_LONG_CP[0]), 
                xytext=(y_lat_good[0]+2.5, X_LONG_CP[0]+5),
                arrowprops=dict(facecolor='green', arrowstyle='->', linewidth=1.5), fontsize=fs-2, color='green')

    ax.annotate('Angled Entry\n($d^2_{y, e}$ != $d^1_{y, e}$)', 
                xy=(y_lat_bad[0], X_LONG_CP[0]), 
                xytext=(y_lat_bad[0]-1, X_LONG_CP[0]+5),
                arrowprops=dict(facecolor='red', arrowstyle='->', linewidth=1.5), fontsize=fs-2, color='red')

    ax.legend(loc='upper right', fontsize=fs)
    ax.set_xlim(-5, 5) 
    plt.tight_layout()
    plt.savefig('constraint_1_normal.pdf', dpi=300, transparent=True)
    print("Saved constraint_1_normal.pdf")


def plot_topology():
    """2. Y-Topology: Visualizes non-crossing Channels."""
    fig, ax = setup_plot()

    # Reference Channel (Center)
    y_lat_ref = np.array([0, 0, 0.5, 1.5, 1.5, 0.5, 0, 0])
    
    # Valid Upper Channel (Strictly > Reference)
    y_lat_valid = np.array([3, 3, 3.5, 6, 4.5, 3.5, 3, 3])
    y_lat_safe  = np.array([7, 7, 7, 7, 7, 7, 7, 7])
    
    # Invalid Upper Channel (Crosses Reference)
    y_lat_invalid = np.array([3, 3, 3.5, -1, 4.5, 3.5, 3, 3])

    # Generate Splines
    spline_ref = make_interp_spline(X_LONG_CP, y_lat_ref, k=3)(X_DENSE)
    spline_valid = make_interp_spline(X_LONG_CP, y_lat_valid, k=3)(X_DENSE)
    spline_safe  = make_interp_spline(X_LONG_CP, y_lat_safe, k=3)(X_DENSE)
    spline_invalid = make_interp_spline(X_LONG_CP, y_lat_invalid, k=3)(X_DENSE)

    # Plot
    # Added markers for control points
    ax.plot(y_lat_ref, X_LONG_CP, 'ko', alpha=0.4, label='_nolegend_')
    ax.plot(spline_ref, X_DENSE, 'k-', linewidth=2.5, label='Channel 1 (Reference)', alpha=0.7)
    
    # Fill safe zone
    # ax.fill_betweenx(X_DENSE, spline_ref, spline_valid, color='green', alpha=0.1, label='Safe Zone')
    ax.fill_betweenx(X_DENSE, spline_ref, spline_safe, color='green', alpha=0.1, label='Safe Zone')
    
    ax.plot(y_lat_valid, X_LONG_CP, 'go', alpha=0.4, label='_nolegend_')
    ax.plot(spline_valid, X_DENSE, 'g-', linewidth=2.5, label='Channel 2 (Valid: $d^k_{y, 2} > d^k_{y, 1}$)')
    
    ax.plot(y_lat_invalid, X_LONG_CP, 'rx', alpha=0.6, label='_nolegend_')
    ax.plot(spline_invalid, X_DENSE, 'r--', linewidth=2.5, label='Channel 2 (Invalid: Crosses)')

    # Annotation
    cross_idx = np.argmin(spline_invalid) # Find roughly where it dips lowest
    ax.annotate('Crossing Violation\n \t$d^k_{y, 2} < d^k_{y, 1}$', 
                xy=(spline_invalid[cross_idx], X_DENSE[cross_idx]), 
                xytext=(-4.5, 4+X_DENSE[cross_idx]), # Fixed position inside left boundary
                arrowprops=dict(facecolor='red', arrowstyle='->', linewidth=1.5), 
                ha='left', va='center', # Align left so text flows into the plot
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9), fontsize=fs-2) # Add background for readability

    ax.legend(loc='upper left', fontsize=fs-2)
    ax.set_xlim(-5, 6.5) 
    plt.tight_layout()
    
    plt.savefig('constraint_2_topology.pdf', dpi=300, transparent=True)
    print("Saved constraint_2_topology.pdf")

def plot_smoothness():
    """3. Y-Smoothness: Visualizes Slew Rate / Step Size limit."""
    fig, ax = setup_plot("3. Y-Smoothness Constraint")

    # Smooth path (Small delta between consecutive CPs)
    y_lat_smooth = np.array([0, 0.2, 0.5, 1.4, 1.8, 2.1, 2.4, 2.5])
    
    # Jerky path (Large delta between CPs)
    y_lat_jerky = np.array([0, 3.0, 3.0, 0, 3.0, 0, 0, 3.0])

    spline_smooth = make_interp_spline(X_LONG_CP, y_lat_smooth, k=3)(X_DENSE)
    spline_jerky = make_interp_spline(X_LONG_CP, y_lat_jerky, k=3)(X_DENSE)

    # Plot
    ax.plot(y_lat_smooth, X_LONG_CP, 'go', label='_nolegend_', alpha=0.6)
    ax.plot(spline_smooth, X_DENSE, 'g-', linewidth=2.5, label='Smooth Curvature (Good)')

    ax.plot(y_lat_jerky, X_LONG_CP, 'rx', label='_nolegend_', alpha=0.6)
    ax.plot(spline_jerky, X_DENSE, 'r--', linewidth=2.5, label='Aggresive Curvature (Bad)')

    # Visualizing the step
    idx = 4
    # Draw line between consecutive CPs to show the "step"
    ax.plot([y_lat_jerky[idx], y_lat_jerky[idx+1]], [X_LONG_CP[idx], X_LONG_CP[idx+1]], 'k-', lw=2.5, alpha=0.5)
    
    ax.annotate('Step too large\n$|d^{j+1}_{y, e} - d^{j}_{y, e}| > \delta_{curv}$', 
                xy=((y_lat_jerky[idx] + y_lat_jerky[idx+1])/2, (X_LONG_CP[idx] + X_LONG_CP[idx+1])/2), 
                xytext=(4, X_LONG_CP[idx]),
                arrowprops=dict(facecolor='red', arrowstyle='->', linewidth=1.5), fontsize=fs-2)

    ax.legend(loc='upper left', fontsize=fs)
    plt.tight_layout()
    plt.savefig('constraint_3_smoothness.pdf', dpi=300, transparent=True)
    print("Saved constraint_3_smoothness.pdf")

def plot_euclidean_distance():
    """4. Inter-Curve Distance: Visualizes MINIMUM distance between CP P_i and Q_i of two curves."""
    fig, ax = setup_plot()

    # Shared longitudinal coordinates (assuming synchronized parameterization)
    # P_i = (X_LONG_CP[i], y_lat_P[i])
    # Q_i = (X_LONG_CP[i], y_lat_Q[i])

    # Curve P (Reference / Ego) - stays straight
    r = 0.3
    y_lat_P = np.zeros_like(X_LONG_CP)
    y_lat_m = y_lat_P - r
    y_lat_p = y_lat_P + r
    
    # Curve Q (Neighbor) - Swoops in dangerously close
    # Starts safe (3.5m), dips to 0.8m (unsafe), returns to 3.5m
    y_lat_Q = np.array([3.5, 3.5, 3.0, 0.8, 3.0, 3.5, 3.5, 3.5])

    # Generate Splines
    spline_P = make_interp_spline(X_LONG_CP, y_lat_P, k=3)(X_DENSE)
    spline_Pm = make_interp_spline(X_LONG_CP, y_lat_m, k=3)(X_DENSE)
    spline_Pp = make_interp_spline(X_LONG_CP, y_lat_p, k=3)(X_DENSE)
    
    spline_Q = make_interp_spline(X_LONG_CP, y_lat_Q, k=3)(X_DENSE)


    # Plot Curves
    ax.plot(spline_P, X_DENSE, 'k-', linewidth=3, label='Curve P (D=0.3 mm)', alpha=0.7)
    ax.plot(spline_Pm, X_DENSE, 'k:', linewidth=3, label='Curve P - D/2', alpha=0.7)
    ax.plot(spline_Pp, X_DENSE, 'k:', linewidth=3, label='Curve P + D/2', alpha=0.7)
    ax.plot(y_lat_P, X_LONG_CP, 'ko', alpha=0.6, label='_nolegend_')

    ax.plot(spline_Q, X_DENSE, 'r-', linewidth=2, label='Curve Q (Intruder)')
    ax.plot(spline_Q - r, X_DENSE, 'r:', linewidth=2, label='_nolegend_')
    ax.plot(spline_Q + r, X_DENSE, 'r:', linewidth=2, label='_nolegend_')
    ax.plot(y_lat_Q, X_LONG_CP, 'ro', alpha=0.6, label='_nolegend_')

    # 1. Visualize Violation (Too Close)
    # Index 4 is where y_lat_Q is 0.8
    idx_viol = 3
    # Coordinates (Note: X axis is Lateral, Y axis is Longitudinal in the plot)
    p_viol = (y_lat_P[idx_viol], X_LONG_CP[idx_viol])
    q_viol = (y_lat_Q[idx_viol], X_LONG_CP[idx_viol])
    
    # Draw connection line
    ax.plot([p_viol[0], q_viol[0]], [p_viol[1], q_viol[1]], 'r-', lw=3)
    
    dist_viol = np.sqrt((p_viol[0]-q_viol[0])**2 + (p_viol[1]-q_viol[1])**2) # Euclidean
    mid_viol = ((p_viol[0]+q_viol[0])/2, (p_viol[1]+q_viol[1])/2)

    ax.annotate(f'Dist: {dist_viol:.1f}mm < 2D\n(Violated)', 
                xy=mid_viol, 
                xytext=(mid_viol[0] + 5, 2 + mid_viol[1]),
                arrowprops=dict(facecolor='red', arrowstyle='->', linewidth=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9), fontsize=fs-2)

    # 2. Visualize Satisfaction (Safe Distance)
    # Index 1 is where y_lat_Q is 3.0
    idx_sat = 2
    p_sat = (y_lat_P[idx_sat], X_LONG_CP[idx_sat])
    q_sat = (y_lat_Q[idx_sat], X_LONG_CP[idx_sat])
    
    # Draw connection line
    ax.plot([p_sat[0], q_sat[0]], [p_sat[1], q_sat[1]], 'g-', lw=3)
    
    dist_sat = np.sqrt((p_sat[0]-q_sat[0])**2 + (p_sat[1]-q_sat[1])**2)
    mid_sat = ((p_sat[0]+q_sat[0])/2, (p_sat[1]+q_sat[1])/2)

    ax.annotate(f'Dist: {dist_sat:.1f}mm > 2D\n(Safe)', 
                xy=mid_sat, 
                xytext=(mid_sat[0] + 3, 2 + mid_sat[1]),
                arrowprops=dict(facecolor='green', arrowstyle='->', linewidth=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.9), fontsize=fs-2)

    ax.legend(loc='upper left', fontsize=fs)
    plt.tight_layout()
    plt.savefig('constraint_4_distance.pdf', dpi=300, transparent=True)
    print("Saved constraint_4_distance.pdf")

if __name__ == "__main__":
    plot_normal_entry_exit()
    plot_topology()
    plot_smoothness()
    plot_euclidean_distance()