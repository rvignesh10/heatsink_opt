import unittest
import numpy as np
import matplotlib.pyplot as plt
from geomdl import BSpline, utilities

# ==============================================================================
# 1. Full Class Implementation (Standard)
# ==============================================================================
class GenBSplineChannels:
    def __init__(self, config: dict, char_props: dict):
        self.char_props = char_props
        self.ncp = config["design_params"]["ncp"] 
        self.ne  = config["design_params"]["ne"]  
        self.nxi = config["design_params"]["nxi"] 
        self.ncp_per_dim = self.ne * self.ncp
        
        self.xstar_cp = np.array(config["spline"]["x_cp"]) / self.char_props.get("L_c", 1.0)
        self.ystar_cp = np.zeros_like(self.xstar_cp)
        self.zstar_cp = np.zeros_like(self.xstar_cp)
        
        self.degree = config["spline"]["degree"]
        self.dist_tol = config["spline"]["dist_tol"] / self.char_props.get("L_c", 1.0)
        self._cached_points = None

    def setInputs(self, alpha_cp):
        self.ystar_cp = alpha_cp[0:self.ncp_per_dim]
        self.zstar_cp = alpha_cp[self.ncp_per_dim:]
        self._cached_points = None

    def _get_basis_and_deriv_vector(self, cp_idx):
        curve = BSpline.Curve()
        curve.degree = self.degree
        curve.ctrlpts = [[0.0]*3 for _ in range(self.ncp)]
        curve.knotvector = utilities.generate_knot_vector(self.degree, self.ncp)
        curve.ctrlpts[cp_idx] = [1.0, 0.0, 0.0]
        
        u_vals = np.linspace(0.0, 1.0, self.nxi)
        N_vals, dN_vals = [], []
        for u in u_vals:
            res = curve.derivatives(u, order=1)
            N_vals.append(res[0][0])
            dN_vals.append(res[1][0])
        return np.array(N_vals), np.array(dN_vals)

    def Interpolate(self):
        x_mat = self.xstar_cp.reshape(self.ne, self.ncp)
        y_mat = self.ystar_cp.reshape(self.ne, self.ncp)
        z_mat = self.zstar_cp.reshape(self.ne, self.ncp)
        
        xstar_mat = np.zeros((self.ne, self.nxi))
        ystar_mat = np.zeros_like(xstar_mat)
        zstar_mat = np.zeros_like(xstar_mat)
        g_mat     = np.zeros_like(xstar_mat)
        xi_mat    = np.zeros_like(xstar_mat)
        
        g_vec = np.array([0.0, 0.0, -9.81])
        g_scale = 1.0 / self.char_props.get("g_c", 1.0)
        u_vals = np.linspace(0.0, 1.0, self.nxi)
        
        for e in range(self.ne):
            curve = BSpline.Curve()
            curve.degree = self.degree
            curve.ctrlpts = np.column_stack((x_mat[e], y_mat[e], z_mat[e])).tolist()
            curve.knotvector = utilities.generate_knot_vector(self.degree, self.ncp)
            
            curr_xi = 0.0
            prev_pt = None
            
            for j, u in enumerate(u_vals):
                res = curve.derivatives(u, order=1)
                pt  = np.array(res[0])
                tan = np.array(res[1])
                
                tan_norm = np.linalg.norm(tan) + 1e-12
                g_mat[e, j] = np.dot(g_vec, tan / tan_norm)
                
                xstar_mat[e, j] = pt[0]
                ystar_mat[e, j] = pt[1]
                zstar_mat[e, j] = pt[2]
                
                if j > 0:
                    dist = np.linalg.norm(pt - prev_pt)
                    curr_xi += dist
                xi_mat[e, j] = curr_xi
                prev_pt = pt

        self._cached_points = np.stack((xstar_mat, ystar_mat, zstar_mat), axis=2)
        return (np.concatenate([xstar_mat.flatten(), ystar_mat.flatten(), zstar_mat.flatten()]), 
                g_mat.flatten() * g_scale, 
                xi_mat.flatten())

    # # -------------------------------------------------------------------------
    # # Point-to-Point (corresponding) Control point constraints
    # # -------------------------------------------------------------------------
    # def _get_cp_matrices(self):
    #     """Reshape flat inputs into (ne, ncp) matrices."""
    #     X = self.xstar_cp.reshape(self.ne, self.ncp)
    #     Y = self.ystar_cp.reshape(self.ne, self.ncp)
    #     Z = self.zstar_cp.reshape(self.ne, self.ncp)
    #     return X, Y, Z

    # def get_cp_distance_vector(self):
    #     """
    #     Calculates Euclidean distance between ALL corresponding control points 
    #     of all curve pairs.
        
    #     Returns:
    #         np.ndarray: 1D array of distances.
    #     """
    #     X, Y, Z = self._get_cp_matrices()
    #     distances = []
        
    #     # Loop over unique pairs of curves (i, j)
    #     for i in range(self.ne):
    #         for j in range(i + 1, self.ne):
    #             # Vectorized distance for ALL CPs in the row
    #             dx = X[i, :] - X[j, :]
    #             dy = Y[i, :] - Y[j, :]
    #             dz = Z[i, :] - Z[j, :]
                
    #             d_row = np.sqrt(dx**2 + dy**2 + dz**2)
    #             distances.append(d_row - self.dist_tol)
                
    #     if not distances:
    #         return np.array([])
            
    #     return np.concatenate(distances)

    # def get_cp_distance_jacobian(self):
    #     """
    #     Calculates Jacobian of the distance vector w.r.t design variables.
    #     Includes ALL control points.
    #     """
    #     X, Y, Z = self._get_cp_matrices()
        
    #     # Count constraints
    #     num_pairs = (self.ne * (self.ne - 1)) // 2
    #     num_constraints = num_pairs * self.ncp
        
    #     # Total Design Variables = 2 * (ne * ncp)
    #     num_dvars = 2 * self.ne * self.ncp
        
    #     Jac = np.zeros((num_constraints, num_dvars))
    #     row_idx = 0
        
    #     for i in range(self.ne):
    #         for j in range(i + 1, self.ne):
    #             # Differences for this pair
    #             dy = Y[i, :] - Y[j, :]
    #             dz = Z[i, :] - Z[j, :]
    #             dx = X[i, :] - X[j, :] 
                
    #             dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-12
                
    #             # Derivatives
    #             grad_y = dy / dist
    #             grad_z = dz / dist
                
    #             # Offsets for global indexing
    #             y_offset_i = i * self.ncp
    #             y_offset_j = j * self.ncp
                
    #             z_offset_i = self.ne * self.ncp + i * self.ncp
    #             z_offset_j = self.ne * self.ncp + j * self.ncp
                
    #             for k in range(self.ncp):
    #                 # Y-derivatives
    #                 Jac[row_idx, y_offset_i + k] =  grad_y[k]
    #                 Jac[row_idx, y_offset_j + k] = -grad_y[k]
                    
    #                 # Z-derivatives
    #                 Jac[row_idx, z_offset_i + k] =  grad_z[k]
    #                 Jac[row_idx, z_offset_j + k] = -grad_z[k]
                    
    #                 row_idx += 1
                    
    #     return Jac
    
    # -------------------------------------------------------------------------
    # Point-to-Point (corresponding) Control point constraints
    # -------------------------------------------------------------------------
    def _get_cp_matrices(self):
        """Reshape flat inputs into (ne, ncp) matrices."""
        X = self.xstar_cp.reshape(self.ne, self.ncp)
        Y = self.ystar_cp.reshape(self.ne, self.ncp)
        Z = self.zstar_cp.reshape(self.ne, self.ncp)
        return X, Y, Z

    def get_cp_distance_vector(self):
        """
        Calculates Euclidean distance between ALL corresponding control points 
        of all curve pairs.
        
        Constraint Form: c = dist_tol - dist
        If c < 0, then dist > dist_tol (Safe/No Overlap)
        """
        X, Y, Z = self._get_cp_matrices()
        distances = []
        
        # Loop over unique pairs of curves (i, j)
        for i in range(self.ne):
            for j in range(i + 1, self.ne):
                # Vectorized distance for ALL CPs in the row
                dx = X[i, :] - X[j, :]
                dy = Y[i, :] - Y[j, :]
                dz = Z[i, :] - Z[j, :]
                
                d_row = np.sqrt(dx**2 + dy**2 + dz**2)
                
                # CHANGED: (dist_tol - dist)
                distances.append(self.dist_tol - d_row)
                
        if not distances:
            return np.array([])
            
        return np.concatenate(distances)

    def get_cp_distance_jacobian(self):
        """
        Calculates Jacobian of the distance vector w.r.t design variables.
        Includes ALL control points.
        
        Since C = Const - dist, the derivatives are NEGATED compared to d(dist).
        """
        X, Y, Z = self._get_cp_matrices()
        
        # Count constraints
        num_pairs = (self.ne * (self.ne - 1)) // 2
        num_constraints = num_pairs * self.ncp
        
        # Total Design Variables = 2 * (ne * ncp)
        num_dvars = 2 * self.ne * self.ncp
        
        Jac = np.zeros((num_constraints, num_dvars))
        row_idx = 0
        
        for i in range(self.ne):
            for j in range(i + 1, self.ne):
                # Differences for this pair
                dy = Y[i, :] - Y[j, :]
                dz = Z[i, :] - Z[j, :]
                dx = X[i, :] - X[j, :] 
                
                dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-12
                
                # Derivatives of distance (d_dist/d_var)
                # We need d(Tol - dist)/d_var = -1 * d_dist/d_var
                grad_y = dy / dist
                grad_z = dz / dist
                
                # Offsets for global indexing
                y_offset_i = i * self.ncp
                y_offset_j = j * self.ncp
                
                z_offset_i = self.ne * self.ncp + i * self.ncp
                z_offset_j = self.ne * self.ncp + j * self.ncp
                
                for k in range(self.ncp):
                    # Y-derivatives (Negated)
                    # Was: +grad_y[k], Now: -grad_y[k]
                    Jac[row_idx, y_offset_i + k] = -grad_y[k]
                    
                    # Was: -grad_y[k], Now: +grad_y[k] (double negative)
                    Jac[row_idx, y_offset_j + k] =  grad_y[k]
                    
                    # Z-derivatives (Negated)
                    Jac[row_idx, z_offset_i + k] = -grad_z[k]
                    Jac[row_idx, z_offset_j + k] =  grad_z[k]
                    
                    row_idx += 1
                    
        return Jac
    
    # -------------------------------------------------------------------------
    # 1. Topology Constraints (Y-Sorting)
    #    y1_c1 < y1_c2 < ... < y1_cn
    # -------------------------------------------------------------------------
    def get_topology_constraints(self):
        """
        Calculates the Y-separation between adjacent channels at every control point.
        Constraint formulation: g = y_{i+1, k} - y_{i, k}
        The optimizer should constrain this g > 0 (or > min_gap).
        
        Returns:
            np.ndarray: 1D array of differences. 
                        Length = (ne - 1) * ncp
        """
        # X, Y, Z shape: (ne, ncp)
        _, Y, _ = self._get_cp_matrices()
        
        # Calculate diff between Row(i+1) and Row(i)
        # Slicing: [1 to end] minus [0 to end-1]
        diffs = Y[1:, :] - Y[:-1, :]
        
        # Flatten to 1D array
        # Order: [ch2-ch1 @cp0, ch2-ch1 @cp1... then ch3-ch2 @cp0...]
        return diffs.flatten()

    def get_topology_jacobian(self):
        """
        Analytical Jacobian for the Y-sorting constraints.
        These are linear constraints, so the Jacobian is constant (1.0 and -1.0).
        """
        num_constraints = (self.ne - 1) * self.ncp
        num_dvars = 2 * self.ne * self.ncp
        
        Jac = np.zeros((num_constraints, num_dvars))
        
        row_idx = 0
        for i in range(self.ne - 1):
            # i is the "lower" channel (coefficient -1)
            # i+1 is the "upper" channel (coefficient +1)
            
            # Start indices in the flat design variable vector
            idx_lower_start = i * self.ncp
            idx_upper_start = (i + 1) * self.ncp
            
            for k in range(self.ncp):
                # Variable indices
                col_upper = idx_upper_start + k
                col_lower = idx_lower_start + k
                
                # Constraint: y_upper - y_lower
                Jac[row_idx, col_upper] =  1.0
                Jac[row_idx, col_lower] = -1.0
                
                row_idx += 1
                
        return Jac

    # -------------------------------------------------------------------------
    # 2. Normal Entry Constraints
    #    y[1] - y[0] = 0  and  z[1] - z[0] = 0
    # -------------------------------------------------------------------------
    def get_normal_entry_constraints(self):
        """
        Calculates the deviation from normality at the inlet (Index 0).
        Constraint: Point[1] - Point[0] = 0
        
        Returns:
            np.ndarray: Concatenated array of [Y_deviations, Z_deviations].
                        Length = 2 * ne
        """
        _, Y, Z = self._get_cp_matrices()
        
        # Y deviation: y_1 - y_0 for all channels
        y_dev = Y[:, 1] - Y[:, 0]
        
        # Z deviation: z_1 - z_0 for all channels
        z_dev = Z[:, 1] - Z[:, 0]
        
        return np.concatenate([y_dev, z_dev])

    def get_normal_entry_jacobian(self):
        """
        Analytical Jacobian for normal entry constraints.
        Linear constraints involving indices 0 and 1 of each channel.
        """
        num_constraints = 2 * self.ne  # ne for Y, ne for Z
        num_dvars = 2 * self.ne * self.ncp
        
        Jac = np.zeros((num_constraints, num_dvars))
        
        # Offset for Z variables in the design vector
        z_offset = self.ne * self.ncp
        
        for i in range(self.ne):
            # --- Y Partials (Rows 0 to ne-1) ---
            # y_1 - y_0
            row_y = i
            idx_y0 = i * self.ncp + 0
            idx_y1 = i * self.ncp + 1
            
            Jac[row_y, idx_y1] =  1.0
            Jac[row_y, idx_y0] = -1.0
            
            # --- Z Partials (Rows ne to 2*ne-1) ---
            # z_1 - z_0
            row_z = self.ne + i
            idx_z0 = z_offset + i * self.ncp + 0
            idx_z1 = z_offset + i * self.ncp + 1
            
            Jac[row_z, idx_z1] =  1.0
            Jac[row_z, idx_z0] = -1.0
            
        return Jac
    
    # -------------------------------------------------------------------------
    # 3. Normal Exit Constraints (ADDED)
    #    y[ncp-1] - y[ncp-2] = 0  and  z[ncp-1] - z[ncp-2] = 0
    # -------------------------------------------------------------------------
    def get_normal_exit_constraints(self):
        """
        Calculates the deviation from normality at the outlet (Last Index).
        Constraint: Point[last] - Point[last-1] = 0
        
        Returns:
            np.ndarray: Concatenated array of [Y_deviations, Z_deviations].
                        Length = 2 * ne
        """
        _, Y, Z = self._get_cp_matrices()
        
        # Y deviation: y_last - y_prev
        y_dev = Y[:, -1] - Y[:, -2]
        
        # Z deviation: z_last - z_prev
        z_dev = Z[:, -1] - Z[:, -2]
        
        return np.concatenate([y_dev, z_dev])

    def get_normal_exit_jacobian(self):
        """
        Analytical Jacobian for normal exit constraints.
        Linear constraints involving indices ncp-1 and ncp-2 of each channel.
        """
        num_constraints = 2 * self.ne
        num_dvars = 2 * self.ne * self.ncp
        
        Jac = np.zeros((num_constraints, num_dvars))
        
        z_offset = self.ne * self.ncp
        
        for i in range(self.ne):
            # --- Y Partials ---
            row_y = i
            idx_y_last = i * self.ncp + (self.ncp - 1)
            idx_y_prev = i * self.ncp + (self.ncp - 2)
            
            Jac[row_y, idx_y_last] =  1.0
            Jac[row_y, idx_y_prev] = -1.0
            
            # --- Z Partials ---
            row_z = self.ne + i
            idx_z_last = z_offset + i * self.ncp + (self.ncp - 1)
            idx_z_prev = z_offset + i * self.ncp + (self.ncp - 2)
            
            Jac[row_z, idx_z_last] =  1.0
            Jac[row_z, idx_z_prev] = -1.0
            
        return Jac
    
    # -------------------------------------------------------------------------
    # 4. Y-Smoothness Constraints (ADDED)
    #    Limits the change between consecutive Y control points.
    #    g = y[k+1] - y[k]
    #    Constraint: -y_bbox/2 < g < y_bbox/2
    # -------------------------------------------------------------------------
    def get_y_smoothness_constraints(self):
        """
        Calculates the difference between consecutive Y control points 
        within each channel.
        
        Returns:
            np.ndarray: 1D array of differences (y_{k+1} - y_k).
                        Length = ne * (ncp - 1)
        """
        _, Y, _ = self._get_cp_matrices()
        
        # Calculate diff between Col(k+1) and Col(k) within the same row
        # Y shape is (ne, ncp)
        # diffs shape is (ne, ncp-1)
        diffs = Y[:, 1:] - Y[:, :-1]
        
        return diffs.flatten()

    def get_y_smoothness_jacobian(self):
        """
        Analytical Jacobian for Y-smoothness.
        """
        num_constraints = self.ne * (self.ncp - 1)
        num_dvars = 2 * self.ne * self.ncp
        
        Jac = np.zeros((num_constraints, num_dvars))
        
        row_idx = 0
        for i in range(self.ne):
            # For each channel, we have ncp-1 constraints
            # Constraint k corresponds to y_{i, k+1} - y_{i, k}
            
            # Start index for this channel's Y variables
            idx_start = i * self.ncp
            
            for k in range(self.ncp - 1):
                idx_k = idx_start + k
                idx_k_plus_1 = idx_start + k + 1
                
                # wrt y_{k+1} is +1
                Jac[row_idx, idx_k_plus_1] = 1.0
                # wrt y_{k} is -1
                Jac[row_idx, idx_k] = -1.0
                
                row_idx += 1
                
        return Jac
        
    def calcConstraints(self, rho=50.0):
        if self._cached_points is None: self.Interpolate()
        pts = self._cached_points
        tolerance = self.dist_tol
        
        all_g = []
        for i in range(self.ne):
            for j in range(i + 1, self.ne):
                diff = pts[i][:, None, :] - pts[j][None, :, :]
                dists = np.linalg.norm(diff, axis=2)
                all_g.append((tolerance - dists).flatten())
        
        if not all_g: return -999.0
        g = np.concatenate(all_g)
        g_max = np.max(g)
        avg_exp = np.mean(np.exp(rho * (g - g_max)))
        return g_max + (1.0/rho) * np.log(avg_exp)

    def getInterpolationGradient_SingleCP(self, global_idx):
        is_z = global_idx >= self.ncp_per_dim
        local_idx = global_idx - self.ncp_per_dim if is_z else global_idx
        curve_e = local_idx // self.ncp
        cp_k    = local_idx % self.ncp
        active_dim = 2 if is_z else 1
        
        if self._cached_points is None: self.Interpolate()
        
        curve = BSpline.Curve()
        curve.degree = self.degree
        cx = self.xstar_cp.reshape(self.ne, self.ncp)[curve_e]
        cy = self.ystar_cp.reshape(self.ne, self.ncp)[curve_e]
        cz = self.zstar_cp.reshape(self.ne, self.ncp)[curve_e]
        curve.ctrlpts = np.column_stack((cx, cy, cz)).tolist()
        curve.knotvector = utilities.generate_knot_vector(self.degree, self.ncp)
        
        u_vals = np.linspace(0.0, 1.0, self.nxi)
        curr_tan, curr_pos = [], []
        for u in u_vals:
            res = curve.derivatives(u, order=1)
            curr_pos.append(res[0])
            curr_tan.append(res[1])
        curr_tan = np.array(curr_tan)
        curr_pos = np.array(curr_pos)

        N_val, dN_val = self._get_basis_and_deriv_vector(cp_k)
        
        total_pts = self.ne * self.nxi
        grad_vxyz = np.zeros(3 * total_pts)
        grad_g    = np.zeros(total_pts)
        grad_xi   = np.zeros(total_pts)
        
        start = curve_e * self.nxi
        end   = start + self.nxi
        
        offset = active_dim * total_pts
        grad_vxyz[offset + start : offset + end] = N_val
        
        g_vec = np.array([0.0, 0.0, -9.81])
        g_scale = 1.0 / self.char_props.get("g_c", 1.0)
        
        for j in range(self.nxi):
            v = curr_tan[j]
            v_norm = np.linalg.norm(v) + 1e-12
            dv_dP = np.zeros(3); dv_dP[active_dim] = dN_val[j]
            A = np.dot(g_vec, v); dA = np.dot(g_vec, dv_dP)
            B = v_norm; dB = np.dot(v, dv_dP) / B
            grad_g[start + j] = ((B * dA - A * dB) / B**2) * g_scale

        dPos_dP = np.zeros((self.nxi, 3)); dPos_dP[:, active_dim] = N_val
        curr_grad_xi = 0.0
        for j in range(1, self.nxi):
            S = curr_pos[j] - curr_pos[j-1]; S_len = np.linalg.norm(S)
            if S_len > 1e-12:
                diff_d = dPos_dP[j] - dPos_dP[j-1]
                curr_grad_xi += np.dot(S/S_len, diff_d)
            grad_xi[start + j] = curr_grad_xi
            
        return grad_vxyz, grad_g, grad_xi

    def getConstraintGradient_SingleCP(self, global_idx, rho=50.0):
        is_z = global_idx >= self.ncp_per_dim
        local_idx = global_idx - self.ncp_per_dim if is_z else global_idx
        curve_e = local_idx // self.ncp
        cp_k    = local_idx % self.ncp
        active_dim = 2 if is_z else 1
        
        if self._cached_points is None: self.Interpolate()
        pts = self._cached_points; tol = self.dist_tol
        N_val, _ = self._get_basis_and_deriv_vector(cp_k)
        
        all_g = []
        for i in range(self.ne):
            for j in range(i + 1, self.ne):
                diff = pts[i][:, None, :] - pts[j][None, :, :]
                dists = np.linalg.norm(diff, axis=2)
                all_g.append((tol - dists).flatten())
        
        if not all_g: return 0.0
        g_flat = np.concatenate(all_g)
        g_max = np.max(g_flat)
        exp_terms = np.exp(rho * (g_flat - g_max))
        sum_exp = np.sum(exp_terms) 
        
        total_grad = 0.0
        others = [c for c in range(self.ne) if c != curve_e]
        for other in others:
            vec = pts[curve_e][:, None, :] - pts[other][None, :, :]
            dist = np.linalg.norm(vec, axis=2)
            g = tol - dist
            term_exp = np.exp(rho * (g - g_max))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                grad_dir = vec[:, :, active_dim] / (dist + 1e-12)
                grad_dir[dist < 1e-9] = 0.0
                
            grad_pair = np.sum(term_exp * (-grad_dir) * N_val[:, None])
            total_grad += grad_pair
            
        return total_grad / sum_exp

    # -------------------------------------------------------------------------
    # The control point j+1 of channel. e, should lie within the circle of -
    # radius dx enclosed by control point j on the Y-Z plane at j+1
    # -------------------------------------------------------------------------
    def calcSlewConstraints(self):
        X, Y, Z = self._get_cp_matrices()
        cons = []
        for e in range(self.ne):
            for j in range(1, self.ncp):
                dx_cp = X[e, j] - X[e, j-1]
                ycenter = Y[e, j-1]
                zcenter = Z[e, j-1]
                cons.append( ( (Y[e, j] - ycenter)**2 + (Z[e, j] - zcenter)**2 ) - dx_cp**2 )
        return np.array(cons)

    def calcSlewConstraintsJacobian(self):
        """
        Calculates the gradients (Jacobian matrix) for the slew constraints
        with respect to the design variables (Y and Z control points only).
        
        Design Variable Ordering:
        [Y_ch1_cp1, Y_ch1_cp2..., Y_ch2_cp1..., Z_ch1_cp1..., Z_ch2_cp1...]
        """
        X, Y, Z = self._get_cp_matrices()
        
        # Dimensions
        num_y_vars = self.ne * self.ncp
        num_des_vars = 2 * num_y_vars  # Total Ys + Total Zs
        num_cons = self.ne * (self.ncp - 1) # One constraint per segment per channel
        
        # Initialize dense Jacobian matrix
        jac = np.zeros((num_cons, num_des_vars))
        
        row_idx = 0
        for e in range(self.ne):
            for j in range(1, self.ncp):
                # Calculate differences
                y_diff = Y[e, j] - Y[e, j-1]
                z_diff = Z[e, j] - Z[e, j-1]
                
                # Derivatives of C = (y_diff^2 + z_diff^2) - dx^2
                
                # 1. Y Derivatives
                # dC/dY_j = 2 * y_diff * (1) = 2 * y_diff
                # dC/dY_{j-1} = 2 * y_diff * (-1) = -2 * y_diff
                grad_y_curr = 2 * y_diff
                grad_y_prev = -2 * y_diff
                
                # 2. Z Derivatives
                # dC/dZ_j = 2 * z_diff
                # dC/dZ_{j-1} = -2 * z_diff
                grad_z_curr = 2 * z_diff
                grad_z_prev = -2 * z_diff
                
                # Calculate Global Indices (Columns)
                # Y indices
                col_y_prev = (e * self.ncp) + (j - 1)
                col_y_curr = (e * self.ncp) + j
                
                # Z indices (offset by total number of Ys)
                col_z_prev = num_y_vars + (e * self.ncp) + (j - 1)
                col_z_curr = num_y_vars + (e * self.ncp) + j
                
                # Populate Jacobian Row
                jac[row_idx, col_y_curr] = grad_y_curr
                jac[row_idx, col_y_prev] = grad_y_prev
                jac[row_idx, col_z_curr] = grad_z_curr
                jac[row_idx, col_z_prev] = grad_z_prev
                
                row_idx += 1
                
        return jac
    
# ==============================================================================
# 2. Combined V-Plot Test Suite (Corrected)
# ==============================================================================
class TestGlobalGradients(unittest.TestCase):
    
    def setUp(self):
        config = {
            "design_params": { "ne": 2, "ncp": 4, "nxi": 15 },
            "spline": { 
                "x_cp": [0.0, 1.0, 2.0, 3.0] * 2,
                "degree": 3,
                "dist_tol": 0.5
            }
        }
        char_props = {"L_c": 1.0, "g_c": 9.81}
        self.gen = GenBSplineChannels(config, char_props)
        
        self.num_inputs = 16 
        
        # Random inputs ensuring collision for constraint test
        np.random.seed(42)
        self.inputs = np.random.uniform(-0.3, 0.3, self.num_inputs)
        self.inputs[4:8] = 0.1; self.inputs[8:12] = 0.2 # Force closeness
        self.gen.setInputs(self.inputs)
        
        # Get baseline outputs sizes
        v, g, xi = self.gen.Interpolate()
        self.size_v, self.size_g, self.size_xi = v.size, g.size, xi.size
        
        # Constraint size
        d_vec = self.gen.get_cp_distance_vector()
        self.size_cp = d_vec.size

    def compute_full_analytic_jacobian(self):
        """Builds Full Jacobian for Physics + Constraints."""
        J_V  = np.zeros((self.size_v, self.num_inputs))
        J_G  = np.zeros((self.size_g, self.num_inputs))
        J_Xi = np.zeros((self.size_xi, self.num_inputs))
        
        # Physics: Column-by-column assembly
        for k in range(self.num_inputs):
            # --- FIX IS HERE ---
            # Was: gv, gg, gxi = self.gen.getConstraintGradient_SingleCP(k)
            # Now: Use the correct physics gradient method
            gv, gg, gxi = self.gen.getInterpolationGradient_SingleCP(k)
            
            J_V[:, k]  = gv
            J_G[:, k]  = gg
            J_Xi[:, k] = gxi
            
        # Constraint: Direct matrix return (Row-based)
        J_CP = self.gen.get_cp_distance_jacobian()
            
        return J_V, J_G, J_Xi, J_CP

    def compute_full_fd_jacobian(self, h):
        """FD Jacobian for step h."""
        base_v, base_g, base_xi = self.gen.Interpolate()
        base_cp = self.gen.get_cp_distance_vector()
        
        J_V_fd  = np.zeros((self.size_v, self.num_inputs))
        J_G_fd  = np.zeros((self.size_g, self.num_inputs))
        J_Xi_fd = np.zeros((self.size_xi, self.num_inputs))
        J_CP_fd = np.zeros((self.size_cp, self.num_inputs))
        
        for k in range(self.num_inputs):
            pert = self.inputs.copy()
            pert[k] += h
            self.gen.setInputs(pert)
            
            new_v, new_g, new_xi = self.gen.Interpolate()
            new_cp = self.gen.get_cp_distance_vector()
            
            J_V_fd[:, k]  = (new_v - base_v) / h
            J_G_fd[:, k]  = (new_g - base_g) / h
            J_Xi_fd[:, k] = (new_xi - base_xi) / h
            J_CP_fd[:, k] = (new_cp - base_cp) / h
            
        self.gen.setInputs(self.inputs)
        return J_V_fd, J_G_fd, J_Xi_fd, J_CP_fd

    def test_global_v_plot(self):
        print("\n=== Full Jacobian Verification (Physics + CP Dist) ===")
        
        # 1. Exact Jacobians
        J_V, J_G, J_Xi, J_CP = self.compute_full_analytic_jacobian()
        
        step_sizes = np.logspace(-1, -12, 12)
        err_v, err_g, err_xi, err_cp = [], [], [], []
        
        # Helper: Relative Frobenius Error
        def get_err(Analytic, FD):
            norm = np.linalg.norm(Analytic)
            return np.linalg.norm(Analytic - FD) / (norm + 1e-10)

        # 2. Loop
        for h in step_sizes:
            J_V_fd, J_G_fd, J_Xi_fd, J_CP_fd = self.compute_full_fd_jacobian(h)
            
            err_v.append(get_err(J_V, J_V_fd))
            err_g.append(get_err(J_G, J_G_fd))
            err_xi.append(get_err(J_Xi, J_Xi_fd))
            err_cp.append(get_err(J_CP, J_CP_fd))

        # 3. Assert
        min_cp_err = min(err_cp)
        print(f"    Min CP Dist Error: {min_cp_err:.2e}")
        self.assertLess(min_cp_err, 1e-5)

        # 4. Plot
        plt.figure(figsize=(10, 8))
        plt.loglog(step_sizes, err_v, 'o-', label='Coords (Physics)')
        plt.loglog(step_sizes, err_g, 's-', label='Gravity (Physics)')
        plt.loglog(step_sizes, err_xi, '^-', label='ArcLen (Physics)')
        plt.loglog(step_sizes, err_cp, 'd-', linewidth=2.5, color='red', label='CP Dist (Constraint)')
        
        # Truncation Reference (Anchor to CP Dist)
        min_idx = np.argmin(err_cp)
        ref_y = err_cp[min_idx] * (step_sizes / step_sizes[min_idx])
        plt.loglog(step_sizes, ref_y, 'k--', alpha=0.4, label='O(h) Ref')

        plt.xlabel("Step Size (h)")
        plt.ylabel("Relative Matrix Error (Frobenius)")
        plt.title("Jacobian V-Plot: Physics + Constraints")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    unittest.main()
