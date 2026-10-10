from sph_base import *

class PBDFluid(SPHBase):
    
    def __init__(self, particle_system, solver_iter_cnt = 6, relax = 4e-2, k = 0.0, c = 0.01, dt = 4e-3):
        super().__init__(particle_system)
        # PBF stores number density sum(W), so its rest density is rho_0 / m.
        self.solver_iter_cnt = solver_iter_cnt
        self.relax = relax
        self.delta_q = 0.1 * self.ps.support_radius
        self.k = k
        self.c = c
        self.dt[None] = dt
        self.m_V_override = self.ps.particle_radius ** 2
        self.mass_override = self.density_0 * self.m_V_override

    @ti.kernel
    def integrate(self):
        for i in range(self.ps.particle_num[None]):
            self.ps.v[i] += ti.Vector([0.0, self.g]) * self.dt[None]
            # Reset predicted position from current position (which may have been
            # clamped by enforce_boundary), not by accumulating on the old pred_x.
            self.ps.pred_x[i] = self.ps.x[i] + self.ps.v[i] * self.dt[None]
            self.ps.lambda_[i] = 0.0


    @ti.kernel
    def calculate_lamdba(self):
        for p_i in range(self.ps.particle_num[None]):
            # Only correct over-density; correcting under-density makes the
            # solver oscillate and diverge.
            constraint = ti.max(self.ps.density[p_i] / self.density_0 - 1.0, 0.0)
            x_i = self.ps.pred_x[p_i]

            d_constraint_i = ti.Vector.zero(float, self.ps.dim)
            d_constraint_j_sqr = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.pred_x[p_j]
                
                dist = x_i - x_j
                r = dist.norm()

                d_constraint_i += 1.0 / self.density_0 * self.cubic_kernel_derivative(dist) * self.mass_override
                d_constraint_j = 1.0 / self.density_0 * self.cubic_kernel_derivative(dist) * self.mass_override
                d_constraint_j_sqr += d_constraint_j.dot(d_constraint_j)

            self.ps.lambda_[p_i] = -constraint / (d_constraint_i.dot(d_constraint_i) + d_constraint_j_sqr + self.relax)

    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.pred_x[p_i]
            self.ps.density[p_i] = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.pred_x[p_j]
                self.ps.density[p_i] += self.cubic_kernel((x_i - x_j).norm()) * self.mass_override

    @ti.kernel
    def calculate_delta_p(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.delta_p[p_i] = 0.0
            lambda_i = self.ps.lambda_[p_i]
            x_i = self.ps.pred_x[p_i]
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                lambda_j = self.ps.lambda_[p_j]
                x_j = self.ps.pred_x[p_j]

                dist = x_i - x_j
                r = dist.norm()
                correction = - self.k *  ti.math.pow(ti.math.min(self.cubic_kernel(r) / self.cubic_kernel(self.delta_q), 1.0), 4)
                self.ps.delta_p[p_i] += (lambda_i + lambda_j + correction) * 1.0 / self.density_0  * self.cubic_kernel_derivative(dist)

    @ti.kernel
    def calculate_pred_x(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.pred_x[p_i] = self.ps.pred_x[p_i] + self.ps.delta_p[p_i]

    @ti.func
    def simulate_collisions(self, p_i, vec, d):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.3
        self.ps.delta_p[p_i] += vec * d


    @ti.kernel
    def enforce_boundary(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.dim == 2:
                if self.ps.material[p_i] == self.ps.material_fluid:
                    pos = self.ps.pred_x[p_i] + self.ps.delta_p[p_i]
                    if pos[0] < self.ps.padding:
                        self.simulate_collisions(
                            p_i, ti.Vector([1.0, 0.0]),
                            self.ps.padding - pos[0])
                    if pos[0] > self.ps.bound[0] - self.ps.padding:
                        self.simulate_collisions(
                            p_i, ti.Vector([-1.0, 0.0]),
                            pos[0] - (self.ps.bound[0] - self.ps.padding))
                    if pos[1] > self.ps.bound[1] - self.ps.padding:
                        self.simulate_collisions(
                            p_i, ti.Vector([0.0, -1.0]),
                            pos[1] - (self.ps.bound[1] - self.ps.padding))
                    if pos[1] < self.ps.padding:
                        self.simulate_collisions(
                            p_i, ti.Vector([0.0, 1.0]),
                            self.ps.padding - pos[1])


    def substep(self):
        self.compute_densities()
        self.calculate_lamdba()
        self.calculate_delta_p()
        self.enforce_boundary()
        self.calculate_pred_x()
        


    @ti.kernel
    def update_position_and_velocity(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.v[p_i] = (self.ps.pred_x[p_i] - self.ps.x[p_i]) / self.dt[None]
            if p_i == 0:
                print('x: {}, v: {} pred_x: {}'.format(self.ps.x[p_i], self.ps.v[p_i], self.ps.pred_x[p_i]))
            self.ps.x[p_i] = self.ps.pred_x[p_i]
            

    @ti.kernel
    def apply_viscosity(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.pred_x[p_i]
            weighted_v = ti.Vector.zero(float, self.ps.dim)
            sum_w = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.pred_x[p_j]
                r = (x_i - x_j).norm()
                w = self.cubic_kernel(r)
                weighted_v += w * self.ps.v[p_j]
                sum_w += w
            if sum_w > 1e-8:
                self.ps.tmp_v[p_i] = (1.0 - self.viscosity) * self.ps.v[p_i] + self.viscosity * weighted_v / sum_w
            else:
                self.ps.tmp_v[p_i] = self.ps.v[p_i]

    def apply_viscosity_pingpong(self):
        self.apply_viscosity()
        tmp = self.ps.tmp_v
        self.ps.tmp_v = self.ps.v
        self.ps.v = tmp

        
    def step(self):
        self.integrate()
        self.ps.initialize_particle_system()
        for _ in range(self.solver_iter_cnt):
            self.substep()
        self.update_position_and_velocity()
        # self.apply_viscosity_pingpong()
        
