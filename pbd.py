from sph_base import *
from particle_system import ParticleSystem


@ti.data_oriented
class PBDParticleSystem(ParticleSystem):
    """Particle system dedicated to the PBD solver.

    It reuses everything from ParticleSystem and only adds the PBD-specific
    per-particle state. The spatial grid and neighbor lists are built from
    ``pred_x`` instead of ``x`` so that the neighbor search stays consistent
    with the positions the PBD constraints actually operate on.
    """

    def __init__(self, res):
        super().__init__(res)
        # PBD-only particle state.
        self.pred_x = ti.Vector.field(self.dim, dtype=float)
        self.lambda_ = ti.field(dtype=float)
        self.delta_p = ti.Vector.field(self.dim, dtype=float)
        self.tmp_v = ti.Vector.field(self.dim, dtype=float)
        self.particles_node.place(self.pred_x, self.lambda_, self.delta_p, self.tmp_v)

    @ti.func
    def search_position(self, p):
        # Build the grid / neighbor lists from the predicted positions.
        return self.pred_x[p]

    @ti.func
    def init_particle_extra(self, p, x, v):
        self.pred_x[p] = x
        self.lambda_[p] = 0.0
        self.delta_p[p] = ti.Vector.zero(float, self.dim)
        self.tmp_v[p] = v


class PBDFluid(SPHBase):
    
    def __init__(self, particle_system, solver_iter_cnt = 6, relax = 4e-2, k = 0.0, c = 0.01, dt = 1e-2):
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
        self.max_speed = 10.0

    @ti.kernel
    def integrate(self):
        for i in range(self.ps.particle_num[None]):
            self.ps.v[i] += ti.Vector([0.0, self.g]) * self.dt[None]
            self.ps.pred_x[i] = self.ps.x[i] + self.ps.v[i] * self.dt[None]
            self.ps.lambda_[i] = 0.0


    @ti.kernel
    def calculate_lamdba(self):
        for p_i in range(self.ps.particle_num[None]):
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
            dir = self.ps.delta_p[p_i] / (self.ps.delta_p[p_i].norm() + 1e-8);
            self.ps.delta_p[p_i] = ti.math.min(self.ps.delta_p[p_i].norm(), self.max_speed * self.dt[None]) * dir

    @ti.kernel
    def calculate_pred_x(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.pred_x[p_i] = self.ps.pred_x[p_i] + self.ps.delta_p[p_i]

    @ti.func
    def simulate_collisions(self, p_i, vec, d):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.3
        self.ps.delta_p[p_i] += vec * d * (1 + c_f)


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

    @ti.kernel
    def copy_viscosity_to_velocity(self):
        # Copy the smoothed velocities back into v. This is a separate pass
        # because apply_viscosity reads v of the neighbors, so it cannot write
        # v in place without a race.
        for p_i in range(self.ps.particle_num[None]):
            self.ps.v[p_i] = self.ps.tmp_v[p_i]

    def apply_viscosity_pass(self):
        self.apply_viscosity()
        self.copy_viscosity_to_velocity()

        
    def step(self):
        self.integrate()
        for _ in range(self.solver_iter_cnt):
            self.ps.initialize_particle_system()
            self.substep()
        self.update_position_and_velocity()
        self.apply_viscosity_pass()
        
