from sph_base import *

class PBDFluid(SPHBase):
    
    def __init__(self, particle_system, solver_iter_cnt = 4, relax = 1e-4, k = 0.1, c = 0.01):
        super().__init__(particle_system)
        self.solver_iter_cnt = solver_iter_cnt
        self.relax = relax
        self.delta_q = 0.1 * self.ps.support_radius
        self.k = k
        self.c = c

    @ti.kernel
    def integrate(self):
        for i in range(self.ps.particle_num[None]):
            self.ps.v[i] += self.g * self.dt[None]
            self.ps.pred_x[i] += self.ps.v[i] * self.dt[None]
            self.ps.lambda_[i] = 0.0

    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.pred_x[p_i]
            self.ps.density[p_i] = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.pred_x[p_j]
                self.ps.density[p_i] += self.ps.m_V * self.cubic_kernel((x_i - x_j).norm())
            self.ps.density[p_i] *= self.density_0

    @ti.kernel
    def calculate_lamdba(self):
        for p_i in range(self.ps.particle_num[None]):
            constraint = self.ps.density[p_i] / self.density_0 - 1.0
            x_i = self.ps.pred_x[p_i]

            d_constraint_i = ti.Vector.zero(float, self.ps.dim)
            d_constraint_j_sqr = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.pred_x[p_j]
                
                dist = x_i - x_j
                r = dist.norm()
                d_constraint_i += 1.0 / self.density_0 * self.cubic_kernel_derivative(dist)
                d_constraint_j = -1.0 / self.density_0 * self.cubic_kernel_derivative(dist)
                d_constraint_j_sqr += d_constraint_j.dot(d_constraint_j)

            self.ps.lambda_[p_i] = -constraint / (d_constraint_i.dot(d_constraint_i) + d_constraint_j_sqr + self.relax)


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
                correction = - self.k * ti.math.pow(self.cubic_kernel(r) / self.cubic_kernel(self.delta_q), 4)
                self.ps.delta_p[p_i] += (lambda_i + lambda_j + correction) * (1 / self.density_0) * self.cubic_kernel_derivative(dist)
            self.ps.pred_x[p_i] = self.ps.pred_x[p_i] + self.ps.delta_p[p_i]
                

    # @ti.kernel
    # def calculate_pred_x(self):
    #     for p_i in range(self.ps.particle_num[None]):
    #         self.ps.pred_x[p_i] = self.ps.pred_x[p_i] + self.ps.delta_p[p_i]


    def substep(self):
        self.compute_densities()
        self.calculate_lamdba()
        self.calculate_delta_p()
        # self.calculate_pred_x()

    @ti.kernel
    def update_position_and_velocity(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.v[p_i] = (self.ps.pred_x[p_i] - self.ps.x[p_i]) / self.dt[None]
            self.ps.x[p_i] = self.ps.pred_x[p_i]

    @ti.kernel
    def apply_viscosity(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.pred_x[p_i]
            self.ps.tmp_v[p_i] = self.ps.v[p_i] * (1.0 - self.viscosity)          
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.pred_x[p_j]
                r = (x_i - x_j).norm()

                self.ps.tmp_v[p_i] += self.cubic_kernel(r) * self.viscosity * self.ps.v[p_j]

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
        self.apply_viscosity_pingpong()
        
