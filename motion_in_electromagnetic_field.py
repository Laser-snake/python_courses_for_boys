
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class Phys:
    charge_part = 1
    mass_part = 1
    speed_light = 1


class Grid:
    def __init__(self, L_X, L_Y, L_Z, N_X, N_Y, N_Z, N_steps, T_max):
        x = [(L_X/N_X)*(i - N_X/2) for i in range(0, N_X)]
        y = [(L_Y/N_Y)*(i - N_X/2) for i in range(0, N_Y)]
        z = [(L_Y/N_Y)*(i - N_X/2) for i in range(0, N_Y)]
        t = [(T_max/N_steps)*(i) for i in range(0, N_steps)]
        self.x, self.y, self.z = np.meshgrid(x, y, z)
        self.grid_points_x = N_X
        self.grid_points_y = N_Y
        self.grid_points_z = N_Y
        self.t = t
        self.dt = T_max/N_steps
        self.time = T_max


class ElectromagneticField:
    def __init__(self, grid_class):
        self.nx = grid_class.grid_points_x
        self.ny = grid_class.grid_points_y
        self.nz = grid_class.grid_points_z
        self.x = grid_class.x
        self.y = grid_class.y
        self.z = grid_class.z

    def electro_component_x(self, x, y, z, amplitude):
        self.electro_field_x = (np.zeros((self.nx, self.ny, self.nz)) +
                                amplitude)

    def electro_component_y(self, x, y, z, amplitude):
        self.electro_field_y = (np.zeros((self.nx, self.ny, self.nz)) +
                                amplitude)

    def electro_component_z(self, x, y, z, amplitude):
        self.electro_field_z = np.zeros((self.nx, self.ny, self.nz))+amplitude

    def magnetic_component_x(self, x, y, z, amplitude):
        self.magnetic_field_x = np.zeros((self.nx, self.ny, self.nz))+amplitude

    def magnetic_component_y(self, x, y, z, amplitude):
        self.magnetic_field_y = np.zeros((self.nx, self.ny, self.nz))+amplitude

    def magnetic_component_z(self, x, y, z, amplitude):
        self.magnetic_field_z = np.zeros((self.nx, self.ny, self.nz))+amplitude

    def electro_x_for_equation(self, x, y, z, t, params=[0, 0, 0]):
        self.electro_x = params[0]

    def electro_y_for_equation(self, x, y, z, t, params=[0, 0, 0]):
        self.electro_y = params[1]

    def electro_z_for_equation(self, x, y, z, t, params=[0, 0, 0]):
        self.electro_z = params[2]

    def magnetic_x_for_equation(self, x, y, z, t, params=[0, 0, 0]):
        self.magnetic_x = params[0]

    def magnetic_y_for_equation(self, x, y, z, t, params=[0, 0, 0]):
        self.magnetic_y = params[1]

    def magnetic_z_for_equation(self, x, y, z, t, params=[0, 0, 0]):
        self.magnetic_z = params[2]

    def electro_vector_for_epuation(self, Ex, Ey, Ez):
        self.electro_vector = np.array([Ex, Ey, Ez])

    def magnetic_vector_for_epuation(self, Hx, Hy, Hz):
        self.magnetic_vector = np.array([Hx, Hy, Hz])

    def field_intensity(self):
        intensity = (np.multiply(self.electro_field_x, self.electro_field_x) +
                     np.multiply(self.electro_field_y, self.electro_field_y) +
                     np.multiply(self.electro_field_z, self.electro_field_z) +
                     np.multiply(self.magnetic_field_x, self.magnetic_field_x)
                     + np.multiply(self.magnetic_field_y,
                                   self.magnetic_field_y)
                     + np.multiply(self.magnetic_field_z,
                                   self.magnetic_field_z))/(4.0 * np.pi)
        return intensity

    def field_visualization_xy(self,z):
        plt.figure(figsize=(10, 10))
        plt.streamplot(self.x[:,:,0], self.y[:, :, 0],
                       self.electro_field_x[:, :, 0],
                       self.electro_field_y[:, :, 0], density=1.4,
                       linewidth=0.5, color='#A23BEC')
        #plt.streamplot(self.x[:, :, 0], self.y[:,:,0], self.magnetic_field_x[:, :, 0], self.magnetic_field_y[:,:,0], density=1.4, linewidth=None, color='red')
        plt.title('Electromagnetic Field')
        plt.xlabel("xlabel")
        plt.ylabel("ylabel")
        plt.legend(['Exy'])
        plt.grid()
        plt.show()

    def field_visualization_yz(self):
        plt.figure(figsize=(10, 10))
        print(self.electro_field_z[:, 0].shape)
        plt.streamplot(self.z[:,0], self.y[:,:,0], self.electro_field_z[0], self.electro_field_y[0], density=1.4, linewidth=None, color='#A23BEC')
        plt.streamplot(self.z[:,0], self.y[:,:,0], self.magnetic_field_z[0], self.magnetic_field_y[0], density=1.4, linewidth=None, color='red')
        plt.title('Electromagnetic Field')
        plt.xlabel("zlabel")
        plt.ylabel("ylabel")
        plt.legend(['Ezy', 'Hzy'])
        plt.grid()
        plt.show()

    def field_visualization_zx(self):
        plt.figure(figsize=(10, 10))
        plt.streamplot(self.z[:, 0], self.x[0], self.electro_field_z[:, 0], self.electro_field_x[:, 0], density=1.4, linewidth=None, color='#A23BEC')
        plt.streamplot(self.z[:, 0], self.x[0], self.magnetic_field_z[:, 0], self.magnetic_field_x[:, 0], density=1.4, linewidth=None, color='red')
        plt.title('Electromagnetic Field')
        plt.xlabel("zlabel")
        plt.ylabel("xlabel")
        plt.grid()
        plt.show()

    def magneticfield_3d(self):
        c = np.sqrt(np.abs(self.magnetic_field_x) ** 2 +
                    np.abs(self.magnetic_field_y) ** 2 +
                    np.abs(self.magnetic_field_z) ** 2)
        c = (c.ravel() - c.min()) / (c.ptp() + 0.1)

        # Repeat for each body line and two head lines
        c = np.concatenate((c, np.repeat(c, 2)))
        # Colormap
        c = plt.cm.jet(c)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.quiver(self.x, self.y, self.z, self.magnetic_field_x,
                  self.magnetic_field_y, self.magnetic_field_z, colors=c,
                  length=0.5)
        plt.show()

    def electrofield_3d(self):
        c = np.sqrt(np.abs(self.electro_field_x) ** 2 +
                    np.abs(self.electro_field_y) ** 2 +
                    np.abs(self.electro_field_z) ** 2)
        c = (c.ravel() - c.min()) / (c.ptp() + 0.1)
        # Repeat for each body line and two head lines
        c = np.concatenate((c, np.repeat(c, 2)))
        # Colormap
        c = plt.cm.jet(c)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.quiver(self.x, self.y, self.z, self.electro_field_x,
                  self.electro_field_y, self.electro_field_z, colors=c,
                  length=0.5)
        plt.show()


class Particle:
    def __init__(self, grid_class, x0=0., y0=0., z0=0., px_0=0., py_0=0.1, pz_0=0.):
        self.nx = grid_class.grid_points_x
        self.ny = grid_class.grid_points_y
        self.nz = grid_class.grid_points_z
        self.x = grid_class.x
        self.y = grid_class.y
        self.z = grid_class.z
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.px_0 = px_0
        self.py_0 = py_0
        self.pz_0 = pz_0
        self.t = np.array(grid_class.t)
        self.dt = grid_class.dt
        self.time = grid_class.time
        self.inition_conduction()

    def inition_conduction(self):
        print('Начальные координаты:\n')
        print(f'x0={self.x0} \ny0={self.y0} \nz0={self.z0} \n')
        print(f'px_0={self.px_0} \npy_0={self.py_0} \npz_0={self.pz_0} \n')

    def solver(self, field_class, params_electro, params_magnetic):
        def gamm(px, py, pz):
            return np.sqrt(1 + (np.multiply(px, px) + np.multiply(py, py) +
                                np.multiply(pz, pz)))

        def func_dpx(px, py, pz, Hz, Hy, Ex):
            return (Ex + 1/(gamm(px, py, pz)) * (np.multiply(py, Hz) -
                                                 np.multiply(Hy, pz)))

        def func_dpy(px, py, pz, Hx, Hz, Ey):
            return (Ey - 1/(gamm(px, py, pz)) * (np.multiply(px, Hz) -
                                                 np.multiply(Hx, pz)))

        def func_dpz(px, py, pz, Hx, Hy, Ez):
            return (Ez + 1/(gamm(px, py, pz)) * (np.multiply(px, Hy) -
                                                 np.multiply(Hx, py)))

        def func_dx(px, py, pz):
            return px/(gamm(px, py, pz))

        def func_dy(px, py, pz):
            return py/(gamm(px, py, pz))

        def func_dz(px, py, pz):
            return pz/(gamm(px, py, pz))

        def system_equation(t, g, params_electro, params_magnetic):
            px, py, pz, x, y, z = g
            field_class.electro_x_for_equation(x, y, z, t, params_electro)
            field_class.electro_y_for_equation(x, y, z, t, params_electro)
            field_class.electro_z_for_equation(x, y, z, t, params_electro)
            field_class.magnetic_x_for_equation(x, y, z, t, params_magnetic)
            field_class.magnetic_y_for_equation(x, y, z, t, params_magnetic)
            field_class.magnetic_z_for_equation(x, y, z, t, params_magnetic)
            Ex = field_class.electro_x
            Ey = field_class.electro_y
            Ez = field_class.electro_z
            Hx = field_class.magnetic_x
            Hy = field_class.magnetic_y
            Hz = field_class.magnetic_z
            dpx = func_dpx(px, py, pz, Hz, Hy, Ex)
            dpy = func_dpy(px, py, pz, Hx, Hz, Ey)
            dpz = func_dpz(px, py, pz, Hx, Hy, Ez)
            dx = func_dx(px, py, pz)
            dy = func_dy(px, py, pz)
            dz = func_dz(px, py, pz)
            return [dpx, dpy, dpz, dx, dy, dz]
        sol = solve_ivp(system_equation, [0, self.time],
                        [self.px_0, self.py_0,
                         self.pz_0, self.x0, self.y0, self.z0],
                        args=(params_electro, params_magnetic),
                        dense_output=True)
        return sol

    def trac_partical_xy(self, z):
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(z.T[:, 3], z.T[:, 4])
        ax.set_xlabel('x/lc')
        ax.set_ylabel('y/lc')
        plt.show()

    def trac_partical_yz(self, z):
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(z.T[:, 4], z.T[:, 5])
        ax.set_xlabel('y/lc')
        ax.set_ylabel('z/lc')
        plt.show()

    def trac_partical_xz(self, z):
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(z.T[:, 3], z.T[:, 5])
        ax.set_xlabel('x/lc')
        ax.set_ylabel('y/lc')
        plt.show()

    def trac_partical_3d(self, z):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.plot(z.T[:, 3], z.T[:, 4], z.T[:, 5])
        ax.set_xlabel('x/lc')
        ax.set_ylabel('y/lc')
        ax.set_zlabel('z/lc')
        plt.title('trac partical 3d')
        plt.show()

    def time_graph_partical(self, z, steps, t_final):
        t = np.linspace(0, t_final, steps)
        plt.plot(t, z.T[:, 3:6])
        plt.legend(['x/lc', 'y/lc', 'z/lc'])
        plt.xlabel('t/tc')
        plt.ylabel('coords/lc')
        plt.title('time_graph')
        plt.show()

    def time_graph_momentum(self, z, steps, t_final):
        t = np.linspace(0, t_final, steps)
        plt.plot(t, z.T[:, 0:3])
        plt.legend(['x/lc', 'y/lc', 'z/lc'])
        plt.xlabel('t/tc')
        plt.ylabel('coords/lc')
        plt.title('time_graph')
        plt.show()


phys = Phys()
lx = 10
ly = 10
lz = 10
nx = 5
ny = 5
nz = 5
steps = 1000
t_final = 1
params_electro = [1., 0., 0.]
params_magnetic = [7., 0., 0.]
grid = Grid(lx, ly, lz, nx, ny, nz, steps, t_final)
a = ElectromagneticField(grid)
a.electro_component_x(grid.x, grid.y, grid.z, params_electro[0])
a.electro_component_y(grid.x, grid.y, grid.z, params_electro[1])
a.electro_component_z(grid.x, grid.y, grid.z, params_electro[2])
a.magnetic_component_x(grid.x, grid.y, grid.z, params_magnetic[0])
a.magnetic_component_y(grid.x, grid.y, grid.z, params_magnetic[1])
a.magnetic_component_z(grid.x, grid.y, grid.z, params_magnetic[2])
a.electrofield_3d()
a.magneticfield_3d()
a.field_visualization_xy()
point = Particle(grid)
res = point.solver(a, params_electro, params_magnetic)
t = np.linspace(0, 1, 1000)
z = res.sol(t)
#point.trac_partical_xy(z)
#point.trac_partical_yz(z)
#point.trac_partical_xz(z)
#point.trac_partical_3d(z)
#point.time_graph_partical(z, steps, t_final)
#point.time_graph_momentum(z, steps, t_final)
'''
    def system_equation(self, t, z,  phys_class, field_class):
        vector, momentum = z
        field_H = np.array(field_class.magnetic_field_x, field_class.magnetic_field_y, field_class.magnetic_field_z)
        field_E = np.array(field_class.electro_field_x, field_class.electro_field_y, field_class.electro_field_z)
        return [phys_class.charge*(field_E + 1/(phys_class.mass * phys_class.speed_light) * np.cross(momentum, field_H)), momentum/phys_class.mass]
'''
