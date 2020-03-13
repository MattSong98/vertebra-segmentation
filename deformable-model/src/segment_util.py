import numpy as np
from mesh_util import elementwise_normalize, face_normal, face_center, vvLUT
from procrustes_analysis import weop
from interpolation.splines import LinearSpline
from sklearn.externals import joblib
from scipy.sparse.linalg import lsmr, cg
from scipy.sparse import lil_matrix


class VertebraSegmentation:
    def __init__(self, img, spacing):
        self.img = img
        self.spacing = spacing
        g1, g2, g3 = np.gradient(img)
        canny_img = self.canny(img)
        self.intensity = self.build_linear_interp_func(img)
        self.g1 = self.build_linear_interp_func(g1)
        self.g2 = self.build_linear_interp_func(g2)
        self.g3 = self.build_linear_interp_func(g3)
        self.C = self.build_linear_interp_func(canny_img)
        self.R = None
        self.vvlut = None
        self.verts_comp = None
        self.verts_mean = None
        self.weights = None
        self.faces = None
        self.delta = 0.3333333
        self.decay = 0.01
        self.gmax = 0.3
        self.beta = 3e-2
        print('initialization ok.')

    def __call__(self, *args, **kwargs):
        return self.segment(*args, **kwargs)


    def mesh_init(self, mesh, qth):
        """
        Rfun: a random forest regressor
        v{}_comp: v{}的10个sample进行GPA后的verts
        v{}_mean: V{}的平均模型
        vvlut:

        """
        rfr = joblib.load('../data_others/Rfunc/v{}.pkl'.format(qth))
        self.R = rfr.predict
        self.vvlut = vvLUT(*mesh)
        self.verts_comp = joblib.load('../data_others/v{}_comp'.format(qth))
        self.verts_mean = joblib.load('../data_others/v{}_mean'.format(qth))[0]
        self.faces = mesh[1]

    def boundary_degree(self, pos, normal):
        gmax = self.gmax
        intensity = self.intensity(pos)
        g1 = self.g1(pos)
        g2 = self.g2(pos)
        g3 = self.g3(pos)
        grad = np.c_[g1, g2, g3]
        direct_cosine = np.sum(grad*normal, axis=1)
        R = self.R(np.c_[intensity, direct_cosine])
        C = self.C(pos)
        gw1 = (1 + C + R)[:, np.newaxis]*grad
        gw1_norm = np.sqrt(np.sum(gw1 ** 2, axis=1))
        gw2 = (1 + R)[:, np.newaxis]*grad

        values_left = gmax*(gmax + gw1_norm)/(gmax ** 2 + gw1_norm ** 2)
        values_right = np.sum((normal*gw2), axis=1)
        values = values_left*values_right
        return values*100  # scaling the result has no effect

    def boundary_detect(self, fpos, fnormal, search_radius):
        decay = self.decay
        delta = self.delta
        fpos = fpos/self.spacing
        fnormal = elementwise_normalize(fnormal)/self.spacing

        sample_parcel = np.array([delta*i for i in range(-search_radius, search_radius + 1)])
        values = np.zeros((fnormal.shape[0], search_radius*2 + 1))
        for i, length in enumerate(sample_parcel):
            new_fpos = fpos + length*fnormal
            values[:, i] = self.boundary_degree(new_fpos, fnormal) - decay*(length ** 2)

        shift = sample_parcel[values.argmax(axis=1)]
        # update weights for calculating external energy
        weights = values.max(axis=1)
        weights[weights < 0] = 0
        self.weights = weights
        new_fpos = fpos + fnormal*shift[:, np.newaxis]
        new_fpos *= self.spacing
        return new_fpos

    def build_constraint(self, verts):
        verts_mean = self.verts_mean
        s, R, t = weop(verts, verts_mean, 1)
        align_verts = s*verts.dot(R) + t.T
        components = self.verts_comp
        A = np.array([c.reshape(-1) for c in components]).T
        b = align_verts.reshape(-1)
        w = lsmr(A, b)[0]
        cons_verts = A.dot(w).reshape((-1, 3))
        cons_verts = (cons_verts - t.T).dot(R.T)/s
        return cons_verts

    def load_intrinsic_params(self, A, b, cons_verts):
        vvlut = self.vvlut
        for i in vvlut.keys():
            vi_ = cons_verts[i]
            xi, yi, zi = i*3, i*3 + 1, i*3 + 2
            for j in vvlut[i]:
                vj_ = cons_verts[j]
                xj, yj, zj = j*3, j*3 + 1, j*3 + 2
                diff = 2*(vi_ - vj_)
                A[xi, xi] += 1
                A[yi, yi] += 1
                A[zi, zi] += 1
                A[xj, xj] += 1
                A[yj, yj] += 1
                A[zj, zj] += 1
                A[xi, xj] -= 1
                A[xj, xi] -= 1
                A[yi, yj] -= 1
                A[yj, yi] -= 1
                A[zi, zj] -= 1
                A[zj, zi] -= 1
                b[xi] += diff[0]
                b[yi] += diff[1]
                b[zi] += diff[2]
                b[xj] -= diff[0]
                b[yj] -= diff[1]
                b[zj] -= diff[2]

    def load_extrinsic_params(self, A, b, fpos, fnormal):
        intensity = self.intensity(fpos)
        g1 = self.g1(fpos)
        g2 = self.g2(fpos)
        g3 = self.g3(fpos)
        grad = np.c_[g1, g2, g3]
        direct_cosine = np.sum(grad*fnormal, axis=1)
        R = self.R(np.c_[intensity, direct_cosine])
        C = self.C(fpos)
        gw1 = (1 + C + R)[:, np.newaxis]*grad
        gw1 = elementwise_normalize(gw1)
        gw1_2 = gw1 ** 2
        wgw1_2 = self.weights[:, np.newaxis]*gw1_2
        faces = self.faces

        for w, tridx, pos in zip(wgw1_2, faces, fpos):
            beta = self.beta
            xa, ya, za = 3*tridx[0], 3*tridx[0] + 1, 3*tridx[0] + 2
            xb, yb, zb = 3*tridx[1], 3*tridx[1] + 1, 3*tridx[1] + 2
            xc, yc, zc = 3*tridx[2], 3*tridx[2] + 1, 3*tridx[2] + 2
            wx, wy, wz = w*beta

            A[xa, xa] += wx
            A[xa, xb] += wx
            A[xa, xc] += wx
            A[xb, xa] += wx
            A[xb, xb] += wx
            A[xb, xc] += wx
            A[xc, xa] += wx
            A[xc, xb] += wx
            A[xc, xc] += wx

            A[ya, ya] += wy
            A[ya, yb] += wy
            A[ya, yc] += wy
            A[yb, ya] += wy
            A[yb, yb] += wy
            A[yb, yc] += wy
            A[yc, ya] += wy
            A[yc, yb] += wy
            A[yc, yc] += wy

            A[za, za] += wz
            A[za, zb] += wz
            A[za, zc] += wz
            A[zb, za] += wz
            A[zb, zb] += wz
            A[zb, zc] += wz
            A[zc, za] += wz
            A[zc, zb] += wz
            A[zc, zc] += wz

            x_, y_, z_ = pos*6
            b[xa] += x_*wx
            b[xb] += x_*wx
            b[xc] += x_*wx

            b[ya] += y_*wy
            b[yb] += y_*wy
            b[yc] += y_*wy

            b[za] += z_*wz
            b[zb] += z_*wz
            b[zc] += z_*wz

    def build_quadratic_system(self, cons_verts, new_fpos, fnormal):
        n = cons_verts.size
        A = lil_matrix((n, n))
        b = np.zeros(n)
        self.load_intrinsic_params(A, b, cons_verts)
        self.load_extrinsic_params(A, b, new_fpos, fnormal)
        b /= 2
        A = A.tocsr()

        return A, b

    @staticmethod
    def build_linear_interp_func(data):
        d0, d1, d2 = data.shape
        a = np.array([0, 0, 0])
        b = np.array([d0 - 1, d1 - 1, d2 - 1])
        orders = np.array([d0, d1, d2])
        lin = LinearSpline(a, b, orders, data)
        return lin

    @staticmethod
    def solve_quadratic_system(A, b):
        x, info = cg(A, b)
        if info == 0:  # successful exit
            return x.reshape((-1, 3))
        elif info > 0:  # convergence to tolerance not achieved, number of iterations
            raise Exception('convergence to tolerance not achieved, number of iterations', info)
        else:  # illegal input or breakdown
            raise Exception('illegal input or breakdown')

    @staticmethod
    def canny(img, sigma=2):
        from scipy.ndimage.filters import gaussian_filter
        g1, g2, g3 = np.gradient(gaussian_filter(img, sigma))
        edge = np.sqrt(g1 ** 2 + g2 ** 2 + g3 ** 2)
        emax, emin = edge.max(), edge.min()
        edge = (edge - emin)/(emax - emin)
        return edge

    def segment(self, mesh, qth, iteration=10, min_search_radius=1, max_search_radius=12):
        self.mesh_init(mesh, qth)

        search_radius_list = np.linspace(max_search_radius, min_search_radius, iteration).astype(int)
        for search_radius in search_radius_list:
            fpos = face_center(*mesh)
            fnormal = face_normal(*mesh)
            new_fpos = self.boundary_detect(fpos, fnormal, search_radius)
            cons_verts = self.build_constraint(mesh[0])
            A, b = self.build_quadratic_system(cons_verts, new_fpos, fnormal)
            new_verts = self.solve_quadratic_system(A, b)
            print((new_verts - cons_verts).std())
            mesh = (new_verts, mesh[1])

        return mesh

    def segment_test(self, mesh, qth, search_radius):
        self.mesh_init(mesh, qth)

        fpos = face_center(*mesh)
        fnormal = face_normal(*mesh)
        new_fpos = self.boundary_detect(fpos, fnormal, search_radius)
        cons_verts = self.build_constraint(mesh[0])
        A, b = self.build_quadratic_system(cons_verts, new_fpos, fnormal)
        new_verts = self.solve_quadratic_system(A, b)
        print((new_verts - mesh[0]).std())
        mesh = (new_verts, mesh[1])

        return mesh, cons_verts, new_fpos
