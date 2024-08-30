import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CIConv2d(nn.Module):
    def gaussian_basis_filters(self, scale, gpu, k=3):
        std = torch.pow(2, scale)

        # Define the basis vector for the current scale
        filtersize = torch.ceil(k * std + 0.5)
        x = torch.arange(start=-filtersize.item(), end=filtersize.item() + 1)
        if gpu is not None: x = x.cuda(gpu); std = std.cuda(gpu)
        x = torch.meshgrid([x, x])

        # Calculate Gaussian filter base
        # Only exponent part of Gaussian function since it is normalized anyway
        g = torch.exp(-(x[0] / std) ** 2 / 2) * torch.exp(-(x[1] / std) ** 2 / 2)
        g = g / torch.sum(g)  # Normalize

        # Gaussian derivative dg/dx filter base
        dgdx = -x[0] / (std ** 3 * 2 * math.pi) * torch.exp(-(x[0] / std) ** 2 / 2) * torch.exp(
            -(x[1] / std) ** 2 / 2)
        dgdx = dgdx / torch.sum(torch.abs(dgdx))  # Normalize

        # Gaussian derivative dg/dy filter base
        dgdy = -x[1] / (std ** 3 * 2 * math.pi) * torch.exp(-(x[1] / std) ** 2 / 2) * torch.exp(
            -(x[0] / std) ** 2 / 2)
        dgdy = dgdy / torch.sum(torch.abs(dgdy))  # Normalize

        # Stack and expand dim
        basis_filter = torch.stack([g, dgdx, dgdy], dim=0)[:, None, :, :]

        return basis_filter

    def E_inv(self, E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
        E = Ex ** 2 + Ey ** 2 + Elx ** 2 + Ely ** 2 + Ellx ** 2 + Elly ** 2
        return E

    def gray_raw_pow_dim1(self, E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
        E = Ex ** 2 + Ey ** 2 + Elx ** 2 + Ely ** 2 + Ellx ** 2 + Elly ** 2
        E = F.instance_norm(torch.log(E + self.eps))
        return E

    def gray_raw_rand_pow_dim1(self, E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
        w = torch.rand(3).cuda()
        w = w / torch.sum(w) * 3
        E = (Ex ** 2 + Ey ** 2) * w[0] + (Elx ** 2 + Ely ** 2) * w[1]  + (Ellx ** 2  +  Elly ** 2) * w[2]
        E = F.instance_norm(torch.log(E + self.eps))
        return E

    def gray_raw_abs_dim1(self, E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
        E = torch.abs(Ex) + torch.abs(Ey) + torch.abs(Elx) + torch.abs(Ely) + torch.abs(Ellx) + torch.abs(Elly)
        E = F.instance_norm(torch.log(E + self.eps))
        return E

    def gray_inv(self, R, Rx, Ry, G, Gx, Gy, B, Bx, By):
        eps = 1.0/255.0
        Rxn = Rx / (R + torch.abs(Rx) + eps)
        Gxn = Gx / (G + torch.abs(Gx) + eps)
        Bxn = Bx / (B + torch.abs(Bx) + eps)
        Ryn = Ry / (R + torch.abs(Ry) + eps)
        Gyn = Gy / (G + torch.abs(Gy) + eps)
        Byn = By / (B + torch.abs(By) + eps)
        out = torch.cat((Rxn, Gxn, Bxn, Ryn, Gyn, Byn), dim=1)
        return out

    def gray_M01_pow_dim1(self, R, Rx, Ry, G, Gx, Gy, B, Bx, By):
        M1x = (Rx * G - Gx * R) / (R * G + torch.abs(Rx * G - Gx * R) + 1e-5)
        M2x = (Gx * B - Bx * G) / (G * B + torch.abs(Gx * B - Bx * G) + 1e-5)
        M3x = (Bx * R - Rx * B) / (R * B + torch.abs(Bx * R - Rx * B) + 1e-5)
        M1y = (Ry * G - Gy * R) / (R * G + torch.abs(Ry * G - Gy * R) + 1e-5)
        M2y = (Gy * B - By * G) / (G * B + torch.abs(Gy * B - By * G) + 1e-5)
        M3y = (By * R - Ry * B) / (R * B + torch.abs(By * R - Ry * B) + 1e-5)
        out = M1x ** 2 + M2x ** 2 + M3x ** 2 + M1y ** 2 + M2y ** 2 + M3y ** 2
        out = F.instance_norm(torch.log(out + self.eps))
        return out

    def gray_inv_M(self, R, Rx, Ry, G, Gx, Gy, B, Bx, By):
        M1x = (Rx * G - Gx * R) / (R * G + 1e-5)
        M2x = (Gx * B - Bx * G) / (G * B + 1e-5)
        M3x = (Bx * R - Rx * B) / (R * B + 1e-5)
        M1y = (Ry * G - Gy * R) / (R * G + 1e-5)
        M2y = (Gy * B - By * G) / (G * B + 1e-5)
        M3y = (By * R - Ry * B) / (R * B + 1e-5)
        out = M1x ** 2 + M2x ** 2 + M3x ** 2 + M1y ** 2 + M2y ** 2 + M3y ** 2
        out = F.instance_norm(torch.log(out + self.eps))
        return out

    def gray_pow_dim1(self, R, Rx, Ry, G, Gx, Gy, B, Bx, By):
        eps = 1.0 / 255.0
        Rxn = Rx / (R + torch.abs(Rx) + eps)
        Gxn = Gx / (G + torch.abs(Gx) + eps)
        Bxn = Bx / (B + torch.abs(Bx) + eps)
        Ryn = Ry / (R + torch.abs(Ry) + eps)
        Gyn = Gy / (G + torch.abs(Gy) + eps)
        Byn = By / (B + torch.abs(By) + eps)
        out = Rxn ** 2 + Gxn ** 2 + Bxn ** 2 + Ryn ** 2 + Gyn ** 2 + Byn ** 2
        out = F.instance_norm(torch.log(out + self.eps))
        return out

    def gray_weighted_pow_dim1(self, R, Rx, Ry, G, Gx, Gy, B, Bx, By):
        eps = 1.0 / 255.0
        Rxn = Rx / (R + torch.abs(Rx) + eps) * self.coeff[0]
        Gxn = Gx / (G + torch.abs(Gx) + eps) * self.coeff[1]
        Bxn = Bx / (B + torch.abs(Bx) + eps) * self.coeff[2]
        Ryn = Ry / (R + torch.abs(Ry) + eps) * self.coeff[3]
        Gyn = Gy / (G + torch.abs(Gy) + eps) * self.coeff[4]
        Byn = By / (B + torch.abs(By) + eps) * self.coeff[5]
        out = Rxn ** 2 + Gxn ** 2 + Bxn ** 2 + Ryn ** 2 + Gyn ** 2 + Byn ** 2
        out = F.instance_norm(torch.log(out + self.eps))
        return out

    def gray_weighted2_pow_dim1(self, R, Rx, Ry, G, Gx, Gy, B, Bx, By):
        eps = 1.0 / 255.0
        Rxn = Rx / (R + torch.abs(Rx * self.coeff[0]) + eps)
        Gxn = Gx / (G + torch.abs(Gx * self.coeff[1]) + eps)
        Bxn = Bx / (B + torch.abs(Bx * self.coeff[2]) + eps)
        Ryn = Ry / (R + torch.abs(Ry * self.coeff[3]) + eps)
        Gyn = Gy / (G + torch.abs(Gy * self.coeff[4]) + eps)
        Byn = By / (B + torch.abs(By * self.coeff[5]) + eps)
        out = Rxn ** 2 + Gxn ** 2 + Bxn ** 2 + Ryn ** 2 + Gyn ** 2 + Byn ** 2
        out = F.instance_norm(torch.log(out + self.eps))
        return out

    def gray_weighted_raw_pow_dim1(self, R, Rx, Ry, G, Gx, Gy, B, Bx, By):
        eps = 1.0 / 255.0
        Rxn = Rx * self.coeff[0]
        Gxn = Gx * self.coeff[1]
        Bxn = Bx * self.coeff[2]
        Ryn = Ry * self.coeff[3]
        Gyn = Gy * self.coeff[4]
        Byn = By * self.coeff[5]
        out = Rxn ** 2 + Gxn ** 2 + Bxn ** 2 + Ryn ** 2 + Gyn ** 2 + Byn ** 2
        out = F.instance_norm(torch.log(out + self.eps))
        return out

    def gray_pow_nolog_dim1(self, R, Rx, Ry, G, Gx, Gy, B, Bx, By):
        eps = 1.0 / 255.0
        Rxn = Rx / (R + torch.abs(Rx) + eps)
        Gxn = Gx / (G + torch.abs(Gx) + eps)
        Bxn = Bx / (B + torch.abs(Bx) + eps)
        Ryn = Ry / (R + torch.abs(Ry) + eps)
        Gyn = Gy / (G + torch.abs(Gy) + eps)
        Byn = By / (B + torch.abs(By) + eps)
        out = Rxn ** 2 + Gxn ** 2 + Bxn ** 2 + Ryn ** 2 + Gyn ** 2 + Byn ** 2
        out = F.instance_norm(out)
        return out

    def gray_abs_dim1(self, R, Rx, Ry, G, Gx, Gy, B, Bx, By):
        eps = 1.0 / 255.0
        Rxn = Rx / (R + torch.abs(Rx) + eps)
        Gxn = Gx / (G + torch.abs(Gx) + eps)
        Bxn = Bx / (B + torch.abs(Bx) + eps)
        Ryn = Ry / (R + torch.abs(Ry) + eps)
        Gyn = Gy / (G + torch.abs(Gy) + eps)
        Byn = By / (B + torch.abs(By) + eps)
        out = torch.abs(Rxn) + torch.abs(Gxn) + torch.abs(Bxn) + torch.abs(Ryn) + torch.abs(Gyn) + torch.abs(Byn)
        out = F.instance_norm(torch.log(out + self.eps))
        return out

    def gray_abs_dim2(self, R, Rx, Ry, G, Gx, Gy, B, Bx, By):
        eps = 1.0/255.0
        Rxn = Rx / (R + torch.abs(Rx) + eps)
        Gxn = Gx / (G + torch.abs(Gx) + eps)
        Bxn = Bx / (B + torch.abs(Bx) + eps)
        Ryn = Ry / (R + torch.abs(Ry) + eps)
        Gyn = Gy / (G + torch.abs(Gy) + eps)
        Byn = By / (B + torch.abs(By) + eps)
        xn = (torch.abs(Rxn) + torch.abs(Gxn) + torch.abs(Bxn))/3
        yn = (torch.abs(Ryn) + torch.abs(Gyn) + torch.abs(Byn))/3
        out = torch.cat((xn, yn), dim=1)
        return out-0.5

    def gray_raw(self, R, Rx, Ry, G, Gx, Gy, B, Bx, By):
        # M1x = (Rx * G - Gx * R) / (R * G + 1e-5)
        # M2x = (Gx * B - Bx * G) / (G * B + 1e-5)
        # M3x = (Bx * R - Rx * B) / (R * B + 1e-5)
        # M1y = (Ry * G - Gy * R) / (R * G + 1e-5)
        # M2y = (Gy * B - By * G) / (G * B + 1e-5)
        # M3y = (By * R - Ry * B) / (R * B + 1e-5)
        # out = M1x ** 2 + M2x ** 2 + M3x ** 2 + M1y ** 2 + M2y ** 2 + M3y ** 2
        eps = 1.0/255.0
        Rxn = Rx #/ (R + torch.abs(Rx) + eps)
        Gxn = Gx #/ (G + torch.abs(Gx) + eps)
        Bxn = Bx #/ (B + torch.abs(Bx) + eps)
        Ryn = Ry #/ (R + torch.abs(Ry) + eps)
        Gyn = Gy #/ (G + torch.abs(Gy) + eps)
        Byn = By #/ (B + torch.abs(By) + eps)
        out = torch.cat((Rxn, Gxn, Bxn, Ryn, Gyn, Byn), dim=1)
        return out

    def gray_raw_abs_dim2(self, R, Rx, Ry, G, Gx, Gy, B, Bx, By):
        Rxn = Rx  # / (R + torch.abs(Rx) + eps)
        Gxn = Gx  # / (G + torch.abs(Gx) + eps)
        Bxn = Bx  # / (B + torch.abs(Bx) + eps)
        Ryn = Ry  # / (R + torch.abs(Ry) + eps)
        Gyn = Gy  # / (G + torch.abs(Gy) + eps)
        Byn = By  # / (B + torch.abs(By) + eps)
        xn = (torch.abs(Rxn) + torch.abs(Gxn) + torch.abs(Bxn)) / 3
        yn = (torch.abs(Ryn) + torch.abs(Gyn) + torch.abs(Byn)) / 3
        out = torch.cat((xn, yn), dim=1)
        return out-2.0

    def W_inv(self, E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
        eps = self.eps
        Wx = Ex / (E + eps)
        Wlx = Elx / (E + eps)
        Wllx = Ellx / (E + eps)
        Wy = Ey / (E + eps)
        Wly = Ely / (E + eps)
        Wlly = Elly / (E + eps)

        W = Wx ** 2 + Wy ** 2 + Wlx ** 2 + Wly ** 2 + Wllx ** 2 + Wlly ** 2
        return W

    def C_inv(self, E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
        Clx = (Elx * E - El * Ex) / (E ** 2 + 1e-5)
        Cly = (Ely * E - El * Ey) / (E ** 2 + 1e-5)
        Cllx = (Ellx * E - Ell * Ex) / (E ** 2 + 1e-5)
        Clly = (Elly * E - Ell * Ey) / (E ** 2 + 1e-5)

        C = Clx ** 2 + Cly ** 2 + Cllx ** 2 + Clly ** 2
        return C

    def N_inv(self, E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
        Nlx = (Elx * E - El * Ex) / (E ** 2 + 1e-5)
        Nly = (Ely * E - El * Ey) / (E ** 2 + 1e-5)
        Nllx = (Ellx * E ** 2 - Ell * Ex * E - 2 * Elx * El * E + 2 * El ** 2 * Ex) / (E ** 3 + 1e-5)
        Nlly = (Elly * E ** 2 - Ell * Ey * E - 2 * Ely * El * E + 2 * El ** 2 * Ey) / (E ** 3 + 1e-5)

        N = Nlx ** 2 + Nly ** 2 + Nllx ** 2 + Nlly ** 2
        return N

    def H_inv(self, E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
        Hx = (Ell * Elx - El * Ellx) / (El ** 2 + Ell ** 2 + 1e-5)
        Hy = (Ell * Ely - El * Elly) / (El ** 2 + Ell ** 2 + 1e-5)
        H = Hx ** 2 + Hy ** 2
        return H

    def __init__(self, invariant, k=3, scale=0.0):
        super(CIConv2d, self).__init__()
        # assert invariant in ['E', 'H', 'N', 'W', 'C', 'gray', 'gray_raw'], 'invalid invariant' # 'gray' is separating R G B channels
        self.eps = 1e-5
        self.invariant = invariant
        if 'gray' in invariant:
            if 'dim2' in invariant:
                self.out_dim = 2
            elif 'dim1' in invariant or invariant == 'gray_inv_M':
                self.out_dim = 1
            else:
                self.out_dim = 6
        else:
            self.out_dim = 1
        inv_switcher = {
            'E': self.E_inv,
            'W': self.W_inv,
            'C': self.C_inv,
            'N': self.N_inv,
            'H': self.H_inv,
            'gray': self.gray_inv, # output_dim 6
            'gray_inv_M': self.gray_inv_M, # output_dim 1
            'gray_M01_pow_dim1': self.gray_M01_pow_dim1, # output_dim 1
            'gray_raw_pow_dim1': self.gray_raw_pow_dim1, # output_dim 1
            'gray_raw_rand_pow_dim1': self.gray_raw_rand_pow_dim1, # output_dim 1
            'gray_learnable_raw_pow_dim1': self.gray_raw_pow_dim1, # output_dim 1
            'gray_pow_dim1': self.gray_pow_dim1, # output_dim 1
            'gray_learnable_pow_dim1': self.gray_pow_dim1, # output_dim 1 learnable
            'gray_learnable_weighted_pow_dim1': self.gray_weighted_pow_dim1, # output_dim 1 learnable
            'gray_learnable_weighted2_pow_dim1': self.gray_weighted2_pow_dim1, # output_dim 1 learnable
            'gray_learnable_raw_weighted_pow_dim1': self.gray_weighted_raw_pow_dim1, # output_dim 1 learnable
            'gray_pow_nolog_dim1': self.gray_pow_nolog_dim1, # output_dim 1
            'gray_raw': self.gray_raw, # output_dim 6
            'gray_abs_dim2':self.gray_abs_dim2, # output_dim 2
            'gray_abs_dim1':self.gray_abs_dim1, # output_dim 1
            'gray_raw_abs_dim2':self.gray_raw_abs_dim2, # output_dim 2
            'gray_raw_abs_dim1':self.gray_raw_abs_dim1, # output_dim 1
        }
        self.inv_function = inv_switcher[invariant]

        self.use_cuda = torch.cuda.is_available()
        self.gpu = torch.cuda.current_device() if self.use_cuda else None

        # Constants
        self.gcm = torch.tensor([[0.06, 0.63, 0.27], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]])
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]
        self.pixel_mean = torch.Tensor(pixel_mean).view(1, -1, 1, 1).cuda()
        self.pixel_std = torch.Tensor(pixel_std).view(1, -1, 1, 1).cuda()
        if self.use_cuda: self.gcm = self.gcm.cuda(self.gpu)
        self.k = k

        # Learnable parameters
        if 'learnable' in invariant: # use learnable filter
            self.scale = torch.tensor([scale])
            w = self.gaussian_basis_filters(scale=self.scale, gpu=self.gpu)  # KCHW
            self.filter_y_half = torch.nn.Parameter(w[0:1,:,0:4,:], requires_grad=True)
            self.filter_x_half = torch.nn.Parameter(w[1:2,:,:,0:4], requires_grad=True)
            self.coeff = torch.nn.Parameter(torch.ones(6).cuda(), requires_grad=True)
        else:
            self.scale = torch.nn.Parameter(torch.tensor([scale]), requires_grad=True)

    def forward(self, batch):
        eps = self.eps
        batch_origin = batch.clone()
        # batch = (batch - self.pixel_mean) / self.pixel_std
        batch = (batch - self.pixel_mean) / torch.sqrt(torch.pow(self.pixel_std, 2).mean(dim=1, keepdim=True)).expand_as(self.pixel_std)
        self.scale.data = torch.clamp(self.scale.data, min=-2.5, max=2.5)

        # Measure E, El, Ell by Gaussian color model
        in_shape = batch.shape  # bchw
        if 'gray' not in self.invariant:
            batch = batch.view((in_shape[:2] + (-1,)))  # flatten image
            batch = torch.matmul(self.gcm, batch)  # estimate E,El,Ell
            batch = batch.view((in_shape[0],) + (3,) + in_shape[2:])  # reshape to original image size
        E, El, Ell = torch.split(batch, 1, dim=1) # for gray, E=R, El=G, Ell=B
        R01, G01, B01 = torch.split(batch_origin, 1, dim=1)
        if 'learnable' in self.invariant:
            filter_x_zero = torch.zeros(self.filter_x_half.shape[0], 1, self.filter_x_half.shape[2], 1).cuda()
            filter_y_zero = torch.zeros(self.filter_y_half.shape[0], 1, 1, self.filter_y_half.shape[3]).cuda()
            filter_x = torch.cat(
                (-torch.abs(self.filter_x_half), filter_x_zero, torch.flip(torch.abs(self.filter_x_half), [3])), dim=3)
            filter_y = torch.cat(
                (-torch.abs(self.filter_y_half), filter_y_zero, torch.flip(torch.abs(self.filter_y_half), [2])), dim=2)
            filter_zero = torch.zeros_like(filter_x)
            w = torch.cat([filter_zero, filter_x, filter_y], dim=0)
        else:
            # Convolve with Gaussian filters
            w = self.gaussian_basis_filters(scale=self.scale, gpu=self.gpu)  # KCHW

        # the padding here works as "same" for odd kernel sizes
        E_out = F.conv2d(input=E, weight=w, padding=int(w.shape[2] / 2))
        El_out = F.conv2d(input=El, weight=w, padding=int(w.shape[2] / 2))
        Ell_out = F.conv2d(input=Ell, weight=w, padding=int(w.shape[2] / 2))
        E, Ex, Ey = torch.split(E_out, 1, dim=1)
        El, Elx, Ely = torch.split(El_out, 1, dim=1)
        Ell, Ellx, Elly = torch.split(Ell_out, 1, dim=1)
        if 'gray' in self.invariant:
            E = R01
            El = G01
            Ell = B01
        inv_out = self.inv_function(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly)
        # inv_out = F.instance_norm(torch.log(inv_out + eps))
        # inv_out = F.instance_norm(inv_out)
        return inv_out