from escnn.gspaces import *
from escnn.nn.modules.equivariant_module import EquivariantModule
from escnn.nn import FieldType
from escnn.nn import GeometricTensor
from escnn.nn.modules.invariantmaps import GroupPooling

from escnn.nn.modules.equivariant_module import EquivariantModule
from escnn.nn.modules.utils import indexes_from_labels

import torch
from torch import nn

from typing import List, Tuple, Any
from collections import defaultdict
import numpy as np

from gtc.functional import get_cayley_table

from gtc.functional import (
    build_Fplus_vectorized,
    clebsch_gordan,
    first_last_cb,
    get_cayley_table,
)


class TCGroupPoolingEfficient(GroupPooling):

    def __init__(self, in_type, group, idx=None, **kwargs):
        super().__init__(in_type, **kwargs)
        self.idx = idx
        self.group = group()
        self.cayley_table = get_cayley_table(self.group)

    def triple_correlation(self, x):
        b, k, d = x.shape
        x = x.reshape((b * k, d))
        nexts = x[:, self.cayley_table]
        mult = x.unsqueeze(1) * x[:, self.cayley_table.swapaxes(0, 1)]
        TC = torch.bmm(mult, nexts)
        TC = TC.reshape((b, k, d, d))
        return TC

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Apply Group Pooling to the input feature map.

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """
        assert input.type == self.in_type

        coords = input.coords
        input = input.tensor
        b, c = input.shape[:2]
        spatial_shape = input.shape[2:]

        for s, contiguous in self._contiguous.items():

            in_indices = getattr(self, "in_indices_{}".format(s))
            out_indices = getattr(self, "out_indices_{}".format(s))

            if contiguous:
                fm = input[:, in_indices[0] : in_indices[1], ...]
            else:
                fm = input[:, in_indices, ...]

            # split the channel dimension in 2 dimensions, separating fields
            fm = fm.view(b, -1, s, *spatial_shape)

            output = self.triple_correlation(fm.squeeze())

            if self.idx is None:
                idx = torch.triu_indices(output.shape[2], output.shape[3])
            else:
                idx = self.idx

            output = output[:, :, idx[0], idx[1]]
            a, b, c = output.shape
            output = output.reshape((a * b, c))
            output = output / (output.norm(dim=0, keepdim=True) + 1e-5)
            output = output.reshape((a, b, c, 1, 1))
        return output

    def export(self):
        raise NotImplementedError


class TCGroupPooling(GroupPooling):

    def __init__(self, in_type, group_type="cyclic", idx=None, **kwargs):
        """
        group_type should be "cyclic" or "dihedral"
        """
        super().__init__(in_type, **kwargs)
        self.idx = idx
        self.group_type = group_type

    def triple_correlation_vectorized_batch_cyclic(self, x):
        b, k, d = x.shape
        x = x.reshape((b * k, d))
        all_rolls = torch.zeros((b * k, d, d)).to(x.device)
        for i in range(d):
            all_rolls[:, :, i] = torch.roll(x, -i, dims=-1)
        rolls_mult = x.unsqueeze(1) * all_rolls
        TC = torch.bmm(rolls_mult, all_rolls)
        TC = TC.reshape((b, k, d, d))
        return TC

    def triple_correlation_vectorized_batch_dihedral(self, x):
        b, k, d = x.shape
        n = d // 2
        x = x.reshape((b * k, d))
        all_rolls = torch.zeros((b * k, d, d)).to(x.device)
        for i in range(d):
            roll0 = torch.roll(x[:, :n], -i, dims=-1)
            roll1 = torch.roll(x[:, n:], -i, dims=-1)
            all_rolls[:, :, i] = torch.hstack([roll0, roll1])
        rolls_mult = x.unsqueeze(1) * all_rolls
        TC = torch.bmm(rolls_mult, all_rolls)
        TC = TC.reshape((b, k, d, d))
        return TC

    def triple_correlation_vectorized_batch_r2(self, x):
        b, k, h, w = x.shape
        x = x.reshape((b * k, h * w))
        all_rolls = torch.zeros((b * k, h * w, h * w)).to(x.device)
        for i in range(h * w):
            all_rolls[:, :, i] = torch.roll(x, -i, dims=-1)
        rolls_mult = x.unsqueeze(1) * all_rolls
        TC = torch.bmm(rolls_mult, all_rolls)
        TC = TC.reshape((b, k, h * w, h * w))
        return TC

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Apply Group Pooling to the input feature map.

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        coords = input.coords
        input = input.tensor
        b, c = input.shape[:2]
        spatial_shape = input.shape[2:]

        for s, contiguous in self._contiguous.items():

            in_indices = getattr(self, "in_indices_{}".format(s))
            out_indices = getattr(self, "out_indices_{}".format(s))

            if contiguous:
                fm = input[:, in_indices[0] : in_indices[1], ...]
            else:
                fm = input[:, in_indices, ...]

            # split the channel dimension in 2 dimensions, separating fields
            fm = fm.view(b, -1, s, *spatial_shape)

            if self.group_type == "cyclic":
                output = self.triple_correlation_vectorized_batch_cyclic(fm.squeeze())

            elif self.group_type == "dihedral":
                output = self.triple_correlation_vectorized_batch_dihedral(fm.squeeze())

            if self.idx is None:
                idx = torch.triu_indices(output.shape[2], output.shape[3])
            else:
                idx = self.idx

            output = output[:, :, idx[0], idx[1]]
            a, b, c = output.shape
            output = output.reshape((a * b, c))
            output = output / (output.norm(dim=0, keepdim=True) + 1e-5)
            output = output.reshape((a, b, c, 1, 1))
        return output

    def export(self):
        raise NotImplementedError


class TCGroupPoolingR2Spatial(torch.nn.Module):

    def __init__(self, idx=None, **kwargs):
        super().__init__(**kwargs)
        self.idx = idx

    def triple_correlation_vectorized_batch(self, x):
        b, k, n, n = x.shape
        d = n * n
        x = x.reshape((b * k, d))
        all_rolls = torch.zeros((b * k, d, d)).to(x.device)
        for i in range(d):
            all_rolls[:, :, i] = torch.roll(x, -i, dims=-1)
        rolls_mult = x.unsqueeze(1) * all_rolls
        TC = torch.bmm(rolls_mult, all_rolls)
        TC = TC.reshape((b, k, d, d))
        return TC

    def forward(self, x):

        output = self.triple_correlation_vectorized_batch(x.squeeze())
        return output

        if self.idx is None:
            idx = torch.triu_indices(output.shape[2], output.shape[3])
        else:
            idx = self.idx

        output = output[:, :, idx[0], idx[1]]
        a, b, c = output.shape
        output = output.reshape((a * b, c))
        output = output / (output.norm(dim=0, keepdim=True) + 1e-5)
        output = output.reshape((a, b, c, 1, 1))

        return output

    def export(self):
        raise NotImplementedError


class BspGroupPooling(GroupPooling):
    def __init__(self, in_type, group_type="cyclic", idx=None, n=8, **kwargs):
        """
        group_type should be "cyclic" or "dihedral" or "octahedral"

        Parameters
        ----------
        in_type  type of geometric tensor

        """
        if n < 8:
            raise ValueError("n should be greater than or equal to 8")
        super().__init__(in_type, **kwargs)
        self.idx = idx
        self.group_type = group_type
        n2 = int(np.floor((n - 1) / 2))
        n3 = n2
        if n % 2 > 0:
            n3 = n2 - 1
        self.indices = torch.zeros(2, n3, dtype=int, requires_grad=False)
        self.CBmatrices = torch.zeros(n3, 4, 4, requires_grad=False)
        self.CBmatrices[0, ...], self.indices[:, 0] = first_last_cb(n, end=False)
        for i in range(2, n2):
            self.CBmatrices[i - 1, ...], self.indices[:, i - 1] = clebsch_gordan(
                1, i, n
            )
        if n % 2 == 0:
            self.CBmatrices[n2 - 1, ...], self.indices[:, n2 - 1] = first_last_cb(
                n, end=True
            )

    def fourier_transform_vectorized_batch_cyclic(self, f):
        """
        Computes the 1D DFT using the FFT algorithm.
        Input: signal f:Z/nZ->C.
        Return: DFT of f, fhat:Z/nZ->C.
        """
        device = f.device
        return torch.fft.fft(f, dim=2).to(device)

    def fourier_transform_vectorized_batch_dihedral(self, f):
        """
        Input: A function f:G->C where G is the dihedral group D_n (symmetries of the n-gon).
        G = {e, a, a^2...,a^{n-1}, x, ax, a^2x,...,a^{n-1}x}.
        Output: returns the Fourier transform of f on G.

        Parameters
        ----------
        f : array-like, shape=[..., k_filters, group_size]

        Returns
        -------
        fhat : array-like, shape=[..., k_filters, 2, 2, n_irreps]
        """
        device = f.device
        n = int(f.shape[2] / 2)
        n2d = int(np.floor((n - 1) / 2))  # number of 2D irreps
        fhat = torch.zeros(f.shape[0], f.shape[1], 2, 2, n2d + 1).to(device)
        # the coeffs for the 1d irreps are stored in fhat[..., 0]
        # the coeffs for the 2d irreps are stored in fhat[..., 1:n2d+1 (included)]
        fhat[:, :, 0, 0, 0] = f.sum(axis=2)
        fhat[:, :, 1, 0, 0] = f[:, :, :n].sum(axis=2) - f[:, :, n:].sum(axis=2)
        if n % 2 == 0:
            fhat[:, :, 0, 1, 0] = f[:, :, 0 : 2 * n : 2].sum(axis=2) - f[
                :, :, 1 : 2 * n : 2
            ].sum(axis=2)
            fhat[:, :, 1, 1, 0] = (
                f[:, :, 0:n:2].sum(axis=2)
                - f[:, :, 1:n:2].sum(axis=2)
                - f[:, :, n : 2 * n : 2].sum(axis=2)
                + f[:, :, n + 1 : 2 * n : 2].sum(axis=2)
            )
        i_range = torch.arange(1, n2d + 1)
        j_range = torch.arange(n)
        # Create all angular shifts combinations
        omega = 2 * torch.pi * i_range[:, None] * j_range / n
        # Create rho tensor of rotation matrices.
        rho = torch.concat(
            (torch.cos(omega), torch.sin(omega), -torch.sin(omega), torch.cos(omega))
        )
        rho = einops.rearrange(rho, "(c1 c2 w) h  -> c1 c2 w h", c1=2, c2=2).to(device)
        rho1 = rho.clone()
        rho1[:, 1] *= -1
        # Computes 2x2 Fourier coefficients
        fhat[..., 1 : n2d + 1] = torch.sum(
            f[:, :, None, None, None, j_range] * rho[None, None, :, :, :], dim=5
        )
        fhat[..., 1 : n2d + 1] += torch.sum(
            f[:, :, None, None, None, j_range + n] * rho1[None, None, :, :, :], dim=5
        )
        return fhat

    def fourier_coef(self, f, irrep, group):
        """Compute Fourier coef of signal f at an irrep of the group."""
        device = f.device
        return sum(
            [
                torch.einsum("bf,de->bfde", f[:, :, i_g], torch.tensor(irrep(g))).to(
                    device
                )
                for i_g, g in enumerate(group.elements)
            ]
        )

    def fourier_coef_tensor(self, f, irrep, irrep_prime, group):
        """Compute the "Fourier" coefficient F(\Theta)_{\rho \otimes \rho'}.

        "Fourier" is in quote, because it cannot really be called Fourier since \rho \otimes \rho'
        is not necessarily irreducible."""
        device = f.device
        tensor_rep = [np.kron(irrep(g), irrep_prime(g)) for g in group.elements]
        return sum(
            [
                torch.einsum(
                    "bf,de->bfde",
                    f[:, :, i_g],
                    torch.tensor(tensor_rep[i_g]).to(device),
                )
                for i_g in range(len(group.elements))
            ]
        )

    def fourier_transform_vectorized_batch_octahedral(self, f):
        """
        Input: A function f:G->C where G is the octahedral group.
        Output: returns the Fourier transform of f on G.

        The octahedral group has 24 elements, thus group_size = 24.
        The octahedral group has 5 irreps.

        The irreps of the octahedral group have maximum dimension 3, which is why the
        output is of shape [..., k_filters, 3, 3, n_irreps].

        Parameters
        ----------
        f : array-like, shape=[..., k_filters, group_size]

        Returns
        -------
        fhat : array-like, shape=[..., k_filters, 3, 3, n_irreps]
        """
        n_irreps = 5
        device = f.device
        batch_size, k_filters, group_size = f.shape
        fhat = torch.zeros(batch_size, k_filters, 3, 3, n_irreps).to(device)
        group = OctahedralGroup()

        for i_irrep, irrep in enumerate(group.irreps()):
            # This assumes that the group elements in f are ordered in the same order a group.elements
            fourier_coef = self.fourier_coef(self, f, irrep, group)
            fhat[:, :, : irrep.size, :: irrep.size, i_irrep] = fourier_coef.to(device)

        return fhat

    def bispectrum_selective_vectorized_batch_cyclic(self, x):
        r"""
        Compute the selective bispectrum beta using 1d DFT.
        Input: The 1d Fourier transform on Z/nZ.
        Returns: Only the |G| bispectrum elements needed for completeness.
        These are given by beta[0,0], beta[0, 1] and beta[1, i-1] for i \in {1,2,...,n-2}
        """
        device = x.device
        fhat = self.fourier_transform_vectorized_batch_cyclic(x)
        b, c, n = fhat.shape
        beta = torch.zeros(fhat.shape) * 1j
        beta = beta.to(device)
        beta[..., 0] = (
            fhat[..., 0] * fhat[..., 0] * torch.conj(fhat[..., 0])
        )  # beta[0, 0]
        beta[..., 1] = (
            fhat[..., 0] * fhat[..., 1] * torch.conj(fhat[..., 1])
        )  # beta[0, 1]
        beta[..., 2:] = (
            fhat[..., 1].unsqueeze(2)
            * fhat[..., 1 : n - 1]
            * torch.conj(fhat[..., 2:n])
        )
        betareal = torch.zeros(b, c, 2 * n).to(device)
        betareal[..., 0 : 2 * n : 2] = beta.real
        betareal[..., 1 : 2 * n : 2] = beta.imag
        return betareal

    def bispectrum_full_vectorized_batch_cyclic(self, x):
        r"""
        Compute the full bispectrum beta using 1d DFT (lower triangular elements because of symmetry).
        Input: The 1d Fourier transform on Z/nZ.
        Returns: Only the |G| bispectrum elements needed for completeness.
        These are given by beta[0,0], beta[0, 1] and beta[1, i-1] for i \in {1,2,...,n-2}
        """
        device = x.device
        fhat = self.fourier_transform_vectorized_batch_cyclic(x)
        b, c, n = fhat.shape
        betareal = torch.zeros(b, c, n * (n + 1))
        betareal = betareal.to(device)
        count = 0
        irange = torch.arange(n)
        for j in range(n):
            temp = (
                fhat[..., irange[j:]]
                * fhat[..., j].unsqueeze(2)
                * torch.conj(fhat[..., (irange[j:] + j) % n])
            )
            endreal = count + n - j
            betareal[..., count:endreal] = temp.real
            endimag = endreal + n - j
            betareal[..., endreal:endimag] = temp.imag
            count = endimag
        return betareal

    def bispectrum_vectorized_batch_dihedral(self, x, n):
        """
        Input: Fourier transform over D_n
        Output:  the bispectral elements needed for completeness

        Parameters
        ----------
        x : array-like, shape=[..., k_filters, group_size]
            Input: signal over the group Dn.
        n : int
            The group size of Dn is 2n+1

        Returns
        -------
        _ : array-like, shape=[..., k_filters, n_bispectrum_scalar_elements]
        """
        device = x.device
        fhat = self.fourier_transform_vectorized_batch_dihedral(x)

        bs, cs = fhat.shape[:2]
        # computes beta_\rho_0,\rho_0
        beta0 = fhat[:, :, 0, 0, 0] ** 3
        # computes beta_\rho_1,\rho_1
        beta10 = torch.sum(
            fhat[:, :, None, None, None, 0, 0, 0]
            * fhat[:, :, :, None, :, 1]
            * fhat[:, :, None, :, :, 1],
            axis=4,
        ).squeeze()

        n2 = int(np.floor((n - 1) / 2))
        n3 = n2
        if n % 2 > 0:
            n3 = n2 - 1
        beta1i = torch.zeros(bs, cs, n3, 4, 4).to(device)
        # indices = np.zeros(2, dtype = int)

        CBmatrix, indices = (
            self.CBmatrices[0, ...].to(device),
            self.indices[:, 0].to(device),
        )  # first_last_cb(n, end=False)
        Fplus = build_Fplus_vectorized(indices, fhat, n, end=False).to(device)
        # beta = (fhat \otimes fhat) * C * F.T * C.T = (fhat \otimes fhat) * C * (C * F).T
        fh_kron_fh = (
            fhat[:, :, :, None, :, None, 1] * fhat[:, :, None, :, None, :, 1]
        ).reshape((bs, cs, 4, 4))
        fh_kron_fh_C = torch.sum(
            fh_kron_fh[:, :, :, :, None] * CBmatrix[None, None, None, :, :], axis=3
        )
        C_Fplus = torch.sum(
            CBmatrix[None, None, :, :, None] * Fplus[:, :, None, :, :], axis=3
        )
        beta1i[:, :, 0, :, :] = torch.sum(
            fh_kron_fh_C[:, :, :, None, :] * C_Fplus[:, :, None, :, :], axis=4
        )
        Fplus = torch.zeros(bs, cs, 4, 4).to(device)
        # CBmatrix = CBmatrix.clone()

        for i in range(2, n2):
            CBmatrix, indices = (
                self.CBmatrices[i - 1, ...].to(device),
                self.indices[:, i - 1].to(device),
            )
            Fplus[..., :2, :2] = fhat[..., indices[0]]
            Fplus[..., 2:, 2:] = fhat[..., indices[1]]
            fh_kron_fh = (
                fhat[:, :, :, None, :, None, 1] * fhat[:, :, None, :, None, :, i]
            ).reshape((bs, cs, 4, 4))
            fh_kron_fh_C = torch.sum(
                fh_kron_fh[:, :, :, :, None] * CBmatrix[None, None, None, :, :], axis=3
            )
            C_Fplus = torch.sum(
                CBmatrix[None, None, :, :, None] * Fplus[:, :, None, :, :], axis=3
            )
            beta1i[:, :, i - 1, :, :] = torch.sum(
                fh_kron_fh_C[:, :, :, None, :] * C_Fplus[:, :, None, :, :], axis=4
            )

        # Fplus = torch.zeros(bs, cs, 4, 4).to(device)
        # CBmatrix = CBmatrix.clone().to(device)
        if n % 2 == 0:
            CBmatrix, indices = (
                self.CBmatrices[n2 - 1, ...].to(device),
                self.indices[:, n2 - 1].to(device),
            )  # first_last_cb(n, end=True)
            Fplus = build_Fplus_vectorized(indices, fhat, n, end=True).to(device)
            fh_kron_fh = (
                fhat[:, :, :, None, :, None, 1] * fhat[:, :, None, :, None, :, n2]
            ).reshape((bs, cs, 4, 4))
            fh_kron_fh_C = torch.sum(
                fh_kron_fh[:, :, :, :, None] * CBmatrix[None, None, None, :, :], axis=3
            )
            C_Fplus = torch.sum(
                CBmatrix[None, None, :, :, None] * Fplus[:, :, None, :, :], axis=3
            )
            beta1i[:, :, n2 - 1, :, :] = torch.sum(
                fh_kron_fh_C[:, :, :, None, :] * C_Fplus[:, :, None, :, :], axis=4
            )

        beta0 = beta0.unsqueeze(2)
        bs, cs, a, b = beta10.shape
        beta10 = beta10.reshape((bs, cs, a * b))
        bs, cs, a, b, c = beta1i.shape
        beta1i = beta1i.reshape((bs, cs, a * b * c))
        return torch.cat((beta0, beta10, beta1i), dim=2)

    def bispectrum_vectorized_batch_octahedral(self, x):
        """
        Input: signal over Oh
        Output:  the bispectral elements needed for completeness

        Parameters
        ----------
        x : array-like, shape=[..., k_filters, group_size]
            Input: signal over the group Oh.

        Returns
        -------
        _ : array-like, shape=[..., k_filters, n_bispectrum_scalar_elements]
        """
        device = x.device
        group = OctahedralGroup()
        fhat = self.fourier_transform_vectorized_batch_octahedral(x)

        batch_size, k_filters, _, _, n_irreps = fhat.shape

        # We only need b00, b10, b11 and b12 which we compute below.

        # beta00 - dim 0
        irrep_0 = group.irreps()[0]
        fourier_coef_0 = self.fourier_coef(x, irrep_0, group)  # shape [b, k, 1, 1]

        # TODO: make into batch computation
        beta00 = abs(fourier_coef_0) ** 2 * fourier_coef_0  # shape is  [bs, k, 1, 1]

        # beta10 - dim 3
        irrep_1 = group.irreps()[1]
        fourier_coef_1 = self.fourier_coef(x, irrep_1, group)  # shape [bs, k, 3, 3]

        # TODO: make into batch computation
        aux_matmul = torch.einsum(
            "bkde,bkef->bkdf", fourier_coef_1.conj().transpose(2, 3), fourier_coef_1
        )
        beta10 = torch.einsum(
            "bk,bkde->bkde", fourier_coef_0.conj().squeeze(), aux_matmul
        )
        beta10 = beta10.real  # shape is [bs, k, 3, 3]

        # beta11  - dim 9 (3*3)
        beta11 = np.kron(
            fourier_coef_1, fourier_coef_1
        ).conj().T @ self.fourier_coef_tensor(
            irrep_1, irrep_1, group
        )  # shape [bs, k, 9, 9]

        # beta12 - dim 9 (3*3)
        irrep_2 = group.irreps()[2]
        fourier_coef_2 = self.fourier_coef(x, irrep_2, group)

        fourier_coef_tensor_12 = self.fourier_coef_tensor(x, irrep_1, irrep_2, group)
        # shape is [bs, k, 9, 9]

        aux = torch.zeros((batch_size, k_filters, 9, 9), dtype=torch.double)
        for b in range(batch_size):
            for k in range(k_filters):
                aux_kron = torch.kron(
                    fourier_coef_1[b, k, :, :], fourier_coef_2[b, k, :, :]
                )
                aux[b, k, :, :] = aux_kron.conj().T

        beta12 = torch.einsum(
            "bkde, bkef->bkdf", aux, fourier_coef_tensor_12
        )  # shape is [bs, k, 9, 9]

        beta00 = beta00.reshape((batch_size, k_filters, -1))
        beta10 = beta10.reshape((batch_size, k_filters, -1))
        beta11 = beta11.reshape((batch_size, k_filters, -1))
        beta12 = beta12.reshape((batch_size, k_filters, -1))

        return torch.cat((beta00, beta10, beta11, beta12), dim=2)

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        """

        Apply Group Pooling to the input feature map.

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        coords = input.coords
        input = input.tensor
        b, c = input.shape[:2]  # b = batch size, c = channel size
        spatial_shape = input.shape[2:]

        for s, contiguous in self._contiguous.items():
            in_indices = getattr(self, "in_indices_{}".format(s))  # self.in_indices_0
            out_indices = getattr(self, "out_indices_{}".format(s))

            if contiguous:
                fm = input[:, in_indices[0] : in_indices[1], ...]
            else:
                fm = input[:, in_indices, ...]

            # split the channel dimension in 2 dimensions, separating fields
            fm = fm.view(b, -1, s, *spatial_shape)
            # fm is feature map
            bm, cm = fm.shape[:2]
            if self.group_type == "cyclic":
                n = fm.shape[2]
                # output = self.bispectrum_full_vectorized_batch_cyclic(fm.squeeze())
                output = self.bispectrum_selective_vectorized_batch_cyclic(fm.squeeze())
            elif self.group_type == "product_cyclic":
                output = self.bispectrum_product_cyclic(fm.squeeze())
            elif self.group_type == "dihedral":
                n = int(fm.shape[2] / 2)
                output = self.bispectrum_vectorized_batch_dihedral(fm.squeeze(), n)
            elif self.group_type == "octahedral":
                output = self.bispectrum_vectorized_batch_octahedral(fm.squeeze())

            """
            if self.idx is None:
                idx = torch.triu_indices(output.shape[2], output.shape[3])
            else:
                idx = self.idx
            """
            # output = output[:, :, idx[0], idx[1]]
            a, b, c = output.shape
            output = output.reshape((a * b, c))
            output = output / (output.norm(dim=0, keepdim=True) + 1e-5)
            output = output.reshape((a, b, c, 1, 1))
            # print(b * c)
        return output

    def export(self):
        raise NotImplementedError
