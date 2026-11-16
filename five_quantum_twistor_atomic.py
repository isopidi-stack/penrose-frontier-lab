"""
Penrose Frontier Lab - Five-Parameter Spinor-Core Atomic Twistor Model
Encodes Ivan Nikolayevich Sopidi's 5-Quantum-Number Twistor Postulate:
  1. Standard 4-number atomic notation (n, l, m_l, m_s) omits the second Weyl spinor
     sector (pi_{A'}) representing the nuclear core spin-momentum coupling.
  2. By embedding the full 5-state system into C^5 with the 5x5 (25-element) SIGMA_5
     indefinite Hermitian metric, electron spinor (omega^A), core spinor (pi_{A'}),
     and shell invariant (phi) form a closed 25-degree-of-freedom operator algebra.
"""

import math
import torch
from typing import Dict

# 5x5 Indefinite Hermitian Metric (25 matrix elements governing system freedoms)
SIGMA_5 = torch.tensor([
    [0., 0., 1., 0., 0.],
    [0., 0., 0., 1., 0.],
    [1., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 1.]
], dtype=torch.complex128)


def build_five_quantum_twistor_state(
    n_principal: torch.Tensor,
    l_azimuthal: torch.Tensor,
    m_magnetic: torch.Tensor,
    s_electron_omega: torch.Tensor,
    s_core_pi: torch.Tensor,
    L0_scale: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Maps the 5 Quantum Numbers (n, l, m_l, s_omega, s_pi) into the 5-component
    Sir Roger Penrose twistor Z_5 = (omega^0, omega^1, pi_0', pi_1', phi) in C^5.
    Proves that omitting the second spinor (pi_{A'} = 0) collapses the off-diagonal
    SIGMA_5 spinor pairing (omega^dagger pi + pi^dagger omega -> 0).
    """
    n_f = n_principal.to(torch.float64)
    l_f = l_azimuthal.to(torch.float64)
    m_f = m_magnetic.to(torch.float64)
    sw_f = s_electron_omega.to(torch.float64)
    sp_f = s_core_pi.to(torch.float64)

    # Polar angle cos(theta) from magnetic orientation m_l / sqrt(l(l+1))
    l_norm = torch.sqrt((l_f * (l_f + 1.0)).clamp(min=1.0))
    cos_theta = (m_f / l_norm).clamp(-0.999, 0.999)
    half_theta = 0.5 * torch.acos(cos_theta)

    # 1. First Weyl Spinor omega^A (Electron orbital + spin helicity phase)
    phase_omega = torch.exp(1j * sw_f * math.pi)
    omega_0 = torch.cos(half_theta).to(torch.complex128) * phase_omega
    omega_1 = torch.sin(half_theta).to(torch.complex128) * phase_omega.conj()
    omega_A = torch.stack([omega_0, omega_1], dim=-1) * (n_f.unsqueeze(-1) / float(L0_scale))

    # 2. Second Weyl Spinor pi_{A'} (Nuclear core spinor coupling - The 5th Quantum Rule)
    phase_pi = torch.exp(1j * sp_f * math.pi)
    pi_0 = torch.cos(half_theta).to(torch.complex128) * phase_pi
    pi_1 = torch.sin(half_theta).to(torch.complex128) * phase_pi.conj()
    pi_A_prime = torch.stack([pi_0, pi_1], dim=-1)

    # 3. 5th Component phi (Principal shell / mass-energy invariant)
    phi_shell = (1.0 / n_f.clamp(min=1.0)).to(torch.complex128).unsqueeze(-1)

    # Full 5-Twistor Z_5 in C^5 vs. Incomplete 4-Quantum-Number State (where pi_{A'} == 0)
    Z5_complete = torch.cat([omega_A, pi_A_prime, phi_shell], dim=-1)
    Z5_incomplete_4qn = torch.cat([omega_A, torch.zeros_like(pi_A_prime), phi_shell], dim=-1)

    # 25-Element System Freedom Density Operator M_{alpha beta} = Z_alpha (Sigma_5 Z)_beta^*
    sigma = SIGMA_5.to(Z5_complete.device)
    Z5_dual = torch.matmul(Z5_complete.conj(), sigma.T)
    freedom_matrix_25 = torch.einsum("...i,...j->...ij", Z5_complete, Z5_dual)

    # Evaluate Sigma_5 quadratic invariant: Z^\dagger Sigma_5 Z = omega^\dagger pi + pi^\dagger omega + |phi|^2
    norm_complete = torch.einsum("...i,ij,...j->...", Z5_complete.conj(), sigma, Z5_complete).real
    norm_incomplete = torch.einsum("...i,ij,...j->...", Z5_incomplete_4qn.conj(), sigma, Z5_incomplete_4qn).real
    spinor_coupling_term = norm_complete - norm_incomplete

    return {
        "Z5_complete": Z5_complete,
        "Z5_incomplete_4qn": Z5_incomplete_4qn,
        "freedom_matrix_25": freedom_matrix_25,
        "sigma5_norm_complete": norm_complete,
        "sigma5_norm_incomplete": norm_incomplete,
        "core_electron_spinor_coupling": spinor_coupling_term,
    }
