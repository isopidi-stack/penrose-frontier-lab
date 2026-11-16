"""
Standalone Audit Suite for Penrose Frontier Lab (v0.1.0-penrose-frontier)
Verifies:
  1. Unitary non-commutative Googly patch coupling between left and right chiral bundles.
  2. Conformal Cartan tractor parallel transport norm preservation.
  3. Covariant constancy of the Infinity Twistor under vacuum Einstein transport.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from googly_transport_core import DualChiralTwistorBundle, compute_cartan_tractor_transport


def test_dual_chiral_googly_unitary_coupling():
    """Verifies that non-commutative Heisenberg patch coupling is unitary and preserves norms."""
    bundle = DualChiralTwistorBundle(num_nodes=16, epsilon_googly=0.08)
    for _ in range(10):
        res = bundle.apply_googly_patch_coupling()
        z_l, z_r, U = res["Z_left"], res["Z_right"], res["U_googly"]
        I5 = torch.eye(5, dtype=torch.complex128).unsqueeze(0).expand_as(U)
        assert torch.allclose(torch.matmul(U, U.conj().transpose(-1, -2)), I5, atol=1e-12)
        assert torch.allclose(torch.linalg.norm(z_l, dim=-1), torch.ones(16, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(torch.linalg.norm(z_r, dim=-1), torch.ones(16, dtype=torch.float64), atol=1e-12)


def test_cartan_tractor_transport_and_einstein_infinity_twistor():
    """
    Verifies local twistor transport norm preservation and proves that a generator
    commuting with the Infinity Twistor preserves I^{alpha beta} with zero defect
    (Penrose's local twistor criterion for the vacuum Einstein equations).
    """
    torch.manual_seed(42)
    z = torch.randn(8, 5, dtype=torch.complex128)
    z = z / torch.linalg.norm(z, dim=-1, keepdim=True)

    # Diagonal projector representing a conserved cosmological Infinity Twistor structure
    I_inf = torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.complex128))
    # Block-diagonal Weyl curvature generator that respects I_inf
    H_ein = torch.zeros(8, 5, 5, dtype=torch.complex128)
    H_ein[:, 0:2, 0:2] = torch.tensor([[0.2, 0.1j], [-0.1j, -0.2]], dtype=torch.complex128)
    H_ein[:, 2:4, 2:4] = torch.tensor([[0.15, -0.05j], [0.05j, -0.15]], dtype=torch.complex128)

    res = compute_cartan_tractor_transport(z, H_ein, infinity_twistor=I_inf)
    assert torch.allclose(torch.linalg.norm(res["Z_transported"], dim=-1), torch.ones(8, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(res["infinity_twistor_defect"], torch.zeros(8, dtype=torch.float64), atol=1e-12)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
