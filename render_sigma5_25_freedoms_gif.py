"""
Penrose Frontier Lab - 25-Element SIGMA_5 Freedom Matrix Animated Renderer
Generates sigma5_25_freedoms_proof.gif comparing:
  - Left 5x5 Panel: Complete 5-Quantum-Number Dual-Spinor State (omega^A, pi_{A'}, phi)
    exhibiting all 25 active matrix freedoms M^alpha_beta = Z^alpha (Sigma_5 Z)_beta^*.
  - Right 5x5 Panel: Incomplete 4-Quantum-Number State (pi_{A'} = 0), showing the
    collapse of the off-diagonal spinor coupling blocks.
Uses continuous erf() radial cell envelopes and complex phase coloring without Z-buffers.
"""

import os
import math
import torch
import numpy as np
from PIL import Image, ImageDraw
from five_quantum_twistor_atomic import SIGMA_5, build_five_quantum_twistor_state


def render_25_freedom_matrix_frame(
    frame_idx: int,
    total_frames: int = 36,
    height: int = 360,
    width: int = 680
) -> np.ndarray:
    """
    Renders a single side-by-side comparison frame of the 5x5 (25-element) SIGMA_5
    system freedom operator M^alpha_beta under continuous dual-spinor precession.
    """
    phase_t = 2.0 * math.pi * (frame_idx / float(total_frames))

    # Precessing 5-quantum-number parameters (n, l, m_l, s_omega, s_pi)
    n_val = torch.tensor([2.0], dtype=torch.float64)
    l_val = torch.tensor([1.0], dtype=torch.float64)
    m_val = torch.tensor([0.85 * math.sin(phase_t)], dtype=torch.float64)
    s_omega = torch.tensor([0.5 * math.cos(phase_t) + 0.35], dtype=torch.float64)
    s_pi = torch.tensor([0.5 * math.sin(phase_t - 0.6) - 0.35], dtype=torch.float64)

    state = build_five_quantum_twistor_state(n_val, l_val, m_val, s_omega, s_pi)
    Z_full = state["Z5_complete"][0]
    Z_inc = state["Z5_incomplete_4qn"][0]

    sigma = SIGMA_5
    M_full = torch.einsum("i,j->ij", Z_full, torch.matmul(Z_full.conj(), sigma.T))
    M_inc = torch.einsum("i,j->ij", Z_inc, torch.matmul(Z_inc.conj(), sigma.T))

    # Build normalized pixel grid for a single 5x5 panel (280x280)
    panel_size = 280
    ys = torch.linspace(-1.0, 1.0, panel_size, dtype=torch.float64)
    xs = torch.linspace(-1.0, 1.0, panel_size, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    def render_5x5_panel(M_mat: torch.Tensor, is_complete: bool) -> np.ndarray:
        canvas = torch.zeros((panel_size, panel_size, 3), dtype=torch.float64)
        centers = torch.linspace(-0.76, 0.76, 5, dtype=torch.float64)
        cell_radius = 0.145
        edge_softness = 0.035

        # Normalize magnitude scale for visual clarity
        mag_matrix = torch.abs(M_mat)
        phase_matrix = torch.angle(M_mat)
        max_mag = mag_matrix.max().clamp(min=1e-6)

        for r in range(5):
            for c in range(5):
                cy = centers[r]
                cx = centers[c]
                dist = torch.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)

                # Continuous erf() soft-box volumetric envelope (no discrete raster aliasing)
                envelope = 0.5 * (1.0 - torch.erf((dist - cell_radius) / edge_softness))
                core_glow = torch.exp(-(dist ** 2) / (2.0 * (0.065 ** 2)))

                mag = float(mag_matrix[r, c] / max_mag)
                ang = float(phase_matrix[r, c])

                # GPO Triad Color Mapping:
                # Diagonal trace elements -> Solar Gold (P_O)
                # Off-diagonal omega/pi spinor blocks -> Emerald (P_G) & Electric Violet/Magenta (P_P)
                if r == c:
                    col_r = 1.00
                    col_g = 0.82
                    col_b = 0.25
                elif (r < 2 and c >= 2) or (r >= 2 and c < 2):
                    # Off-diagonal spinor coupling blocks (omega <-> pi)
                    w_g = 0.5 + 0.5 * math.cos(ang + phase_t)
                    w_p = 1.0 - w_g
                    col_r = 0.15 * w_g + 0.95 * w_p
                    col_g = 0.98 * w_g + 0.20 * w_p
                    col_b = 0.75 * w_g + 0.95 * w_p
                else:
                    w_g = 0.5 + 0.5 * math.sin(ang - phase_t)
                    col_r = 0.20 + 0.65 * (1.0 - w_g)
                    col_g = 0.88 * w_g + 0.25 * (1.0 - w_g)
                    col_b = 0.95

                intensity = mag * (0.65 * envelope + 0.65 * core_glow)
                # Subtle background cell slot outline so all 25 positions are clearly visible
                slot_ring = 0.12 * torch.exp(-((dist - cell_radius) ** 2) / (2.0 * (0.008 ** 2)))

                canvas[:, :, 0] += intensity * col_r + slot_ring * (0.25 if is_complete else 0.45)
                canvas[:, :, 1] += intensity * col_g + slot_ring * (0.45 if is_complete else 0.18)
                canvas[:, :, 2] += intensity * col_b + slot_ring * (0.55 if is_complete else 0.25)

        canvas = (canvas.clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
        return canvas

    left_panel = render_5x5_panel(M_full, is_complete=True)
    right_panel = render_5x5_panel(M_inc, is_complete=False)

    # Assemble full HUD frame (360x680)
    full_img = Image.new("RGB", (width, height), (8, 10, 18))
    left_pil = Image.fromarray(left_panel)
    right_pil = Image.fromarray(right_panel)

    full_img.paste(left_pil, (35, 58))
    full_img.paste(right_pil, (365, 58))

    draw = ImageDraw.Draw(full_img)
    coupling_val = float(state["core_electron_spinor_coupling"][0].item())
    norm_full = float(state["sigma5_norm_complete"][0].item())

    draw.text((40, 14), "5-QN Complete Twistor (25 SIGMA_5 Freedoms)", fill=(80, 250, 190))
    draw.text((40, 32), f"omega^A + pi_A' Active | Tr(M_25) = {norm_full:+.3f}", fill=(200, 220, 255))

    draw.text((370, 14), "4-QN Incomplete State (Missing pi_A' = 0)", fill=(255, 110, 180))
    draw.text((370, 32), f"20/25 Freedoms Lost | Coupling = {coupling_val:+.3f}", fill=(230, 190, 220))

    draw.text(
        (175, 342),
        "Penrose Postulate IV: M^alpha_beta = Z^alpha (Sigma_5 Z)_beta^* (5x5 = 25 Elements)",
        fill=(245, 205, 95)
    )

    return np.array(full_img, dtype=np.uint8)


def generate_sigma5_25_freedoms_gif(
    out_path: str = "sigma5_25_freedoms_proof.gif",
    num_frames: int = 36,
    duration_ms: int = 55
) -> str:
    frames = []
    for idx in range(num_frames):
        rgb = render_25_freedom_matrix_frame(idx, total_frames=num_frames)
        frames.append(Image.fromarray(rgb))

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False
    )
    return os.path.abspath(out_path)


if __name__ == "__main__":
    saved = generate_sigma5_25_freedoms_gif()
    print(f"[SUCCESS] Generated 25-element SIGMA_5 proof GIF: {saved}")
