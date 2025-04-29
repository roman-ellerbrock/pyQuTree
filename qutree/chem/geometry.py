import torch


"""
Torch version of BAT coordinates for automated differentiation
Translated from ASE implementation with slight adjustments
"""

def get_dihedral(X, a0, a1, a2, a3):
    """
    Compute dihedral for vertices a0, a1, a2, a3 in X
    
    X: cartesian coordinates
    a0, a1, a2, a3: indices of atoms
    """
    b0 = X[a1] - X[a0]
    b1 = X[a2] - X[a1]
    b2 = X[a3] - X[a2]

    b1_norm = b1 / torch.norm(b1)

    # Project b0 and b2 onto plane perpendicular to b1
    v = -b0 - torch.dot(-b0, b1_norm) * b1_norm
    w = b2 - torch.dot(-b2, b1_norm) * b1_norm

    if torch.allclose(v, torch.zeros_like(v)) or torch.allclose(w, torch.zeros_like(w)):
        raise ZeroDivisionError("Undefined dihedral for planar inner angle")

    x = torch.dot(v, w)
    y = torch.linalg.cross(b1_norm, v).dot(w)
    angle = torch.atan2(y, x)
    return torch.rad2deg(angle % (2 * torch.pi))


def masked_rotate(X, center, axis, angle_rad, mask):
    """
    Rotate mask-atoms in X atound axis. Center has to be point on axis
    center: (3,)
    axis: (3,), not necessarily normalized
    angle_rad: scalar
    mask: (N,) boolean
    X: (N, 3)
    """

    axis = axis / torch.norm(axis)

    # Select subgroup
    subgroup = X[mask] - center  # translate to origin

    # Rodrigues' rotation formula
    cos = torch.cos(angle_rad)
    sin = torch.sin(angle_rad)
    dot = (subgroup * axis).sum(dim=1, keepdim=True)
    rotated = (
        subgroup * cos
        + torch.cross(axis.expand_as(subgroup), subgroup, dim=1) * sin
        + axis * dot * (1 - cos)
    )

    # Translate back and write to X
    X[mask] = rotated + center


def set_dihedral(X, a0, a1, a2, a3, angle, mask=None, indices=None):
    """
    Set dihedral in X
    
    X: atom positions
    a0, a1, a2, a3: indices o atoms
    angle: taget value for dihedral
    mask: target atom indices that will move
    """
    # angle = angle_deg * torch.pi / 180

    if mask is None and indices is None:
        mask = torch.zeros(len(X), dtype=torch.bool, device=X.device)
        mask[a3] = True
    elif indices is not None:
        mask = torch.zeros(len(X), dtype=torch.bool, device=X.device)
        mask[indices] = True
    else:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=X.device)

    current = get_dihedral(X, a0, a1, a2, a3)
    diff = torch.deg2rad(angle - current)
    axis = X[a2] - X[a1]
    center = X[a2]
    masked_rotate(X, center, axis, diff, mask)


def moment_of_inertia(coords: torch.Tensor) -> torch.Tensor:
    coords = coords - coords.mean(dim=0, keepdim=True)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    Ixx = (y**2 + z**2).sum()
    Iyy = (x**2 + z**2).sum()
    Izz = (x**2 + y**2).sum()
    Ixy = -(x * y).sum()
    Ixz = -(x * z).sum()
    Iyz = -(y * z).sum()
    return torch.stack(
        [
            torch.stack([Ixx, Ixy, Ixz]),
            torch.stack([Ixy, Iyy, Iyz]),
            torch.stack([Ixz, Iyz, Izz]),
        ]
    )
