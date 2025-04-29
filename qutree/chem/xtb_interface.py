import ase
from ase.io import write
import tempfile
import os
import re
import subprocess
import numpy as np
import torch


_xtb_method = {
    "gff": {"method_flags": ("--gff")},
    "gfn1": {"method_flags": ("--gfn", "1")},
    "gfn2": {"method_flags": ("--gfn", "2")},
}

def read_gradient(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f if line.strip() and not line.lstrip().startswith('#')]
    natoms = int(lines[0])
    grad_vals = [float(x) for x in lines[2 : 2 + natoms * 3]]
    return np.array(grad_vals).reshape(natoms, 3)


def xtb(molecule: ase.Atoms, method: str,
        charge: int, n_threads: int, solvent: str = None,
        gradient: bool = False) -> tuple:
    """
    Run xTB (energy or gradient)

    Parameters
    ----------
    molecule : ase.Atoms
        Computes gradient for this Molecule

    Returns
    -------
    energy : float
        Energy in Hartree
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write temporary file
        xyz_file = os.path.join(temp_dir, "tmp_input.xyz")
        write(xyz_file, molecule)
        flags = ["xtb", xyz_file]
        flags += [_xtb_method[method.lower()]["method_flags"]]
        flags += ["--charge", str(charge)]
        flags += ["--parallel", str(n_threads)]
        flags += ["--alpb", solvent] if solvent is not None else []
        if gradient:
            flags += ["--grad"]
        xtb_command = flags

        try:
            result = subprocess.run(
                xtb_command, check=True, capture_output=True, text=True, cwd=temp_dir
            )
        except subprocess.CalledProcessError as e:
            print("Error running xtb command:", e.stderr)
            raise e

        match = re.search(r"TOTAL ENERGY\s+([-+]?\d*\.\d+)\s+Eh", result.stdout)

        if match:
            energy = float(match.group(1))  # Extract and convert the energy to a float
        else:
            energy = None

        if gradient:
            dx = read_gradient(os.path.join(temp_dir, "tmp_input.engrad"))
            return energy, dx
        else:
            return energy, molecule.positions
        

class differentiable_xtb(torch.autograd.Function):
    """
    Autograd implementation of xTB
    """
    @staticmethod
    def forward(ctx, x, mol, method, charge, parallel, solvent = None):
        mol.positions = x.detach().numpy()
        e, _ = xtb(mol, method, charge, parallel, solvent, gradient = False)
        ctx.args = (mol, method, charge, parallel, solvent)
        ctx.save_for_backward(x)
        return torch.tensor([e], dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        mol, method, charge, parallel, solvent = ctx.args
        (x,) = ctx.saved_tensors
        mol.positions = x.detach().numpy()
        _, dx = xtb(mol, method, charge, parallel, solvent, gradient = True)
        grad_x = grad_output * torch.tensor(dx, dtype=grad_output.dtype, device=grad_output.device)
        return grad_x, None, None, None, None, None


def xtb_lbfgs(x, mol, method = 'gff', charge=0, parallel=1, solvent=[], max_iter = 100, lr = 0.1):
    """
    run L-BFGS on xTB using autograd

    x: torch tensor cartesian coordinates
    mol: ase.atoms molecule
    method (gff|gfn1|gfn2): xTB method
    max_iter: maximal number of iterations
    lr: initial step length
    """
    optimizer = torch.optim.LBFGS([x], max_iter=max_iter, lr=lr)
    
    def closure():
        optimizer.zero_grad()
        energy = differentiable_xtb.apply(x, mol, method, charge, parallel)
        energy.backward()
        return energy
    
    optimizer.step(closure)

