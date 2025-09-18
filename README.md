# liteMRT: Lightweight Python Radiative Transfer Model

A purely Python-based radiative transfer code that bridges the gap between educational accessibility and operational accuracy. Built upon the Gauss-Seidel iterative method, liteMRT demonstrates that sophisticated radiative transfer modeling can be achieved through a concise, well-structured codebase.

## Key Features

- **Educational Accessibility**: Only 278 lines of core Python code with five well-defined functions
- **Operational Accuracy**: Maximum relative errors <0.004% against LIDORT benchmark
- **Pure Python Implementation**: Minimal dependencies, accelerated via Numba JIT
- **Non-homogeneous Atmospheres**: Handles vertically varying optical properties
- **BRDF Surface Models**: Support for non-Lambertian surface reflections
- **Computational Efficiency**: Near-native performance through JIT compilation

## Installation

### Dependencies
- Python 3.7+
- NumPy
- Numba

### Quick Install
```bash
git clone https://github.com/TaoRen-Rad/liteMRT.git
cd liteMRT
```

## Usage

### Basic Example
```python
from liteMRT.lite_mrt import solve_lattice
from liteMRT.support_func import gauss_zeroes_weights
from liteMRT.brdf import rho

# Define atmospheric parameters
nit = 30          # Number of iterations
ng1 = 24          # Gaussian nodes per hemisphere  
nm = 32           # Azimuthal Fourier moments
dtau = 0.002      # Layer optical thickness
nlr = 125         # Number of layers

# Solar and viewing geometry
szds = [20, 40, 60]  # Solar zenith angles (degrees)
vzds = [0, 30, 60]   # Viewing zenith angles (degrees) 
azds = [0, 90, 180]  # Relative azimuth angles (degrees)

# Atmospheric optical properties
xk = ...    # Phase function moments [nlr, nk]
ssa = ...   # Single scattering albedo [nlr]
brdf_pars = ...  # BRDF parameters

# Solve radiative transfer
results = solve_lattice(nit, ng1, nm, szds, vzds, azds, 
                       dtau, nlr, xk, ssa, brdf_pars)
```

### Example script

See `main.py` for complete examples.

## Code Structure

```
liteMRT/
├── liteMRT/
│   ├── lite_mrt.py      # Core RT algorithms (278 lines)
│   ├── support_func.py  # Mathematical utilities
│   ├── brdf.py         # Surface reflection models
│   └── __init__.py     # Auto-JIT decoration
├── testutils/          # Validation utilities
├── benchmark/          # LIDORT comparison
└── main.py            # Example applications
```

## Core Algorithms

The model implements five core functions corresponding to the mathematical formulation:

1. **`single_scattering_up/down`**: Single-scattering approximation
2. **`gauss_seidel_iterations_m`**: Gauss-Seidel iterative solver  
3. **`source_function_integrate_up/down`**: Source function integration
4. **`solve_lattice`**: Main solution procedure

## Mathematical Foundation

liteMRT solves the radiative transfer equation using:

- **Fourier Decomposition**: Azimuthal symmetry exploitation
- **Gaussian Quadrature**: Efficient angular integration
- **Gauss-Seidel Iteration**: Multiple scattering solution
- **Source Function Integration**: Arbitrary viewing directions

For complete mathematical details, see the accompanying paper.
