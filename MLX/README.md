# MLX — Apple MLX Building Blocks

Shared MLX components that don't fit neatly inside any one strategy
family.  Strategies in `NNNC/`, `NNMT/`, etc. import from here when they
need a Mamba layer, parallel-scan kernel, or other low-level MLX util.

Apple-Silicon-only.  If `mlx.core.metal.is_available()` is False these
modules import but fail at runtime.

## Main files

| File | What it does |
|---|---|
| `mamba_mlx.py` | Mamba (selective state-space model) implementation in MLX.  Used by `NNNClassifierMLX_Mamba`. |
| `mamba2_mlx.py` | Mamba 2 implementation — improved variant with Triton-style kernels.  Used by `NNNClassifierMLX_Mamba2` and `NNNClassifierMLX_TSMamba`. |
| `mamba_lm_mlx.py` | Language-model-style wrapper around Mamba (sequence prediction head). |
| `pscan_mlx.py` | Parallel-scan primitive used by the Mamba implementations. |
| `mlx_misc.py` | Small utility layers and helpers shared across MLX strategies. |
| `mlx_utils.py` | General MLX utility functions (tensor manipulation, weight init helpers, etc.). |

## When to add things here

Anything that's:
* MLX-specific (uses `mlx.core` or `mlx.nn` directly), AND
* Used by more than one strategy family.

Strategy- or family-specific MLX code should live in the family
directory itself (e.g. `NNMT/NNMTClassifierMLX.py`,
`NNNC/NNNClassifierMLX.py`).
