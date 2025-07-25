# pyRISE2

A modern Python package for solving and simulating DSGE models.

## Installation

```bash
pip install pyrise2
```

## Usage

```python
import pyrise2 as pr

model = pr.load_model("examples/japan_regime.yaml")
solution = pr.solve(model, order=2)
irfs = pr.simulate.irf(solution, horizon=20, shock="eps_r", regimes="all")
```

## Features

-   Load models from YAML files.
-   Solve models using first or second-order perturbation.
-   Simulate impulse response functions.
-   Support for Markov-switching models.

## Contributing

Interested in contributing? Check out the contributing guidelines.

Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

MIT
