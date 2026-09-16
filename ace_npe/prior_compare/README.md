# Multi-prior comparison

This optional pipeline compares three ACE parameter-generating distributions:

1. independent `Uniform(0,1)` components;
2. normalized independent uniforms; and
3. a Dirichlet distribution on the ACE simplex.

Run the scripts from the `ace_npe` directory (or use their full paths from
anywhere):

```bash
python prior_compare/01_generate_prior_data.py --n_samples 50000
python prior_compare/02_validate_dirichlet_prior.py
python prior_compare/03_train_prior_models.py --epochs 500 --device cpu
python prior_compare/04_compare_prior_performance.py \
       --n_samples 500 --n_posterior_samples 1000
```

The generated data and model/output directory names remain `prior_comparison`
for compatibility with existing runs. The main `ace_npe` directory contains
the selected Dirichlet workflow and its fixed-N evaluation studies.
