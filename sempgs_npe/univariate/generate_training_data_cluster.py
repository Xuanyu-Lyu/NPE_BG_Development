"""
Data Generation Script for Neural Network Training for Univariate SEMPGS as a Validation - COMBINED VERSION

This script runs forward-time simulations and immediately analyzes them without
saving raw simulation data. The input is the set of data-generating parameters, and the output is the covariances
of variables used for univariate SEMPGS model. 

Usage:
    python generate_training_data_cluster.py
    (Run via SLURM array job - see submit_cluster_job.sh)
"""

import numpy as np
import pandas as pd
import sys
import os
import traceback
from pathlib import Path

# Add the SimulationFunctions directory to path
script_dir = Path(__file__).parent
simfunc_dir = script_dir.parent.parent.parent / "SimulationFunctions"
sys.path.insert(0, str(simfunc_dir))

from core_simulation import AssortativeMatingSimulation

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory setup - using a new directory for this combined approach
PROJECT_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data_Vali/DataGeneratingNN_uniSEMPGS_20250223")

# Get SLURM array task ID
SLURM_TASK_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', '1'))

# Simulation parameters
N_CONDITIONS = 1000              # Total number of parameter conditions to generate
ITERATIONS_PER_CONDITION = 30
POP_SIZE = 20000             
N_GENERATIONS = 20           
FINAL_GENS = [18, 19]         
N_CV = 2000                   
MAF_MIN = 0.01
MAF_MAX = 0.5

# Parameter ranges for data generation (5+ values each)
PARAM_RANGES = {
    'f11': [0.00, 0.07, 0.14, 0.21, 0.28, 0.35],
    'f22': [0.00, 0.07, 0.14, 0.21, 0.28, 0.35],
    'f12': [0.00, 0.07, 0.14, 0.21, 0.28, 0.35],
    'f21': [0.00, 0.07, 0.14, 0.21, 0.28, 0.35],
    'prop_h2_latent1': [0.00, 0.20, 0.40, 0.60, 0.80, 1.00],
    'prop_h2_latent2': [0.00, 0.20, 0.40, 0.60, 0.80, 1.00],
    'vg1': [0.05, 0.20, 0.40, 0.60, 0.80, 1.00],
    'vg2': [0.05, 0.20, 0.40, 0.60, 0.80, 1.00],
    're':  [0.00, 0.20, 0.40, 0.60, 0.80, 1.00],
    'am11': [0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
    'am12': [0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
    'am21': [0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
    'am22': [0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
    'rg':   [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
}

# Fixed parameters
FIXED_PARAMS = {
    #'am11': 0,
    #'am12': 0,
    #'am21': 0,
    #'f12': 0,
    #'f21': 0,
    #'re': 0,
    #'rg':0
}

# Relationship types to analyze
# RELATIONSHIP_TYPES = [
#     'S',        # Siblings
#     'M',        # Mates
#     'P',        # Parent-child
#     'PP',       # Grandparent-grandchild
#     # 'HSFS',     # Half-siblings
#     # 'PSC',      # Parent-sibling-child (avuncular)
#     # 'PPSCC',    # First cousins
#     # 'MS',       # Mate's sibling (sibling-in-law)
#     # 'SMS',      # Sibling's mate's sibling
#     # 'MSC',      # Mate's sibling's child (nibling-in-law)
#     # 'MSM',      # Mate's sibling's mate
#     # 'SMSC',     # Sibling's mate's sibling's child
#     # 'SMSM',     # Sibling's mate's sibling's mate
#     # 'SMSMS',    # Sibling's mate's sibling's mate's sibling
#     # 'MSMSM',    # Mate's sibling's mate's sibling's mate
#     # 'MSMSC',    # Mate's sibling's mate's sibling's child
#     # 'PSMSC',    # Parent's sibling's mate's sibling's child
#     # 'SMSMSC',   # Sibling's mate's sibling's mate's sibling's child
#     # 'MSMSMS',   # Mate's sibling's mate's sibling's mate's sibling
# ]

# ============================================================================
# PARAMETER GENERATION
# ============================================================================

def generate_conditions(n_conditions=1000, seed=62):
    """Generate parameter combinations."""
    np.random.seed(seed)
    
    conditions = []
    param_names = list(PARAM_RANGES.keys())
    
    for i in range(n_conditions):
        condition = {'name': f'Condition_{i+1:04d}'}
        
        for param_name in param_names:
            param_values = PARAM_RANGES[param_name]
            condition[param_name] = np.random.choice(param_values)
        
        condition.update(FIXED_PARAMS)
        conditions.append(condition)
    
    return conditions

CONDITIONS_FILE = PROJECT_BASE / "conditions_config.csv"

def save_conditions_config():
    """Save all conditions to a CSV file for reference and reproducibility."""
    PROJECT_BASE.mkdir(parents=True, exist_ok=True)
    
    n_conditions_needed = N_CONDITIONS
    needs_regeneration = False
    
    if CONDITIONS_FILE.exists():
        # Check if existing file has the right number of conditions
        existing_df = pd.read_csv(CONDITIONS_FILE)
        if len(existing_df) != n_conditions_needed:
            print(f"⚠ Existing config has {len(existing_df)} conditions, need {n_conditions_needed}")
            needs_regeneration = True
        else:
            print(f"✓ Using existing conditions from {CONDITIONS_FILE} ({len(existing_df)} conditions)")
    else:
        needs_regeneration = True
    
    if needs_regeneration:
        print(f"Generating {n_conditions_needed} parameter conditions...")
        conditions = generate_conditions(n_conditions=n_conditions_needed, seed=42)
        df = pd.DataFrame(conditions)
        df.to_csv(CONDITIONS_FILE, index=False)
        print(f"✓ Saved {len(conditions)} conditions to {CONDITIONS_FILE}")

def load_condition(task_id):
    """Load the condition for a specific task ID."""
    df = pd.read_csv(CONDITIONS_FILE)
    if task_id > len(df):
        raise ValueError(f"Task ID {task_id} exceeds number of conditions ({len(df)})")
    return df.iloc[task_id - 1].to_dict()

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def setup_matrices(params):
    """Setup covariance and other matrices based on simulation parameters."""
    vg1 = params['vg1']
    vg2 = params['vg2']
    rg = params['rg']
    re = params['re']
    prop_h2_latent1 = params['prop_h2_latent1']
    prop_h2_latent2 = params['prop_h2_latent2']
    
    # Implied variables (t0)
    k2_matrix = np.array([[1, rg], [rg, 1]])
    
    # Observable genetic variance components
    vg_obs1 = vg1 * (1 - prop_h2_latent1)
    vg_obs2 = vg2 * (1 - prop_h2_latent2)
    d11 = np.sqrt(vg_obs1)
    d21 = 0
    d22 = np.sqrt(vg_obs2 - d21**2)
    delta_mat = np.array([[d11, 0], [d21, d22]])
    
    # Latent genetic variance components
    vg_lat1 = vg1 * prop_h2_latent1
    vg_lat2 = vg2 * prop_h2_latent2
    a11 = np.sqrt(vg_lat1)
    a21 = 0
    a22 = np.sqrt(vg_lat2 - a21**2)
    a_mat = np.array([[a11, 0], [a21, a22]])
    
    # Total genetic covariance
    covg_mat = (delta_mat @ k2_matrix @ delta_mat.T) + (a_mat @ k2_matrix @ a_mat.T)
    
    # Environmental covariance
    ve1 = 1 - vg1
    ve2 = 1 - vg2
    cove = re * np.sqrt(ve1 * ve2)
    cove_mat = np.array([[ve1, cove], [cove, ve2]])
    
    # Total phenotypic covariance
    covy_mat = covg_mat + cove_mat
    
    # Mate correlation matrix for bivariate AM (both traits)
    am11 = params['am11']
    am12 = params['am12']
    am21 = params['am21']
    am22 = params['am22']
    am_mat = np.array([[am11, am12], [am21, am22]])

    # Vertical transmission matrix
    f11 = params['f11']
    f12 = params['f12']
    f21 = params['f21']
    f22 = params['f22']
    f_mat = np.array([[f11, f12], [f21, f22]])
    
    # Social homogamy matrix (set to zero - phenotypic AM only)
    s_mat = np.zeros((2, 2))
    
    # AM list: list of 2x2 matrices for bivariate mating on both traits
    am_list = [am_mat.copy() for _ in range(N_GENERATIONS)]

    return {
        'cove_mat': cove_mat,
        'f_mat': f_mat,
        's_mat': s_mat,
        'a_mat': a_mat,
        'd_mat': delta_mat,
        'am_list': am_list,
        'mate_on_trait': None,
        'covy_mat': covy_mat,
        'k2_matrix': k2_matrix
    }

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

# Ordered 16 column labels for the covariance matrix
_MEMBERS   = ['dad', 'mom', 'off1', 'off2']
_VARS      = ['Y1', 'Y2', 'PGS1', 'PGS2']
COV_LABELS = [f'{m}_{v}' for m in _MEMBERS for v in _VARS]  # 16 labels


def extract_and_analyze_relationships(results, iteration):
    """
    Build a 16x16 covariance matrix across nuclear families.

    The 16 variables are (in order):
        dad_Y1, dad_Y2, dad_PGS1, dad_PGS2,
        mom_Y1, mom_Y2, mom_PGS1, mom_PGS2,
        off1_Y1, off1_Y2, off1_PGS1, off1_PGS2,
        off2_Y1, off2_Y2, off2_PGS1, off2_PGS2

    Nuclear families are identified from the FINAL 2 generations only.
    Parents of the youngest generation must appear in the second-to-last
    generation's PHEN frame (naturally enforced by the trimmed_results pool).
    Families with exactly one offspring leave all off2 columns as NaN
    (those families contribute to 12x12 sub-covariances but not the off2 block).
    Families with >2 offspring: 2 are randomly selected.

    Returns: (cov_df, n_families)
        cov_df     -- 16x16 DataFrame with COV_LABELS as both index and columns
        n_families -- int, number of families used
    """
    vars_of_interest = ['Y1', 'Y2', 'PGS1', 'PGS2']

    # ------------------------------------------------------------------ #
    # 1. Build measures lookup from the final 2 PHEN frames                #
    # ------------------------------------------------------------------ #
    phen_frames = results['HISTORY']['PHEN']   # 2-element list
    all_phen = pd.concat(phen_frames, ignore_index=True)

    for col in ['ID', 'Father.ID', 'Mother.ID',
                'Y1', 'Y2', 'TPO1', 'TPO2', 'TMO1', 'TMO2']:
        all_phen[col] = pd.to_numeric(all_phen[col], errors='coerce')

    all_phen['PGS1'] = all_phen['TPO1'] + all_phen['TMO1']
    all_phen['PGS2'] = all_phen['TPO2'] + all_phen['TMO2']

    measures_lookup = all_phen.set_index('ID')[vars_of_interest].to_dict('index')

    # ------------------------------------------------------------------ #
    # 2. Identify nuclear families: offspring in the LAST phen frame       #
    #    whose parents appear in the second-to-last frame                  #
    # ------------------------------------------------------------------ #
    offspring = phen_frames[-1].copy()  # youngest generation
    for col in ['ID', 'Father.ID', 'Mother.ID']:
        offspring[col] = pd.to_numeric(offspring[col], errors='coerce')
    offspring = offspring.dropna(subset=['Father.ID', 'Mother.ID'])
    offspring = offspring[
        offspring['Father.ID'].isin(measures_lookup) &
        offspring['Mother.ID'].isin(measures_lookup)
    ]

    if offspring.empty:
        print(f"    ✗ No nuclear families found in final generation")
        return None, 0

    rng = np.random.default_rng(seed=iteration)

    family_records = []
    for (father_id, mother_id), sib_df in offspring.groupby(['Father.ID', 'Mother.ID']):
        dad_m = measures_lookup.get(father_id)
        mom_m = measures_lookup.get(mother_id)
        if dad_m is None or mom_m is None:
            continue

        sib_ids = sib_df['ID'].tolist()
        if len(sib_ids) == 1:
            off1_m = measures_lookup.get(sib_ids[0])
            off2_m = None
        else:
            chosen = rng.choice(sib_ids, size=2, replace=False)
            off1_m = measures_lookup.get(int(chosen[0]))
            off2_m = measures_lookup.get(int(chosen[1]))

        record = {}
        for v in vars_of_interest:
            record[f'dad_{v}']  = dad_m.get(v, np.nan)
            record[f'mom_{v}']  = mom_m.get(v, np.nan)
            record[f'off1_{v}'] = off1_m.get(v, np.nan) if off1_m else np.nan
            record[f'off2_{v}'] = off2_m.get(v, np.nan) if off2_m else np.nan
        family_records.append(record)

    if not family_records:
        print(f"    ✗ No valid nuclear families assembled")
        return None, 0

    fam_df = pd.DataFrame(family_records, columns=COV_LABELS)
    n_families = len(fam_df)
    print(f"    Found {n_families} nuclear families")

    # ------------------------------------------------------------------ #
    # 3. Compute the 16x16 covariance matrix (pairwise complete-obs)       #
    # ------------------------------------------------------------------ #
    cov_array = np.full((16, 16), np.nan)
    for i, c1 in enumerate(COV_LABELS):
        for j, c2 in enumerate(COV_LABELS):
            if j < i:
                cov_array[i, j] = cov_array[j, i]  # symmetric
                continue
            valid = fam_df[[c1, c2]].dropna()
            if len(valid) > 1:
                cov_array[i, j] = np.cov(valid[c1], valid[c2], ddof=1)[0, 1]

    cov_df = pd.DataFrame(cov_array, index=COV_LABELS, columns=COV_LABELS)
    return cov_df, n_families

def run_single_iteration(iteration, condition_name, params, matrices):
    """Run a single simulation iteration and analyze it immediately."""
    print(f"\n  Iteration {iteration + 1}/{ITERATIONS_PER_CONDITION}")
    
    # Set seed for reproducibility
    condition_hash = hash(condition_name) % 100000
    seed = condition_hash * 100 + iteration + 1
    
    # Extract mate_on_trait (None = bivariate mating on both traits)
    mate_on_trait = matrices.pop('mate_on_trait', None)
    
    # Initialize simulation
    sim = AssortativeMatingSimulation(
        n_CV=N_CV,
        rg_effects=params['rg'],
        maf_min=MAF_MIN,
        maf_max=MAF_MAX,
        num_generations=N_GENERATIONS,
        pop_size=POP_SIZE,
        mating_type="phenotypic",
        avoid_inbreeding=True,
        save_each_gen=True,
        save_covs=True,
        seed=seed,
        output_summary_filename=None,  # Don't save summary
        mate_on_trait=mate_on_trait,
        **matrices
    )
    
    # Run simulation
    print(f"    Running simulation...")
    results = sim.run_simulation()
    
    # Trim results to only keep final generations to save memory
    # Note: MATES has offset indexing - MATES[i+1] contains mates FROM generation i
    # For FINAL_GENS = [12, 13, 14], we need MATES[12], [13], [14], and [15] (for mating in gen 14)
    mates_indices = FINAL_GENS + [FINAL_GENS[-1] + 1] if FINAL_GENS[-1] + 1 < len(results['HISTORY']['MATES']) else FINAL_GENS
    
    trimmed_results = {
        'HISTORY': {
            'PHEN': [results['HISTORY']['PHEN'][i] for i in FINAL_GENS],
            'XO': [results['HISTORY']['XO'][i] for i in FINAL_GENS],
            'XL': [results['HISTORY']['XL'][i] for i in FINAL_GENS],
            'MATES': [results['HISTORY']['MATES'][i] if i < len(results['HISTORY']['MATES']) else None for i in mates_indices]
        }
    }
    
    # Compute nuclear-family covariance matrix
    print(f"    Computing nuclear family covariance matrix...")
    cov_df, n_families = extract_and_analyze_relationships(trimmed_results, iteration)

    if cov_df is not None:
        print(f"    ✓ Built 16x16 covariance matrix from {n_families} families")
        return cov_df, n_families
    else:
        print(f"    ✗ No covariance matrix computed")
        return None, 0

# ============================================================================
# MAIN CONDITION PROCESSING
# ============================================================================

def run_condition(condition, project_base):
    """Run all iterations for a single condition."""
    condition_name = condition['name']
    
    print(f"\n{'#'*70}")
    print(f"# {condition_name}")
    print(f"# Parameters:")
    print(f"#   f11={condition['f11']:.4f}, vg1={condition['vg1']:.4f}, prop_h2_latent1={condition['prop_h2_latent1']:.4f}")
    print(f"#   f22={condition['f22']:.4f}, vg2={condition['vg2']:.4f}, am22={condition['am22']:.4f}")
    print(f"#   rg={condition['rg']:.4f}")
    print(f"# Running {ITERATIONS_PER_CONDITION} iterations")
    print(f"# SLURM Task ID: {SLURM_TASK_ID}")
    print(f"{'#'*70}\n")
    
    # Create condition-specific directory
    project_dir = project_base / condition_name
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Save condition parameters immediately
    params_file = project_dir / "condition_parameters.csv"
    params_df = pd.DataFrame([condition])
    params_df.to_csv(params_file, index=False)
    print(f"  ✓ Saved parameters to {params_file}")
    
    # Setup matrices
    matrices = setup_matrices(condition)
    
    # Track iteration status and results
    iteration_status = []
    all_cov_matrices = []   # list of (iteration_number, cov_df 16x16, n_families)

    # Run iterations
    for iteration in range(ITERATIONS_PER_CONDITION):
        iter_info = {
            'Iteration': iteration + 1,
            'Status': 'Failed',
            'Error': None,
            'N_Families': 0
        }

        try:
            cov_df, n_families = run_single_iteration(iteration, condition_name, condition, matrices.copy())

            if cov_df is not None:
                iter_info['Status'] = 'Success'
                iter_info['N_Families'] = n_families
                all_cov_matrices.append((iteration + 1, cov_df, n_families))
        except Exception as e:
            iter_info['Error'] = str(e)
            print(f"    ✗ Error in Iteration {iteration+1}: {e}")
            traceback.print_exc()

        iteration_status.append(iter_info)

    # Save iteration status summary
    status_file = project_dir / "iteration_status.csv"
    status_df = pd.DataFrame(iteration_status)
    status_df.to_csv(status_file, index=False)
    print(f"\n  ✓ Saved iteration status to {status_file}")

    # Save results for this condition
    if all_cov_matrices:
        print(f"\n  Saving covariance matrix results for {condition_name}...")

        # ---------------------------------------------------------------- #
        # (a) Mean 16x16 covariance matrix across iterations               #
        # ---------------------------------------------------------------- #
        stack = np.array([cm.values for _, cm, _ in all_cov_matrices])
        mean_cov = pd.DataFrame(
            np.nanmean(stack, axis=0),
            index=COV_LABELS, columns=COV_LABELS
        ).round(8)
        mean_cov_file = project_dir / "mean_cov_matrix.csv"
        mean_cov.to_csv(mean_cov_file)
        print(f"  ✓ Saved mean 16x16 covariance matrix to {mean_cov_file}")

        # ---------------------------------------------------------------- #
        # (b) All-iterations file: upper triangle flattened, one row/iter  #
        #     Columns: Iteration, N_Families, cov_{label_i}_{label_j}      #
        # ---------------------------------------------------------------- #
        upper_idx = [(i, j) for i in range(16) for j in range(i, 16)]
        upper_cols = [f"cov_{COV_LABELS[i]}_{COV_LABELS[j]}" for i, j in upper_idx]

        iter_rows = []
        for iter_num, cov_df, n_fam in all_cov_matrices:
            flat = {upper_cols[k]: cov_df.values[i, j] for k, (i, j) in enumerate(upper_idx)}
            flat['Iteration']  = int(iter_num)
            flat['N_Families'] = int(n_fam)
            iter_rows.append(flat)

        all_iters_df = pd.DataFrame(iter_rows)
        # Reorder: Iteration, N_Families first
        lead_cols = ['Iteration', 'N_Families']
        all_iters_df = all_iters_df[lead_cols + upper_cols]
        all_iters_file = project_dir / "all_iterations_cov_matrices.csv"
        all_iters_df.to_csv(all_iters_file, index=False)
        print(f"  ✓ Saved {len(all_iters_df)} iteration matrices ({len(upper_cols)} upper-triangle entries each) to {all_iters_file}")

        # ---------------------------------------------------------------- #
        # (c) NN training format: params + upper triangle, one row/iter    #
        # ---------------------------------------------------------------- #
        print(f"  Creating NN training format...")
        nn_training_data = []
        for iter_num, cov_df, n_fam in all_cov_matrices:
            row = {'Iteration': int(iter_num), 'Condition': condition_name,
                   'N_Families': int(n_fam)}
            for param in PARAM_RANGES.keys():
                row[f'param_{param}'] = condition[param]
            for k, (i, j) in enumerate(upper_idx):
                row[upper_cols[k]] = cov_df.values[i, j]
            nn_training_data.append(row)

        nn_training_df = pd.DataFrame(nn_training_data)
        nn_training_file = project_dir / "nn_training_format.csv"
        nn_training_df.to_csv(nn_training_file, index=False)
        print(f"  ✓ Saved NN training format to {nn_training_file}")

        n_success = sum(1 for s in iteration_status if s['Status'] == 'Success')
        print(f"\n  Summary: {n_success}/{ITERATIONS_PER_CONDITION} iterations successful")
        return True
    else:
        print(f"\n  ⚠ No covariance matrices computed for {condition_name}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("DATA GENERATION FOR NEURAL NETWORK TRAINING - COMBINED VERSION")
    print("="*70)
    print(f"Project base: {PROJECT_BASE}")
    print(f"SLURM Task ID: {SLURM_TASK_ID}")
    print(f"Population size: {POP_SIZE}")
    print(f"Generations: {N_GENERATIONS} (analyzing final 2)")
    print(f"Iterations per condition: {ITERATIONS_PER_CONDITION}")
    print(f"Causal variants: {N_CV}")
    print("="*70 + "\n")
    
    # Save/load conditions configuration
    save_conditions_config()
    
    # Load condition for this task
    try:
        condition = load_condition(SLURM_TASK_ID)
        print(f"Loaded condition for Task {SLURM_TASK_ID}")
    except Exception as e:
        print(f"Error loading condition: {e}")
        return
    
    # Run the condition
    success = run_condition(condition, PROJECT_BASE)
    
    if success:
        print(f"\n{'='*70}")
        print(f"TASK {SLURM_TASK_ID} COMPLETED SUCCESSFULLY")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print(f"TASK {SLURM_TASK_ID} COMPLETED WITH ERRORS")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
