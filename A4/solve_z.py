import re
import sys
import numpy as np
from collections import defaultdict
import random

def parse_bif(filename):
    """Parse the BIF file to extract network structure and parameters."""
    with open(filename, 'r') as f:
        text = f.read()
    
    # Extract variable definitions
    var_pattern = re.compile(
        r'variable\s+"([^"]+)"\s*\{\s*type\s+discrete\s*\[\s*(\d+)\s*\]\s*\{\s*([^}]+)\s*\}\s*;',
        re.MULTILINE
    )

    variables = {}
    for name, num_values, values_str in var_pattern.findall(text):
        values = [v.strip().strip('"') for v in re.split(r'\s+', values_str) if v.strip()]
        variables[name] = {
            "values": values,
            "parents": [],
            "probabilities": {}
        }
    
    # Extract probability tables
    prob_pattern = re.compile(
        r'probability\s*\(\s*"([^"]+)"\s*(?:\|\s*([^)]+))?\s*\)\s*\{(.*?)\};',
        re.DOTALL | re.MULTILINE
    )

    for target, parents_str, body in prob_pattern.findall(text):
        target = target.strip()
        parents = [p.strip().strip('"') for p in parents_str.split(',')] if parents_str else []

        if target not in variables:
            variables[target] = {"values": [], "parents": parents, "probabilities": {}}
        else:
            variables[target]["parents"] = parents

        # Parse probability values
        body = body.strip()
        if body.startswith("table"):
            # Simple unconditional case
            table_vals = re.findall(r'[-\d\.]+', body)
            probs = [float(v) if v != '-1' else -1 for v in table_vals]
            variables[target]["probabilities"]["()"] = probs
        else:
            # Conditional case with parent values
            row_pattern = re.compile(r'\(([^\)]+)\)\s*([^\n;]+)')
            for parent_vals, probs_str in row_pattern.findall(body):
                key = tuple(v.strip().strip('"') for v in parent_vals.split(','))
                probs = [float(x) if x != '-1' else -1 for x in re.findall(r'[-\d\.]+', probs_str)]
                variables[target]["probabilities"][key] = probs

    return variables

def parse_data(filename, variables):
    """Parse the weather data records."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        values = line.strip().split()
        record = {}
        for i, (var_name, var_info) in enumerate(variables.items()):
            if i < len(values):
                if values[i] == "?":
                    record[var_name] = None  # Missing value
                else:
                    # Convert to index in variable's values list
                    try:
                        record[var_name] = var_info["values"].index(values[i].strip('"'))
                    except ValueError:
                        # Handle potential formatting issues
                        record[var_name] = None
            else:
                record[var_name] = None
        data.append(record)
    return data

def initialize_parameters(variables):
    """Initialize missing parameters with uniform distributions."""
    for var_name, var_info in variables.items():
        for parent_config, probs in var_info["probabilities"].items():
            # Check if any probability is -1 (missing)
            if -1 in probs:
                # Initialize with uniform distribution
                num_values = len(var_info["values"])
                uniform_prob = 1.0 / num_values
                variables[var_name]["probabilities"][parent_config] = [uniform_prob] * num_values
    return variables

def get_parent_config(record, parents, variables):
    """Get the parent configuration for a record."""
    config = []
    for parent in parents:
        if record[parent] is None:
            return None  # Can't determine config if parent is missing
        parent_val = variables[parent]["values"][record[parent]]
        config.append(parent_val)
    return tuple(config)

def learn_parameters(variables, data, max_iterations=50, convergence_threshold=1e-4):
    """Learn parameters using Expectation-Maximization algorithm."""
    # Initialize missing parameters
    variables = initialize_parameters(variables)
    
    # Count occurrences for each variable configuration
    counts = {}
    for var_name, var_info in variables.items():
        counts[var_name] = {}
        for parent_config in var_info["probabilities"]:
            counts[var_name][parent_config] = [0] * len(var_info["values"])
    
    # EM algorithm
    prev_error = float('inf')
    for iteration in range(max_iterations):
        # E-step: Calculate expected counts
        new_counts = {}
        for var_name, var_info in variables.items():
            new_counts[var_name] = {}
            for parent_config in var_info["probabilities"]:
                new_counts[var_name][parent_config] = [0.0] * len(var_info["values"])
        
        for record in data:
            # Process each record
            for var_name, var_info in variables.items():
                parents = var_info["parents"]
                
                # Get parent configuration
                if parents:
                    parent_config = get_parent_config(record, parents, variables)
                    if parent_config is None:
                        continue  # Skip if parent config can't be determined
                else:
                    parent_config = ()
                
                # If variable value is observed
                if record[var_name] is not None:
                    value_idx = record[var_name]
                    new_counts[var_name][parent_config][value_idx] += 1
                else:
                    # Variable is missing - use current probabilities to estimate
                    probs = variables[var_name]["probabilities"][parent_config]
                    for i, p in enumerate(probs):
                        new_counts[var_name][parent_config][i] += p
        
        # M-step: Update parameters based on expected counts
        error = 0
        for var_name, var_info in variables.items():
            for parent_config, probs in var_info["probabilities"].items():
                total = sum(new_counts[var_name][parent_config])
                if total > 0:
                    new_probs = [count / total for count in new_counts[var_name][parent_config]]
                    # Calculate error
                    for i in range(len(probs)):
                        error += abs(probs[i] - new_probs[i])
                    # Update probabilities
                    variables[var_name]["probabilities"][parent_config] = new_probs
        
        # Check for convergence
        if abs(error - prev_error) < convergence_threshold:
            break
        prev_error = error
    
    return variables

def write_bif(variables, output_filename):
    """Write the learned parameters to a BIF file."""
    with open(output_filename, 'w') as f:
        # Write variable definitions
        for var_name, var_info in variables.items():
            f.write(f'variable "{var_name}" {{\n')
            f.write(f'  type discrete[{len(var_info["values"])}] {{ ')
            f.write(' '.join(f'"{v}"' for v in var_info["values"]))
            f.write(' };\n')
            f.write('}\n\n')
        
        # Write probability tables
        for var_name, var_info in variables.items():
            f.write(f'probability ( "{var_name}"')
            if var_info["parents"]:
                f.write(' | ')
                f.write(' '.join(f'"{p}"' for p in var_info["parents"]))
            f.write(' ) {\n')
            
            for parent_config, probs in var_info["probabilities"].items():
                if parent_config == "()":
                    # No parents
                    f.write('  table ')
                    f.write(' '.join(f'{p:.4f}' for p in probs))
                    f.write(' ;\n')
                else:
                    # Has parents
                    f.write('  ( ')
                    f.write(' '.join(f'"{v}"' for v in parent_config))
                    f.write(' ) ')
                    f.write(' '.join(f'{p:.4f}' for p in probs))
                    f.write(' ;\n')
            
            f.write('}\n\n')

def main():
    if len(sys.argv) != 3:
        print("Usage: python solve.py <hailfinder.bif> <records.dat>")
        sys.exit(1)
    
    bif_file = sys.argv[1]
    data_file = sys.argv[2]
    
    # Parse the BIF file
    variables = parse_bif(bif_file)
    
    # Parse the data file
    data = parse_data(data_file, variables)
    
    # Learn parameters
    learned_variables = learn_parameters(variables, data)
    
    # Write the learned parameters to a new BIF file
    write_bif(learned_variables, "solved_hailfinder.bif")
    
    print("Parameter learning completed. Results saved to solved_hailfinder.bif")

if __name__ == "__main__":
    main()