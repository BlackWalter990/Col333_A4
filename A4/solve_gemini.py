#!/usr/bin/env python3

"""
solve.py

This script implements the Expectation-Maximization (EM) algorithm to learn
the parameters (Conditional Probability Tables) of a Bayesian Network
from a dataset with missing values.

This version corrects two critical bugs from previous attempts:
1.  **Data Parser:** Now correctly splits data rows by comma (',')
    instead of spaces, resolving the 'malformed data row' warnings.
2.  **BIF Output:** Uses a new, more robust regular expression to
    find and replace the CPT tables, ensuring all learned -1
    values are correctly overwritten in the output file.

Algorithm:
1.  **Parsing:**
    -   Parses the .bif file to build the network graph structure
        (variables, values, parent-child relationships).
    -   Parses the .dat file (comma-separated) to load the weather records.
2.  **Initialization (Theta_0):**
    -   Initializes all CPTs with random, non-uniform probabilities.
        This provides a robust starting point.
3.  **Iteration (EM steps):**
    -   Repeats for a fixed number of iterations:
    -   **E-Step (Expectation):**
        -   Creates a new set of "expected counts" tables.
        -   Iterates through each data row:
            -   **Complete Row:** Adds 1.0 to the corresponding counts.
            -   **Incomplete Row (missing X_i):**
                -   Calculates the posterior probability
                    P(X_i | evidence, Theta_t) using the Markov Blanket.
                -   Distributes these fractional weights to the
                    expected counts tables.
    -   **M-Step (Maximization):**
        -   Updates the network CPTs (Theta_{t+1}) using the
            `expected_counts` from the E-step, applying
            Laplace (add-1) smoothing.
4.  **Output:**
    -   Writes the final learned CPTs to 'solved_hailfinder.bif',
        formatting all probabilities to four decimal places.
"""

import sys
import re
import itertools
import math
from copy import deepcopy
import random

def logsumexp(log_probs):
    """
    Numerically stable computation of log(sum(exp(log_probs))).
    """
    max_log = max(log_probs)
    if max_log == -float('inf'):
        return -float('inf')
    
    sum_exp = 0.0
    for lp in log_probs:
        sum_exp += math.exp(lp - max_log)
        
    return max_log + math.log(sum_exp)

class Variable:
    """
    Represents a node in the Bayesian Network.
    """
    def __init__(self, name, values):
        self.name = name
        self.values = values  # List of possible string values
        self.parents = []     # List of parent Variable objects
        self.children = []    # List of child Variable objects
        self.cpt = {}         # The Conditional Probability Table

    def __repr__(self):
        return f"Variable({self.name})"

class BayesNet:
    """
    Represents the Bayesian Network structure and parameters.
    """
    def __init__(self):
        self.variables = [] # Ordered list of variables
        self.var_map = {}   # Fast lookup map of {name: Variable_object}
        self.data = []      # List of data rows

    def parse_bif(self, filepath):
        """
        Parses the .bif file to build the network structure.
        Initializes CPTs but does not fill them.
        """
        with open(filepath, 'r') as f:
            content = f.read()

        # Regex to find variable blocks
        var_pattern = re.compile(
            r'variable "([^"]+)" \{\s*type discrete\[\d+\] \{ ([^}]+) \};\s*\}',
            re.MULTILINE
        )
        
        for match in var_pattern.finditer(content):
            name = match.group(1)
            values = [v.strip('"') for v in match.group(2).split()]
            var = Variable(name, values)
            self.variables.append(var)
            self.var_map[name] = var

        # Regex to find probability blocks
        prob_pattern = re.compile(
            r'probability\s*\(\s*"([^"]+)"([^)]*)\s*\)\s*\{([^}]+)\}',
            re.MULTILINE
        )
        parent_name_pattern = re.compile(r'"([^"]+)"')
        
        for match in prob_pattern.finditer(content):
            var_name = match.group(1)
            parent_names_str = match.group(2)
            var = self.var_map[var_name]
            parent_names = parent_name_pattern.findall(parent_names_str)
            
            for pname in parent_names:
                parent_var = self.var_map[pname]
                var.parents.append(parent_var)
                parent_var.children.append(var)
            
            # Initialize CPT structure
            parent_value_lists = [p.values for p in var.parents]
            all_parent_configs = itertools.product(*parent_value_lists)
            
            for config in all_parent_configs:
                var.cpt[config] = {val: 0.0 for val in var.values}
                
            if not var.parents:
                var.cpt[()] = {val: 0.0 for val in var.values}

    def parse_data(self, filepath):
        """
        Parses the data file (e.g., records.dat).
        
        *** FIX ***: Now splits by comma (',') and strips whitespace
        and quotes to handle the correct data format.
        """
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # BUG FIX: Split by comma, not space.
                # Strip whitespace, then strip quotes.
                values = [v.strip().strip('"') for v in line.split(',')]
                
                if len(values) == len(self.variables):
                    self.data.append(values)
                else:
                    # This warning should no longer appear with the fix
                    print(f"Warning: Skipping malformed data row: {line}", file=sys.stderr)

    def get_prob(self, var, var_value, parent_values_dict):
        """
        Gets a specific probability from the CPT.
        P(var=var_value | Parents=parent_values_dict)
        """
        try:
            pa_config = tuple(parent_values_dict[p.name] for p in var.parents)
            return var.cpt[pa_config][var_value]
        except KeyError:
            return 1e-9 # Return small probability for safety

    def _get_row_as_dict(self, row_list):
        """Converts a list of values to a {var_name: value} dict."""
        return {self.variables[i].name: row_list[i] for i in range(len(row_list))}

    def _add_counts(self, counts_dict, row_dict, weight):
        """
        Helper function to add (potentially fractional) counts to a
        counts dictionary during the E-step.
        """
        for var in self.variables:
            var_name = var.name
            var_val = row_dict[var_name]
            
            try:
                pa_config = tuple(row_dict[p.name] for p in var.parents)
            except KeyError:
                continue
                
            c_var = counts_dict.setdefault(var_name, {})
            c_config = c_var.setdefault(pa_config, {})
            c_config[var_val] = c_config.get(var_val, 0.0) + weight

    def initialize_params(self):
        """
        Initializes parameters with random, non-uniform probabilities
        to break symmetry and provide a good starting point for EM.
        """
        print("Initializing parameters with random values...", file=sys.stderr)
        
        for var in self.variables:
            parent_value_lists = [p.values for p in var.parents]
            all_parent_configs = itertools.product(*parent_value_lists)
            if not var.parents:
                 all_parent_configs = [()]
            
            for pa_config in all_parent_configs:
                random_probs = [random.random() for _ in var.values]
                total_prob = sum(random_probs)
                
                if total_prob == 0:
                    num_vals = len(var.values)
                    for i, val in enumerate(var.values):
                        var.cpt[pa_config][val] = 1.0 / num_vals
                else:
                    # Normalize random values to sum to 1
                    for i, val in enumerate(var.values):
                        var.cpt[pa_config][val] = random_probs[i] / total_prob
                        
        print("Random initialization complete.", file=sys.stderr)


    def e_step(self):
        """
        Performs the E-step of the EM algorithm.
        Calculates expected counts based on current parameters (self.cpt).
        """
        expected_counts = {}
        
        for row in self.data:
            try:
                miss_idx = row.index('?')
                missing_var = self.variables[miss_idx]
                row_dict = self._get_row_as_dict(row)
                
            except ValueError:
                # Case 1: Complete data row
                row_dict = self._get_row_as_dict(row)
                self._add_counts(expected_counts, row_dict, 1.0)
                continue

            # Case 2: Incomplete data row (with 1 missing var)
            log_unnormalized_probs = {}
            
            # P(X|e) \propto P(X | Parents(X)) * \prod_{C \in Children(X)} P(C | Parents(C))
            for val in missing_var.values:
                log_prob_v = 0.0
                
                # Term 1: P(X_i | Parents(X_i))
                pa_vals_i = {p.name: row_dict[p.name] for p in missing_var.parents}
                log_prob_v += math.log(self.get_prob(missing_var, val, pa_vals_i) + 1e-9)

                # Term 2: \prod P(Child_j | Parents(Child_j))
                for child_var in missing_var.children:
                    child_val = row_dict[child_var.name]
                    
                    pa_vals_child = {}
                    for p_child in child_var.parents:
                        if p_child.name == missing_var.name:
                            pa_vals_child[p_child.name] = val # Use hypothesized value
                        else:
                            pa_vals_child[p_child.name] = row_dict[p_child.name]
                            
                    log_prob_v += math.log(self.get_prob(child_var, child_val, pa_vals_child) + 1e-9)
                
                log_unnormalized_probs[val] = log_prob_v

            # Normalize to get weights
            log_probs_list = list(log_unnormalized_probs.values())
            log_total = logsumexp(log_probs_list)
            
            weights = {}
            if log_total == -float('inf'):
                num_vals = len(missing_var.values)
                weights = {val: 1.0 / num_vals for val in missing_var.values}
            else:
                for val, log_p in log_unnormalized_probs.items():
                    weights[val] = math.exp(log_p - log_total)
            
            # Add fractional counts based on weights
            for val, weight in weights.items():
                if weight == 0:
                    continue
                temp_row_dict = row_dict.copy()
                temp_row_dict[missing_var.name] = val
                self._add_counts(expected_counts, temp_row_dict, weight)

        return expected_counts

    def m_step(self, expected_counts):
        """
        Performs the M-step of the EM algorithm.
        Updates self.cpt based on expected_counts, using Laplace smoothing.
        """
        for var in self.variables:
            num_vals = len(var.values)
            
            parent_value_lists = [p.values for p in var.parents]
            all_parent_configs = itertools.product(*parent_value_lists)
            if not var.parents:
                 all_parent_configs = [()]
            
            for pa_config in all_parent_configs:
                counts_for_config = expected_counts.get(var.name, {})\
                                                   .get(pa_config, {})
                
                # Apply Laplace (add-1) smoothing
                total_count = sum(counts_for_config.values())
                denominator = total_count + num_vals
                
                for val in var.values:
                    numerator = counts_for_config.get(val, 0.0) + 1.0
                    new_prob = numerator / denominator
                    var.cpt[pa_config][val] = new_prob

    def train(self, max_iterations):
        """
        Runs the full EM training loop.
        """
        self.initialize_params()
        
        print(f"Starting EM for {max_iterations} iterations...", file=sys.stderr)
        for i in range(max_iterations):
            expected_counts = self.e_step()
            self.m_step(expected_counts)
            print(f"  Iteration {i+1}/{max_iterations} complete.", file=sys.stderr)
        print("Training complete.", file=sys.stderr)

    def output_bif(self, input_bif_path, output_bif_path):
        """
        Generates the solved .bif file by replacing 'table' lines
        with the learned CPT parameters.
        
        *** FIX ***: Uses a new, more robust regex that correctly
        finds all table blocks and replaces their values.
        """
        print(f"Writing solved network to {output_bif_path}...", file=sys.stderr)
        
        # BUG FIX: This new regex is more general.
        # It captures:
        # Group 1: The start of the block, up to and including 'table '
        #   Group 2: The variable name (nested inside group 1)
        # Group 3: The table values (e.g., "-1 -1")
        # Group 4: The closing semicolon and brace
        prob_pattern = re.compile(
            r'(probability\s*\(\s*"([^"]+)"[^)]*\)\s*\{[^}]*?table\s+)([^;]+)(;[^}]*\})',
            re.MULTILINE | re.DOTALL
        )
        
        with open(input_bif_path, 'r') as f_in:
            content = f_in.read()

        def replace_table(match):
            full_block_start = match.group(1)
            var_name = match.group(2)
            # original_table_vals = match.group(3) # This is the -1 -1 part
            full_block_end = match.group(4)
            
            var = self.var_map[var_name]
            
            # Output the table in the exact BIF order:
            # Iterate through var values *last*
            # Iterate through parent configs *first*
            new_table_entries = []
            
            parent_value_lists = [p.values for p in var.parents]
            all_parent_configs = list(itertools.product(*parent_value_lists))
            if not var.parents:
                 all_parent_configs = [()]
            
            for val in var.values:
                for pa_config in all_parent_configs:
                    prob = var.cpt[pa_config][val]
                    new_table_entries.append(f"{prob:.4f}")
            
            new_table_str = " ".join(new_table_entries)
            
            return f"{full_block_start}{new_table_str}{full_block_end}"

        solved_content = prob_pattern.sub(replace_table, content)
        
        with open(output_bif_path, 'w') as f_out:
            f_out.write(solved_content)
        
        print("Done.", file=sys.stderr)

def main():
    if len(sys.argv) != 3:
        print("Usage: ./run.sh <input.bif> <input.dat>")
        print(f"Example: {sys.argv[0]} hailfinder.bif records.dat")
        sys.exit(1)

    bif_input_file = sys.argv[1]
    data_input_file = sys.argv[2]
    bif_output_file = "solved_hailfinder.bif"
    
    # 10 iterations is a reasonable trade-off for speed/accuracy
    MAX_ITERATIONS = 10 
    
    net = BayesNet()
    
    net.parse_bif(bif_input_file)
    net.parse_data(data_input_file)
    net.train(max_iterations=MAX_ITERATIONS)
    net.output_bif(bif_input_file, bif_output_file)

if __name__ == "__main__":
    main()