#!/usr/bin/env python3
"""
Bayesian Network Parameter Learning using EM Algorithm
Handles missing data in observations and learns CPT parameters
"""

import sys
import re

class BayesNet:
    def __init__(self):
        self.variables = {}  # var_name -> list of values
        self.parents = {}    # var_name -> list of parent names
        self.cpts = {}       # var_name -> CPT list
        self.var_order = []  # order of variables as they appear
        self.original_content = ""  # Store original BIF content
        
    def parse_bif(self, filename):
        """Parse BIF file and extract structure and initial CPTs"""
        with open(filename, 'r') as f:
            self.original_content = f.read()
        
        content = self.original_content
        
        # Method 1: Try to find variable blocks with various formats
        # Remove comments first
        content_no_comments = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        
        # Find all variable declarations
        # Try multiple patterns to handle different BIF formats
        patterns = [
            # Pattern 1: variable "NAME" { ... }
            r'variable\s+"([^"]+)"\s*\{(.*?)\n\}',
            # Pattern 2: variable NAME { ... }
            r'variable\s+(\w+)\s*\{(.*?)\n\}',
        ]
        
        for pattern in patterns:
            var_blocks = list(re.finditer(pattern, content_no_comments, re.DOTALL))
            if var_blocks:
                print(f"Found {len(var_blocks)} variable blocks with pattern")
                break
        
        if not var_blocks:
            print("ERROR: Could not find variable declarations")
            # Try to show first 500 chars to debug
            print("First 500 chars of file:")
            print(content[:500])
            return
        
        for match in var_blocks:
            var_name = match.group(1)
            var_content = match.group(2)
            
            # Extract values from the type discrete declaration
            # Try multiple patterns for values
            values = None
            
            # Pattern 1: type discrete[N] { "val1" "val2" ... }
            values_match = re.search(r'type\s+discrete\s*\[\s*\d+\s*\]\s*\{([^}]+)\}', var_content)
            if values_match:
                values_str = values_match.group(1)
                values = re.findall(r'"([^"]+)"', values_str)
            
            # Pattern 2: type discrete[N] { val1, val2, ... }
            if not values:
                values_match = re.search(r'type\s+discrete\s*\[\s*\d+\s*\]\s*\{([^}]+)\}', var_content)
                if values_match:
                    values_str = values_match.group(1)
                    values = [v.strip().strip('"').strip(',') for v in values_str.split() if v.strip()]
            
            if values:
                self.variables[var_name] = values
                self.var_order.append(var_name)
        
        if len(self.variables) == 0:
            print("ERROR: No variables found! Check BIF file format.")
            return
        
        print(f"Parsed {len(self.variables)} variables")
        print(f"First few variables: {self.var_order[:5]}")
        
        # Extract probability tables
        # Try multiple patterns
        prob_patterns = [
            # Pattern 1: probability("VAR" "PARENT1" ...) { table ... ; }
            r'probability\s*\(\s*"([^"]+)"([^)]*)\)\s*\{([^}]+)\}',
            # Pattern 2: probability(VAR PARENT1 ...) { table ... ; }
            r'probability\s*\(\s*(\w+)([^)]*)\)\s*\{([^}]+)\}',
        ]
        
        for pattern in prob_patterns:
            prob_blocks = list(re.finditer(pattern, content_no_comments, re.DOTALL))
            if prob_blocks:
                print(f"Found {len(prob_blocks)} probability blocks")
                break
        
        for match in prob_blocks:
            var_name = match.group(1).strip('"')
            parents_str = match.group(2)
            prob_content = match.group(3)
            
            # Parse parents - handle both quoted and unquoted
            parents = re.findall(r'"([^"]+)"', parents_str)
            if not parents:
                # Try unquoted
                parents = [p.strip() for p in parents_str.split() if p.strip()]
            
            self.parents[var_name] = parents
            
            # Parse table values
            table_match = re.search(r'table\s+([^;]+);', prob_content)
            if table_match:
                table_str = table_match.group(1)
                # Extract all numbers (including -1)
                values = []
                for token in table_str.split():
                    token = token.strip().strip(',')
                    if token:
                        try:
                            values.append(float(token))
                        except ValueError:
                            continue
                self.cpts[var_name] = values
        
        print(f"Parsed {len(self.cpts)} CPTs")
    
    def write_bif(self, output_filename):
        """Write learned BIF file by replacing probability tables"""
        content = self.original_content
        
        # Replace all probability tables
        def replace_table(match):
            full_match = match.group(0)
            # Extract variable name (could be quoted or not)
            var_name_match = re.search(r'probability\s*\(\s*"?([^"\s]+)"?', full_match)
            if not var_name_match:
                return full_match
            
            var_name = var_name_match.group(1)
            
            if var_name not in self.cpts:
                return full_match
            
            # Find the table statement
            table_match = re.search(r'(table\s+)([^;]+)(;)', full_match)
            if not table_match:
                return full_match
            
            # Format new values to 4 decimal places
            new_values = ' '.join(f'{v:.4f}' for v in self.cpts[var_name])
            
            # Replace the table values
            new_match = full_match.replace(
                table_match.group(2),
                new_values
            )
            
            return new_match
        
        # Try both quoted and unquoted patterns
        patterns = [
            r'probability\s*\([^)]+\)\s*\{[^}]+\}',
        ]
        
        for pattern in patterns:
            new_content = re.sub(pattern, replace_table, content, flags=re.DOTALL)
            if new_content != content:
                content = new_content
                break
        
        with open(output_filename, 'w') as f:
            f.write(content)
    
    def get_cpt_index(self, var_name, values_dict):
        """Get CPT index for a variable given parent values"""
        parents = self.parents.get(var_name, [])
        var_values = self.variables[var_name]
        var_val = values_dict[var_name]
        
        if not parents:
            # No parents - direct index
            return var_values.index(var_val)
        
        # Calculate index based on parent configuration
        # CPT layout: for each parent configuration, list all values of the variable
        index = 0
        multiplier = len(var_values)
        
        # Iterate through parents in reverse order
        for parent in reversed(parents):
            parent_values = self.variables[parent]
            parent_val = values_dict[parent]
            parent_idx = parent_values.index(parent_val)
            index += parent_idx * multiplier
            multiplier *= len(parent_values)
        
        # Add variable's own value index
        index += var_values.index(var_val)
        return index
    
    def get_probability(self, var_name, var_val, parent_vals):
        """Get P(var=var_val | parents=parent_vals)"""
        parents = self.parents.get(var_name, [])
        
        # Build values dict
        values_dict = {var_name: var_val}
        for i, parent in enumerate(parents):
            values_dict[parent] = parent_vals[i]
        
        cpt_idx = self.get_cpt_index(var_name, values_dict)
        prob = self.cpts[var_name][cpt_idx]
        
        # Avoid zero probabilities for numerical stability
        return max(prob, 1e-10)


def load_data(filename):
    """Load data records from file"""
    records = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Handle both space and comma separated values
            # Remove commas and extra whitespace
            cleaned = line.replace(',', ' ')
            values = []
            
            # Extract tokens (may be quoted)
            tokens = cleaned.split()
            for token in tokens:
                token = token.strip().strip('"').strip("'")
                if token:
                    values.append(token)
            
            if values:
                records.append(values)
    
    return records


def initialize_cpts(bn):
    """Initialize unknown CPT values with uniform distribution"""
    initialized_count = 0
    
    for var_name in bn.variables:
        if var_name not in bn.cpts:
            continue
            
        cpt = bn.cpts[var_name]
        var_values = bn.variables[var_name]
        num_values = len(var_values)
        
        # Calculate number of parent configurations
        parents = bn.parents.get(var_name, [])
        num_parent_configs = 1
        for parent in parents:
            num_parent_configs *= len(bn.variables[parent])
        
        # Initialize unknown values (-1) with uniform distribution
        for config_idx in range(num_parent_configs):
            start_idx = config_idx * num_values
            
            # Check if this configuration needs initialization
            if start_idx < len(cpt):
                needs_init = any(
                    start_idx + i < len(cpt) and cpt[start_idx + i] < 0 
                    for i in range(num_values)
                )
                
                if needs_init:
                    uniform_prob = 1.0 / num_values
                    for i in range(num_values):
                        if start_idx + i < len(cpt) and cpt[start_idx + i] < 0:
                            cpt[start_idx + i] = uniform_prob
                            initialized_count += 1
    
    print(f"Initialized {initialized_count} unknown parameters")


def em_algorithm(bn, records, max_iterations=50, tolerance=1e-5):
    """EM algorithm for parameter learning with missing data"""
    
    for iteration in range(max_iterations):
        # Initialize counts for M-step
        counts = {}
        for var_name in bn.variables:
            if var_name not in bn.cpts:
                continue
                
            var_values = bn.variables[var_name]
            parents = bn.parents.get(var_name, [])
            
            # Total size of CPT
            cpt_size = len(var_values)
            for parent in parents:
                cpt_size *= len(bn.variables[parent])
            
            counts[var_name] = [0.0] * cpt_size
        
        # E-step: Process each record
        for record_idx, record in enumerate(records):
            if len(record) != len(bn.var_order):
                continue  # Skip malformed records
            
            # Find missing variable (if any)
            missing_var_idx = None
            for i, val in enumerate(record):
                if val == '?':
                    missing_var_idx = i
                    break
            
            if missing_var_idx is None:
                # Complete data - just count
                values_dict = {}
                for i, var_name in enumerate(bn.var_order):
                    values_dict[var_name] = record[i]
                
                # Update counts for all variables
                for var_name in bn.variables:
                    if var_name not in counts:
                        continue
                    try:
                        idx = bn.get_cpt_index(var_name, values_dict)
                        counts[var_name][idx] += 1.0
                    except (ValueError, IndexError, KeyError):
                        continue
            else:
                # Missing data - compute expected counts
                missing_var = bn.var_order[missing_var_idx]
                missing_var_values = bn.variables[missing_var]
                
                # Compute probability for each possible value of missing variable
                probs = []
                for val in missing_var_values:
                    # Create complete record
                    complete_record = record[:]
                    complete_record[missing_var_idx] = val
                    
                    # Compute joint probability
                    prob = 1.0
                    values_dict = {}
                    for i, var_name in enumerate(bn.var_order):
                        values_dict[var_name] = complete_record[i]
                    
                    for var_name in bn.var_order:
                        if var_name not in bn.cpts:
                            continue
                        try:
                            parents = bn.parents.get(var_name, [])
                            parent_vals = [values_dict[p] for p in parents]
                            var_val = values_dict[var_name]
                            prob *= bn.get_probability(var_name, var_val, parent_vals)
                        except (ValueError, IndexError, KeyError):
                            prob = 0.0
                            break
                    
                    probs.append(prob)
                
                # Normalize probabilities
                total_prob = sum(probs)
                if total_prob > 0:
                    probs = [p / total_prob for p in probs]
                else:
                    # Uniform if all probabilities are zero
                    probs = [1.0 / len(missing_var_values)] * len(missing_var_values)
                
                # Add expected counts
                for val_idx, val in enumerate(missing_var_values):
                    complete_record = record[:]
                    complete_record[missing_var_idx] = val
                    
                    values_dict = {}
                    for i, var_name in enumerate(bn.var_order):
                        values_dict[var_name] = complete_record[i]
                    
                    weight = probs[val_idx]
                    
                    for var_name in bn.variables:
                        if var_name not in counts:
                            continue
                        try:
                            idx = bn.get_cpt_index(var_name, values_dict)
                            counts[var_name][idx] += weight
                        except (ValueError, IndexError, KeyError):
                            continue
        
        # M-step: Update parameters
        max_change = 0.0
        for var_name in bn.variables:
            if var_name not in bn.cpts:
                continue
                
            var_values = bn.variables[var_name]
            num_values = len(var_values)
            parents = bn.parents.get(var_name, [])
            
            num_parent_configs = 1
            for parent in parents:
                num_parent_configs *= len(bn.variables[parent])
            
            # Normalize each parent configuration
            for config_idx in range(num_parent_configs):
                start_idx = config_idx * num_values
                
                # Get counts for this configuration
                config_counts = [counts[var_name][start_idx + i] for i in range(num_values)]
                total_count = sum(config_counts)
                
                # Update probabilities
                if total_count > 0:
                    for i in range(num_values):
                        old_prob = bn.cpts[var_name][start_idx + i]
                        new_prob = config_counts[i] / total_count
                        bn.cpts[var_name][start_idx + i] = new_prob
                        max_change = max(max_change, abs(new_prob - old_prob))
                else:
                    # No data for this configuration - keep uniform
                    uniform = 1.0 / num_values
                    for i in range(num_values):
                        bn.cpts[var_name][start_idx + i] = uniform
        
        # Check convergence
        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}: max change = {max_change:.8f}")
        
        if max_change < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break


def main():
    if len(sys.argv) != 3:
        print("Usage: python solve.py hailfinder.bif records.dat")
        sys.exit(1)
    
    bif_file = sys.argv[1]
    data_file = sys.argv[2]
    output_file = "solved_hailfinder.bif"
    
    print("Loading Bayesian Network...")
    bn = BayesNet()
    bn.parse_bif(bif_file)
    
    if len(bn.variables) == 0:
        print("ERROR: Failed to parse BIF file. Exiting.")
        sys.exit(1)
    
    print(f"Network has {len(bn.variables)} variables")
    
    print("Loading data records...")
    records = load_data(data_file)
    print(f"Loaded {len(records)} records")
    
    # Validate data
    if records and len(records[0]) != len(bn.var_order):
        print(f"Warning: Record length {len(records[0])} != number of variables {len(bn.var_order)}")
        print(f"First record has {len(records[0])} values, expected {len(bn.var_order)}")
    
    print("Initializing CPTs...")
    initialize_cpts(bn)
    
    print("Running EM algorithm...")
    em_algorithm(bn, records, max_iterations=50)
    
    print(f"Writing output to {output_file}...")
    bn.write_bif(output_file)
    
    print("Done!")


if __name__ == "__main__":
    main()