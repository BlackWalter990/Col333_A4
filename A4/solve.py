import re
import sys
import numpy as np
from startup_code import parse_bif

class Network:
    def __init__(self):
        self.variables = []  # List of Variable objects in order
        self.var_map = {}  # Map from variable name to Variable object
        
    def add_variable(self, var):
        self.variables.append(var)
        self.var_map[var.name] = var
        
    def get_variable(self, name):
        return self.var_map.get(name)
    
    def get_children(self, var_name):
        children = []
        for var in self.variables:
            if var_name in var.parents:
                children.append(var.name)
        return children

class Variable:
    def __init__(self, name, values):
        self.name = name
        self.values = values  # List of possible values
        self.parents = []  # List of parent variable names
        self.cpt = None  # Conditional Probability Table
        self.is_known = None  # Boolean array indicating if CPT entries are known
        
    def add_parent(self, parent_name):
        self.parents.append(parent_name)
        
    def initialize_cpt(self, num_entries):
        self.cpt = np.zeros(num_entries)
        self.is_known = np.zeros(num_entries, dtype=bool)

def build_network(bif_data):
    """Build Network object from parsed bif data"""
    network = Network()
    
    # First pass: create all variables
    for var_name, var_info in bif_data.items():
        var = Variable(var_name, var_info["values"])
        network.add_variable(var)
    
    # Second pass: set parents and CPTs
    for var_name, var_info in bif_data.items():
        var = network.get_variable(var_name)
        
        # Set parents
        for parent_name in var_info["parents"]:
            var.add_parent(parent_name)
        
        # Calculate CPT size
        parent_sizes = [len(network.get_variable(p).values) for p in var.parents]
        child_size = len(var.values)
        
        if parent_sizes:
            cpt_size = child_size * np.prod(parent_sizes)
        else:
            cpt_size = child_size
            
        var.initialize_cpt(cpt_size)
        
        # Fill CPT values
        for key, probs in var_info["probabilities"].items():
            if key == "()":
                # No parents
                for i, prob in enumerate(probs):
                    var.cpt[i] = prob if prob != -1 else 0.0
                    var.is_known[i] = prob != -1
            else:
                # Has parents - Convert string values to indices
                parent_value_strings = list(key) if isinstance(key, tuple) else [key]
                parent_indices = []
                
                # Convert each parent value string to its index
                for parent_name, parent_val_str in zip(var.parents, parent_value_strings):
                    parent_var = network.get_variable(parent_name)
                    try:
                        parent_idx = parent_var.values.index(parent_val_str)
                    except ValueError:
                        # Handle case where value might be quoted
                        parent_val_str = parent_val_str.strip('"')
                        parent_idx = parent_var.values.index(parent_val_str)
                    parent_indices.append(parent_idx)
                
                # Calculate base index
                base_idx = 0
                for i, idx in enumerate(parent_indices):
                    if i < len(parent_indices) - 1:
                        base_idx += idx * np.prod(parent_sizes[i+1:])
                    else:
                        base_idx += idx
                
                # Fill CPT
                for i, prob in enumerate(probs):
                    idx = base_idx * child_size + i
                    var.cpt[idx] = prob if prob != -1 else 0.0
                    var.is_known[idx] = prob != -1
    
    return network

def parse_data(filename, network):
    """Parse the data file"""
    data = []
    missing_indices = []  # List of tuples (row_idx, var_idx)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Split by comma or space
        tokens = re.split(r'[,\s]+', line)
        row = []
        missing_var_idx = None
        
        for var_idx, token in enumerate(tokens):
            var = network.variables[var_idx]
            
            # Clean the token by stripping whitespace and quotes
            cleaned_token = token.strip().strip('"')
            
            if cleaned_token == '?':
                row.append(-1)  # Special value for missing
                missing_var_idx = var_idx
            else:
                # Convert token to index in variable's values
                try:
                    val_idx = var.values.index(cleaned_token)
                    row.append(val_idx)
                except ValueError:
                    # If not found, try without stripping quotes (in case values are stored with quotes)
                    try:
                        val_idx = var.values.index(token)
                        row.append(val_idx)
                    except ValueError:
                        # If still not found, print an error and exit
                        print(f"Error: Value '{cleaned_token}' (original: '{token}') not found in variable '{var.name}'")
                        print(f"Valid values for {var.name} are: {var.values}")
                        sys.exit(1)
        
        data.append(row)
        if missing_var_idx is not None:
            missing_indices.append((line_idx, missing_var_idx))
    
    return data, missing_indices

def get_parent_indices(var, row, network):
    """Extract parent indices from a data row"""
    parent_indices = []
    for parent_name in var.parents:
        parent_var = network.get_variable(parent_name)
        parent_idx = network.variables.index(parent_var)
        parent_val = row[parent_idx]
        parent_indices.append(parent_val)
    return parent_indices

def initialize_parameters(network):
    """Initialize unknown parameters with random values and normalize"""
    for var in network.variables:
        if not var.parents:
            # No parents, just normalize the distribution
            if not var.is_known.all():
                # Random initialization for unknown values
                for i in range(len(var.cpt)):
                    if not var.is_known[i]:
                        var.cpt[i] = np.random.random()
                
                # Normalize
                total = np.sum(var.cpt)
                if total > 0:
                    var.cpt = var.cpt / total
        else:
            # Has parents, normalize for each parent configuration
            parent_sizes = [len(network.get_variable(p).values) for p in var.parents]
            num_parent_configs = np.prod(parent_sizes)
            child_size = len(var.values)
            
            for parent_config in range(num_parent_configs):
                start_idx = parent_config * child_size
                end_idx = start_idx + child_size
                
                # Check if all values are known
                if np.all(var.is_known[start_idx:end_idx]):
                    continue
                
                # Random initialization for unknown values
                for i in range(start_idx, end_idx):
                    if not var.is_known[i]:
                        var.cpt[i] = np.random.random()
                
                # Normalize
                total = np.sum(var.cpt[start_idx:end_idx])
                if total > 0:
                    var.cpt[start_idx:end_idx] = var.cpt[start_idx:end_idx] / total

def em_algorithm(network, data, missing_indices, max_iter=50, epsilon=1e-4):
    """EM algorithm to learn missing parameters"""
    # Initialize parameters
    initialize_parameters(network)
    
    for iteration in range(max_iter):
        # E-Step: Calculate expected counts
        counts = {}
        for var in network.variables:
            if not var.parents:
                counts[var.name] = np.zeros(len(var.values))
            else:
                parent_sizes = [len(network.get_variable(p).values) for p in var.parents]
                num_parent_configs = np.prod(parent_sizes)
                counts[var.name] = np.zeros((num_parent_configs, len(var.values)))
        
        # Process complete data
        for row_idx, row in enumerate(data):
            if -1 not in row:
                # Complete case
                for var in network.variables:
                    var_idx = network.variables.index(var)
                    child_val = row[var_idx]
                    
                    if not var.parents:
                        counts[var.name][child_val] += 1
                    else:
                        parent_indices = get_parent_indices(var, row, network)
                        parent_config = 0
                        
                        for i, parent_idx in enumerate(parent_indices):
                            if i < len(parent_indices) - 1:
                                parent_config += parent_idx * np.prod([len(network.get_variable(p).values) for p in var.parents[i+1:]])
                            else:
                                parent_config += parent_idx
                        
                        counts[var.name][parent_config, child_val] += 1
        
        # Process incomplete data
        for row_idx, var_idx in missing_indices:
            row = data[row_idx]
            var = network.variables[var_idx]
            
            # Calculate posterior distribution for the missing variable
            posterior = np.zeros(len(var.values))
            
            for child_val in range(len(var.values)):
                # Create a temporary row with the missing value filled
                temp_row = row.copy()
                temp_row[var_idx] = child_val
                
                # Calculate P(X=child_val | parents)
                if not var.parents:
                    p_x = var.cpt[child_val]
                else:
                    parent_indices = get_parent_indices(var, temp_row, network)
                    parent_config = 0
                    
                    for i, parent_idx in enumerate(parent_indices):
                        if i < len(parent_indices) - 1:
                            parent_config += parent_idx * np.prod([len(network.get_variable(p).values) for p in var.parents[i+1:]])
                        else:
                            parent_config += parent_idx
                    
                    p_x = var.cpt[parent_config * len(var.values) + child_val]
                
                # Calculate product of children probabilities
                children = network.get_children(var.name)
                p_children = 1.0
                
                for child_name in children:
                    child_var = network.get_variable(child_name)
                    child_idx = network.variables.index(child_var)
                    child_val_in_row = temp_row[child_idx]
                    
                    if child_val_in_row == -1:
                        continue  # Skip if child is also missing
                    
                    # Get parent configuration for child
                    child_parent_indices = []
                    for parent_name in child_var.parents:
                        parent_var = network.get_variable(parent_name)
                        parent_idx = network.variables.index(parent_var)
                        parent_val = temp_row[parent_idx]
                        child_parent_indices.append(parent_val)
                    
                    # Calculate P(child | parents)
                    if not child_var.parents:
                        p_child = child_var.cpt[child_val_in_row]
                    else:
                        parent_config = 0
                        for i, parent_idx in enumerate(child_parent_indices):
                            if i < len(child_parent_indices) - 1:
                                parent_config += parent_idx * np.prod([len(network.get_variable(p).values) for p in child_var.parents[i+1:]])
                            else:
                                parent_config += parent_idx
                        
                        p_child = child_var.cpt[parent_config * len(child_var.values) + child_val_in_row]
                    
                    p_children *= p_child
                
                posterior[child_val] = p_x * p_children
            
            # Normalize posterior
            total = np.sum(posterior)
            if total > 0:
                posterior = posterior / total
            
            # Update counts with fractional values
            for child_val in range(len(var.values)):
                weight = posterior[child_val]
                
                if not var.parents:
                    counts[var.name][child_val] += weight
                else:
                    parent_indices = get_parent_indices(var, row, network)
                    parent_config = 0
                    
                    for i, parent_idx in enumerate(parent_indices):
                        if i < len(parent_indices) - 1:
                            parent_config += parent_idx * np.prod([len(network.get_variable(p).values) for p in var.parents[i+1:]])
                        else:
                            parent_config += parent_idx
                    
                    counts[var.name][parent_config, child_val] += weight
                
                # Update counts for children
                children = network.get_children(var.name)
                for child_name in children:
                    child_var = network.get_variable(child_name)
                    child_idx = network.variables.index(child_var)
                    child_val_in_row = row[child_idx]
                    
                    if child_val_in_row == -1:
                        continue  # Skip if child is also missing
                    
                    # Get parent configuration for child
                    child_parent_indices = []
                    for parent_name in child_var.parents:
                        parent_var = network.get_variable(parent_name)
                        parent_idx = network.variables.index(parent_var)
                        if parent_name == var.name:
                            parent_val = child_val  # Use the current child_val
                        else:
                            parent_val = row[parent_idx]
                        child_parent_indices.append(parent_val)
                    
                    # Update child counts
                    if not child_var.parents:
                        counts[child_name][child_val_in_row] += weight
                    else:
                        parent_config = 0
                        for i, parent_idx in enumerate(child_parent_indices):
                            if i < len(child_parent_indices) - 1:
                                parent_config += parent_idx * np.prod([len(network.get_variable(p).values) for p in child_var.parents[i+1:]])
                            else:
                                parent_config += parent_idx
                        
                        counts[child_name][parent_config, child_val_in_row] += weight
        
        # M-Step: Update parameters
        max_change = 0
        for var in network.variables:
            if not var.parents:
                # No parents
                total = np.sum(counts[var.name])
                if total > 0:
                    new_cpt = counts[var.name] / total
                    for i in range(len(var.cpt)):
                        if not var.is_known[i]:
                            change = abs(var.cpt[i] - new_cpt[i])
                            max_change = max(max_change, change)
                            var.cpt[i] = new_cpt[i]
            else:
                # Has parents
                parent_sizes = [len(network.get_variable(p).values) for p in var.parents]
                num_parent_configs = np.prod(parent_sizes)
                child_size = len(var.values)
                
                for parent_config in range(num_parent_configs):
                    total = np.sum(counts[var.name][parent_config])
                    if total > 0:
                        new_probs = counts[var.name][parent_config] / total
                        for i in range(child_size):
                            idx = parent_config * child_size + i
                            if not var.is_known[idx]:
                                change = abs(var.cpt[idx] - new_probs[i])
                                max_change = max(max_change, change)
                                var.cpt[idx] = new_probs[i]
        
        # Check for convergence
        if max_change < epsilon:
            print(f"Converged after {iteration+1} iterations")
            break
    
    return network

def write_bif(network, input_filename, output_filename):
    """Write the network with learned parameters to a BIF file, preserving formatting."""
    with open(input_filename, 'r') as f:
        content = f.read()
    
    # Regex to find probability blocks and capture their content
    prob_pattern = re.compile(
        r'(probability\s*\(\s*([^\|\)]+?)\s*(?:\|\s*([^\)]+))?\s*\)\s*\{)(.*?)(\};)',
        re.DOTALL | re.MULTILINE
    )
    
    def replace_prob_table(match):
        opening = match.group(1)
        target = match.group(2).strip().strip('"')
        parents_str = match.group(3)
        body = match.group(4)
        closing = match.group(5)
        
        var = network.get_variable(target)
        if not var:
            return match.group(0)
        
        parents = [p.strip().strip('"') for p in parents_str.split(',')] if parents_str else []
        
        new_body = ""
        
        if not parents:
            # No parents, simple table
            probs_to_write = var.cpt.tolist()

            # Normalize just in case
            total = sum(probs_to_write)
            if total > 0:
                probs_to_write = [p / total for p in probs_to_write]
            else:
                probs_to_write = [1.0 / len(probs_to_write)] * len(probs_to_write)

            rounded_probs = [round(p, 4) for p in probs_to_write]
            sum_rounded = sum(rounded_probs)
            if sum_rounded != 1.0:
                delta = 1.0 - sum_rounded
                # Add delta to the largest probability
                max_prob_index = rounded_probs.index(max(rounded_probs))
                rounded_probs[max_prob_index] += delta

            new_body = " table " + " ".join([f"{p:.4f}" for p in rounded_probs]) + ";"
        else:
            # Has parents, conditional table
            parent_sizes = [len(network.get_variable(p).values) for p in parents]
            num_parent_configs = np.prod(parent_sizes)
            child_size = len(var.values)
            
            new_body_lines = []
            for parent_config in range(num_parent_configs):
                start_idx = parent_config * child_size
                end_idx = start_idx + child_size
                
                probs_to_write = var.cpt[start_idx:end_idx].tolist()

                # Normalize this row
                total = sum(probs_to_write)
                if total > 0:
                    probs_to_write = [p / total for p in probs_to_write]
                else:
                    probs_to_write = [1.0 / len(probs_to_write)] * len(probs_to_write)

                rounded_probs = [round(p, 4) for p in probs_to_write]
                sum_rounded = sum(rounded_probs)
                if sum_rounded != 1.0:
                    delta = 1.0 - sum_rounded
                    # Add delta to the largest probability
                    max_prob_index = rounded_probs.index(max(rounded_probs))
                    rounded_probs[max_prob_index] += delta

                # Convert parent_config to parent values
                parent_indices = []
                temp_config = parent_config
                for size in parent_sizes:
                    parent_indices.append(temp_config % size)
                    temp_config //= size
                
                parent_values = []
                for i, parent_name in enumerate(parents):
                    parent_var = network.get_variable(parent_name)
                    parent_values.append(f'\'{parent_var.values[parent_indices[i]]}\'')
                
                line = f"  ({', '.join(parent_values)}) "
                
                probs_str = " ".join([f"{p:.4f}" for p in rounded_probs])
                line += probs_str + " ;"
                new_body_lines.append(line)
            
            new_body = "\n".join(new_body_lines)
        
        return f"{opening}\n{new_body}\n{match.group(5)}"
    
    new_content = prob_pattern.sub(replace_prob_table, content)
    
    with open(output_filename, 'w') as f:
        f.write(new_content)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python solve.py <hailfinder.bif> <records.dat>")
        sys.exit(1)
    
    bif_file = sys.argv[1]
    data_file = sys.argv[2]
    
    # Phase 1: Setup and Parsing
    print("Parsing network...")
    bif_data = parse_bif(bif_file)
    network = build_network(bif_data)
    
    print("Parsing data...")
    data, missing_indices = parse_data(data_file, network)
    
    print(f"Network has {len(network.variables)} variables")
    print(f"Data has {len(data)} records with {len(missing_indices)} missing values")
    
    # Phase 2: Learning
    print("Running EM algorithm...")
    network = em_algorithm(network, data, missing_indices)
    
    # Phase 3: Output
    print("Writing output...")
    write_bif(network, bif_file, "solved_hailfinder.bif")
    
    print("Done!")
