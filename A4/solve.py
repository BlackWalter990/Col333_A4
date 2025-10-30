import re
import sys
import numpy as np
from startup_code import parse_bif

class Network:
    def __init__(self):
        self.vars = []  
        self.var_map = {}  
        
    def add_variable(self, var):
        self.vars.append(var)
        self.var_map[var.name] = var
        
    def get_variable(self, name):
        return self.var_map.get(name)
    
    def get_children(self, var_name):
        children = []
        for var in self.vars:
            if var_name in var.parents:
                children.append(var.name)
        return children

class Variable:
    def __init__(self, name, values):
        self.name = name
        self.values = values  
        self.parents = [] 
        self.cpt = None 
        self.is_known = None 
        
    def add_parent(self, parent_name):
        self.parents.append(parent_name)
        
    def initialize_cpt(self, num_entries):
        self.cpt = np.zeros(num_entries)
        self.is_known = np.zeros(num_entries, dtype=bool)

def build_network(bif_data):
    network = Network()
    
    # first pass: create all vars
    for var_name, var_info in bif_data.items():
        var = Variable(var_name, var_info["values"])
        network.add_variable(var)
    
    # second pass: set parents and CPTs
    for var_name, var_info in bif_data.items():
        var = network.get_variable(var_name)
        
        # set parents
        for parent_name in var_info["parents"]:
            var.add_parent(parent_name)
        
        # calculate CPT size
        parent_sizes = [len(network.get_variable(p).values) for p in var.parents]
        child_size = len(var.values)
        
        if parent_sizes:
            cpt_size = child_size * np.prod(parent_sizes)
        else:
            cpt_size = child_size
            
        var.initialize_cpt(cpt_size)
        
        # fill CPT values
        for key, probs in var_info["probabilities"].items():
            if key == "()":
                # no parents
                for i, prob in enumerate(probs):
                    var.cpt[i] = prob if prob != -1 else 0.0
                    var.is_known[i] = prob != -1
            else:
                # has parents - Convert string values to indices
                parent_value_strings = list(key) if isinstance(key, tuple) else [key]
                parent_indices = []
                
                # convert each parent value string to its index
                for parent_name, parent_val_str in zip(var.parents, parent_value_strings):
                    parent_var = network.get_variable(parent_name)
                    try:
                        parent_idx = parent_var.values.index(parent_val_str)
                    except ValueError:
                        # handle case where value might be quoted
                        parent_val_str = parent_val_str.strip('"')
                        parent_idx = parent_var.values.index(parent_val_str)
                    parent_indices.append(parent_idx)
                
                # calculate base index - fixed to be consistent
                base_idx = 0
                mult = 1
                for i in range(len(parent_indices) - 1, -1, -1):
                    base_idx += parent_indices[i] * mult
                    mult *= parent_sizes[i]
                
                # fill CPT
                for i, prob in enumerate(probs):
                    idx = base_idx * child_size + i
                    var.cpt[idx] = prob if prob != -1 else 0.0
                    var.is_known[idx] = prob != -1
    
    return network

def parse_data(filename, network):
    data = []
    missing_indices = [] 
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # split by coma or space
        tokens = re.split(r'[,\s]+', line)
        row = []
        missing_var_idx = None
        
        for var_idx, token in enumerate(tokens):
            var = network.vars[var_idx]
            
            # cclean the token by stripping whitespace and quotes
            cleaned_token = token.strip().strip('"')
            
            if cleaned_token == '?':
                row.append(-1)  
                missing_var_idx = var_idx
            else:
                # convert token to index in variable's values
                try:
                    val_idx = var.values.index(cleaned_token)
                    row.append(val_idx)
                except ValueError:
                    # if not found, try without stripping quotes (in case values are stored with quotes)
                    try:
                        val_idx = var.values.index(token)
                        row.append(val_idx)
                    except ValueError:
                        # if still not found, print an error and exit
                        print(f"Error: Value '{cleaned_token}' (original: '{token}') not found in variable '{var.name}'")
                        print(f"Valid values for {var.name} are: {var.values}")
                        sys.exit(1)
        
        data.append(row)
        if missing_var_idx is not None:
            missing_indices.append((line_idx, missing_var_idx))
    
    return data, missing_indices

def get_parent_indices(var, row, network):
    parent_indices = []
    for parent_name in var.parents:
        parent_var = network.get_variable(parent_name)
        parent_idx = network.vars.index(parent_var)
        parent_val = row[parent_idx]
        parent_indices.append(parent_val)
    return parent_indices

def calc_config_idx(parent_indices, parent_sizes):
    config = 0
    mult = 1
    for i in range(len(parent_indices) - 1, -1, -1):
        config += parent_indices[i] * mult
        mult *= parent_sizes[i]
    return config

def initialize_parameters(network):
    for var in network.vars:
        if not var.parents:
            # no parents, just normalize the distribution
            if not var.is_known.all():
                # random initialization for unknown values
                for i in range(len(var.cpt)):
                    if not var.is_known[i]:
                        var.cpt[i] = np.random.random()
                
                # normalize
                total = np.sum(var.cpt)
                if total > 0:
                    var.cpt = var.cpt / total
        else:
            # has parents, normalize for each parent configuration
            parent_sizes = [len(network.get_variable(p).values) for p in var.parents]
            num_parent_configs = np.prod(parent_sizes)
            child_size = len(var.values)
            
            for parent_config in range(num_parent_configs):
                start_idx = parent_config * child_size
                end_idx = start_idx + child_size
                
                # check if all values are known
                if np.all(var.is_known[start_idx:end_idx]):
                    continue
                
                # random initialization for unknown values
                for i in range(start_idx, end_idx):
                    if not var.is_known[i]:
                        var.cpt[i] = np.random.random()
                
                # nnormalize
                total = np.sum(var.cpt[start_idx:end_idx])
                if total > 0:
                    var.cpt[start_idx:end_idx] = var.cpt[start_idx:end_idx] / total

def em_algorithm(network, data, missing_indices, max_iter=100, epsilon=1e-6):
    # initialize parameters
    initialize_parameters(network)
    
    # Pprecompute network structure for efficiency
    var_to_idx = {var.name: i for i, var in enumerate(network.vars)}
    
    for iteration in range(max_iter):
        # estep: calculate expected counts
        counts = {}
        for var in network.vars:
            if not var.parents:
                counts[var.name] = np.zeros(len(var.values))
            else:
                parent_sizes = [len(network.get_variable(p).values) for p in var.parents]
                num_parent_configs = int(np.prod(parent_sizes))
                counts[var.name] = np.zeros((num_parent_configs, len(var.values)))
        
        # process complete data
        for row_idx, row in enumerate(data):
            if -1 not in row:
                # complete case
                for var in network.vars:
                    var_idx = network.vars.index(var)
                    child_val = row[var_idx]
                    
                    if not var.parents:
                        counts[var.name][child_val] += 1
                    else:
                        parent_indices = get_parent_indices(var, row, network)
                        parent_sizes = [len(network.get_variable(p).values) for p in var.parents]
                        parent_config = calc_config_idx(parent_indices, parent_sizes)
                        
                        counts[var.name][parent_config, child_val] += 1
        
        # process incomplete data with more sophisticated inference
        for row_idx, var_idx in missing_indices:
            row = data[row_idx]
            var = network.vars[var_idx]
            
            # calculate posterior distribution for the missing variable
            # using bayes rule: P(X|evidence) ∝ P(X|parents) * P(children|X, other_parents)
            posterior = np.zeros(len(var.values))
            
            for child_val in range(len(var.values)):
                # create a temporary row with the missing value filled
                temp_row = row.copy()
                temp_row[var_idx] = child_val
                
                # prior: P(X=child_val | parents)
                if not var.parents:
                    p_x = var.cpt[child_val]
                else:
                    parent_indices = get_parent_indices(var, temp_row, network)
                    parent_sizes = [len(network.get_variable(p).values) for p in var.parents]
                    parent_config = calc_config_idx(parent_indices, parent_sizes)
                    
                    cpt_idx = parent_config * len(var.values) + child_val
                    p_x = var.cpt[cpt_idx]
                
                # qdd small epsilon to avoid zero probabilities causing issues
                p_x = max(p_x, 1e-10)
                
                # likelihood: calculate product of children probabilities
                children = network.get_children(var.name)
                p_children = 1.0
                
                for child_name in children:
                    child_var = network.get_variable(child_name)
                    child_idx = network.vars.index(child_var)
                    child_val_in_row = temp_row[child_idx]
                    
                    if child_val_in_row == -1:
                        continue 
                    
                    # get parent configuration for child
                    child_parent_indices = []
                    for parent_name in child_var.parents:
                        parent_var = network.get_variable(parent_name)
                        parent_idx = network.vars.index(parent_var)
                        parent_val = temp_row[parent_idx]
                        child_parent_indices.append(parent_val)
                    
                    # caclulate P(child | parents)
                    if not child_var.parents:
                        p_child = child_var.cpt[child_val_in_row]
                    else:
                        child_parent_sizes = [len(network.get_variable(p).values) for p in child_var.parents]
                        parent_config = calc_config_idx(child_parent_indices, child_parent_sizes)
                        
                        cpt_idx = parent_config * len(child_var.values) + child_val_in_row
                        p_child = child_var.cpt[cpt_idx]
                    
                    # add epsilon to avoid zero likelihood
                    p_child = max(p_child, 1e-10)
                    p_children *= p_child
                
                posterior[child_val] = p_x * p_children
            
            # nnormalize posterior with numerical stability check
            total = np.sum(posterior)
            if total > 1e-15:
                posterior = posterior / total
            else:
                # fallback to uniform if all probabilities are too small
                posterior = np.ones(len(var.values)) / len(var.values)
            
            # update counts for the missing variable with fractional values
            for child_val in range(len(var.values)):
                weight = posterior[child_val]
                
                if weight < 1e-10: 
                    continue
                
                if not var.parents:
                    counts[var.name][child_val] += weight
                else:
                    parent_indices = get_parent_indices(var, row, network)
                    parent_sizes = [len(network.get_variable(p).values) for p in var.parents]
                    parent_config = calc_config_idx(parent_indices, parent_sizes)
                    
                    counts[var.name][parent_config, child_val] += weight
            
            # update child counts using expected sufficient statistics
            children = network.get_children(var.name)
            for child_name in children:
                child_var = network.get_variable(child_name)
                child_idx = network.vars.index(child_var)
                child_val_in_row = row[child_idx]
                
                if child_val_in_row == -1:
                    continue  
                
                # weight each possible parent value by its posterior
                for possible_parent_val in range(len(var.values)):
                    weight = posterior[possible_parent_val]
                    
                    if weight < 1e-10:  
                        continue
                    
                    # geet parent configuration for child with hypothetical parent value
                    child_parent_indices = []
                    for parent_name in child_var.parents:
                        parent_var = network.get_variable(parent_name)
                        parent_idx = network.vars.index(parent_var)
                        if parent_name == var.name:
                            parent_val = possible_parent_val
                        else:
                            parent_val = row[parent_idx]
                        child_parent_indices.append(parent_val)
                    
                    # update child counts
                    if not child_var.parents:
                        counts[child_name][child_val_in_row] += weight
                    else:
                        child_parent_sizes = [len(network.get_variable(p).values) for p in child_var.parents]
                        parent_config = calc_config_idx(child_parent_indices, child_parent_sizes)
                        
                        counts[child_name][parent_config, child_val_in_row] += weight
        
        # mstep: update parameters with Laplace smoothing for robustness
        max_change = 0
        smoothing = 0.01 
        
        for var in network.vars:
            if not var.parents:
                smoothed_counts = counts[var.name] + smoothing
                total = np.sum(smoothed_counts)
                if total > 0:
                    new_cpt = smoothed_counts / total
                    for i in range(len(var.cpt)):
                        if not var.is_known[i]:
                            change = abs(var.cpt[i] - new_cpt[i])
                            max_change = max(max_change, change)
                            var.cpt[i] = new_cpt[i]
            else:
                # has parents
                parent_sizes = [len(network.get_variable(p).values) for p in var.parents]
                num_parent_configs = int(np.prod(parent_sizes))
                child_size = len(var.values)
                
                for parent_config in range(num_parent_configs):
                    # add smoothing to avoid zero probabilities
                    smoothed_counts = counts[var.name][parent_config] + smoothing
                    total = np.sum(smoothed_counts)
                    
                    if total > 0:
                        new_probs = smoothed_counts / total
                        for i in range(child_size):
                            idx = parent_config * child_size + i
                            if not var.is_known[idx]:
                                change = abs(var.cpt[idx] - new_probs[i])
                                max_change = max(max_change, change)
                                var.cpt[idx] = new_probs[i]
        
        # check for convergence with better criteria
        if iteration > 5 and max_change < epsilon:  
            break
    
    for var in network.vars:
        if not var.parents:
            total = np.sum(var.cpt)
            if total > 0:
                var.cpt = var.cpt / total
            else:
                var.cpt = np.ones(len(var.values)) / len(var.values)
        else:
            parent_sizes = [len(network.get_variable(p).values) for p in var.parents]
            num_parent_configs = int(np.prod(parent_sizes))
            child_size = len(var.values)
            
            for parent_config in range(num_parent_configs):
                start_idx = parent_config * child_size
                end_idx = start_idx + child_size
                
                total = np.sum(var.cpt[start_idx:end_idx])
                if total > 0:
                    var.cpt[start_idx:end_idx] = var.cpt[start_idx:end_idx] / total
                else:
                    var.cpt[start_idx:end_idx] = np.ones(child_size) / child_size
    
    return network

def write_bif(network, input_filename, output_filename):
    with open(input_filename, 'r') as f:
        content = f.read()
    
    # regex to find probability blocks and capture their content
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
            # no parents, simple table
            total = np.sum(var.cpt)
            if total > 0:
                normalized = var.cpt / total
            else:
                normalized = var.cpt

            new_body = "table "
            for p in normalized:
                new_body += f"{p:.4f} "
            new_body = new_body.strip() + " ;"
        else:
            # has parents, conditional table
            parent_sizes = [len(network.get_variable(p).values) for p in parents]
            num_parent_configs = np.prod(parent_sizes)
            child_size = len(var.values)
            
            for parent_config in range(num_parent_configs):
                start_idx = parent_config * child_size
                end_idx = start_idx + child_size
                
                # normalize this row
                total = np.sum(var.cpt[start_idx:end_idx])
                if total > 0:
                    normalized = var.cpt[start_idx:end_idx] / total
                else:
                    normalized = var.cpt[start_idx:end_idx]

                # convert parent_config to parent values
                parent_indices = []
                temp_config = parent_config
                mult = 1
                for i in range(len(parent_sizes)):
                    mult *= parent_sizes[i]
                
                for i in range(len(parent_sizes)):
                    mult //= parent_sizes[i]
                    parent_indices.append(temp_config // mult)
                    temp_config %= mult
                
                parent_values = []
                for i, parent_name in enumerate(parents):
                    parent_var = network.get_variable(parent_name)
                    parent_values.append(f'"{parent_var.values[parent_indices[i]]}"')
                
                new_body += f"({', '.join(parent_values)}) "
                
                probs_str = " ".join([f"{p:.4f}" for p in normalized])
                new_body += probs_str + " ;\n"
            
            new_body = new_body.strip()
        
        return f"{opening}{new_body}{closing}"
    
    new_content = prob_pattern.sub(replace_prob_table, content)
    
    with open(output_filename, 'w') as f:
        f.write(new_content)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python solve.py <hailfinder.bif> <records.dat>")
        sys.exit(1)
    
    bif_file = sys.argv[1]
    data_file = sys.argv[2]
    
    print("Parsing network...")
    bif_data = parse_bif(bif_file)
    network = build_network(bif_data)
    
    print("Parsing data...")
    data, missing_indices = parse_data(data_file, network)
    
    print(f"Network has {len(network.vars)} vars")
    print(f"Data has {len(data)} records with {len(missing_indices)} missing values")
    
    print("Running EM algorithm...")
    network = em_algorithm(network, data, missing_indices)
    
    print("Writing output...")
    write_bif(network, bif_file, "solved_hailfinder.bif")
    
    print("Done!")