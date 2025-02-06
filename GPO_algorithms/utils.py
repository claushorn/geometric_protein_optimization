import os
import numpy as np
import pickle
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

def get_wildtype_smile(complexName): 
    file_name = "{complexName}.yaml"
    with open(file_name) as f:
        cfg = yaml.safe_load(f)
        return cfg['wildtype'], cfg['smile']

def get_reward(wildtype,smile):
    if config['scoring_function'] == 'BIND':
        return dockq_score(wildtype,smile)
    # Todo
    return

def load_or_run(pkl_file, func, *args):
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)
    else:
        result = func(*args)
        with open(pkl_file, 'wb') as f:
            pickle.dump(result, f)
        return result

def mutation_representation(sequence,wildtype):
    return ",".join([str(i) for i in range(len(sequence)) if sequence[i] != wildtype[i]]) 

def print_variant(sequence, wildtype):
    s = ""
    for i in range(len(sequence)):
        if sequence[i] != wildtype[i]:
            s += f"\033[91m{sequence[i]}\033[0m" # red letter
        else:
            s += sequence[i]
    print(s)

def affinity_to_pKd(affinity):
    return -np.log10(1/affinity - 1)

def get_fold_increase(sequence_reward, wildtype_reward):
    value_pKd = -np.log10(1/sequence_reward - 1)
    start_value_pKd = -np.log10(1/wildtype_reward - 1)
    fold_increase = 10**(value_pKd - start_value_pKd)
    return fold_increase 

def get_deltapKd(sequence_reward, wildtype_reward):
    value_pKd = -np.log10(1/sequence_reward - 1)
    start_value_pKd = -np.log10(1/wildtype_reward - 1)
    deltapKd = (value_pKd - start_value_pKd)
    return deltapKd 

def print_status(sequence,iStep, wildtype, smile, wildtype_reward, total_importance=None, outfile=None):
    reward = get_reward(sequence,smile)
    s = ""
    if total_importance is not None:
        new_importance = reward - wildtype_reward
        effect_remaining = 100*new_importance/total_importance
        s = f", effect_remaining: {effect_remaining:.2f}%"
    #
    names = ",".join([wildtype[i]+str(i+1) for i in range(len(sequence)) if sequence[i] != wildtype[i]]) # should be i+1 for official name, starting at 1, adapted 25Nov
    #
    fold_increase = get_fold_increase(reward, wildtype_reward)
    #
    nmutations = sum([1 for i in range(len(sequence)) if sequence[i] != wildtype[i]])
    print(f"step {iStep}: #mutations={nmutations}{s}, fold_increase: {fold_increase:5.1f} ({names})")
    if outfile is not None:
        with open(outfile, 'a') as f:
            f.write(f"{iStep},{nmutations},{reward},{fold_increase}\n")
    #print(f"sequence: {sequence}")
    print_variant(sequence, wildtype)
    return fold_increase
