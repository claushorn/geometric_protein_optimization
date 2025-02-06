import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.helpers.test_inf_BIND import get_reward
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig


# 1. we start with the wildtype
config_file = "conf_default.yaml" # located in /usw/rl-enzyme-engineering/
hydra.initialize(config_path="..") # must be relative to this files directory, not cwd!
cfg = hydra.compose(config_name=config_file.split('.')[0])
wildtype, smile = cfg.experiment.wildtype_AA_seq, cfg.experiment.ligand_smile

out_file = "random_mutations.csv"
sequence = wildtype
max_reward = 0
for i in tqdm(range(1000)):
    # get reward
    reward = get_reward(sequence, smile)
    max_reward = max(max_reward, reward)
    # save to file
    with open(out_file, 'a') as f:
        f.write(f"{i},{reward},{max_reward}\n")
    # next mutation  
    i = np.random.randint(0, len(sequence))
    aa = np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'))
    sequence = sequence[:i] + aa + sequence[i+1:]
    pass


