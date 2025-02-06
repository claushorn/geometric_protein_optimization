import os, sys
import time

from polars import count
from regex import F
from torch import lt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.helpers.test_inf_BIND import get_reward
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from analysis.iterative_pruning import affinity_to_pKd, get_fold_increase
from analysis.from_config import get_wildtype_smile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../analysis')))
from analysis.buildUp_compareMutationSets import get_variants, get_variants2D, mutation_representation, process_importancesAll
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_top_sequences(population, population_scores, n_top):
    return [sequence for _, sequence in sorted(zip(population_scores, population), reverse=True)[:n_top]]

def get_top_sequences2(population, population_scores, n_top, wildtype, igeneration, complexName, method):
    lsequences = []  
    for rank,(score,sequence) in enumerate(sorted(zip(population_scores, population), reverse=True)):  
        sline = f"{igeneration},{rank+1} {[i for i in range(len(sequence)) if sequence[i] != wildtype[i]]}, {score:.3f}"
        print(sline)
        ##
        with open(f"directedevolution_rankings_{method}_{complexName}_{timestamp}.json", 'a') as f:
            #f.write(sline+"\n")
            data = {
                "generation": igeneration,
                "rank": rank + 1,
                "mutations": [i for i in range(len(sequence)) if sequence[i] != wildtype[i]],
                "score": round(score, 5)
            }
            json.dump(data, f)
            f.write('\n')

        lsequences.append(sequence)
    return lsequences[:n_top]

def run_directed_evolution_agent(n_generations, wildtype, smile, out_file): # compare to agent
    max_reward = 0
    wildtype_reward = get_reward(wildtype, smile)
    population = [wildtype]
    population_scores = [get_reward(wildtype, smile)]
    iStep = 0
    for _ in tqdm(range(n_generations)):
        new_population_scores = []
        new_population = []
        for start_sequence in get_top_sequences(population, population_scores, cfg.frontier_buffer.n_top):
            for _1 in range( cfg.on_policy_trainer.epochs ): # start with same sequence several times 
                sequence = start_sequence    
                for _2 in range( cfg.on_policy_trainer.steps_per_epoch * cfg.on_policy_trainer.epochs ):
                    # new mutation  
                    i = np.random.randint(0, len(sequence))
                    aa = np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'))
                    sequence = sequence[:i] + aa + sequence[i+1:]
                    new_population.append(sequence)
                    # get reward
                    reward = get_reward(sequence, smile)
                    new_population_scores.append(reward)
                    # save to file
                    max_reward = max(max_reward, reward)
                    iStep += 1
                    with open(out_file, 'a') as f:
                        f.write(f"{iStep},{reward},{max_reward},{affinity_to_pKd(reward)},{get_fold_increase(reward, wildtype_reward)}\n")
        # update population
        population = new_population
        population_scores = new_population_scores
        pass
    return

def get_mutants_DE(start_sequence,descendents_per_sequence,nmutations_per_round, notused, notused2, notused3):
    for _1 in range( descendents_per_sequence ): 
        sequence = start_sequence
        for _2 in range( nmutations_per_round ): 
            # new mutation  
            i = np.random.randint(0, len(sequence))
            aa = np.random.choice(list('ACDEFGHIKLMNPQRSTVWY')) # always includes BackMutations!
            sequence = sequence[:i] + aa + sequence[i+1:]
            yield sequence        

def get_mutants_DEplusBaselineAlgo(start_sequence,descendents_per_sequence,nmutations_per_round, maxImportancesAll, mAAperSite, ltried):
    sizeMutationSet = descendents_per_sequence
    assert nmutations_per_round==1, "only 1 mutation per round supported"   
    # setting: descendents_per_sequence = sizeMutationSet
    lsequences = get_variants(start_sequence, sizeMutationSet, descendents_per_sequence, maxImportancesAll, mAAperSite, wildtype, smile, ltried)
    #lsequences = get_variants2D(start_sequence, sizeMutationSet, sizeMutationSet, maxImportancesAll, mAAperSite, wildtype, smile)
    #lsequences = [s for s in lsequences if s not in new_population]
    #lsequences = lsequences[:descendents_per_sequence]
    for sequence in lsequences:
        yield sequence

def run_directed_evolution_baseline(n_generations, wildtype, smile, out_file, fget_mutants, complexName, method): # compare to baseline algorithm
    max_reward = 0
    population = [wildtype]
    wildtype_reward = get_reward(wildtype, smile)
    population_scores = [wildtype_reward]
    nmutations_per_round = 1 # 3 # learv at 1!
    descendents_per_sequence = 3 # population_size//n_top # was 3 # how many are created
    n_top = 7 # was 10 # how many are selected
    population_size =  n_top * descendents_per_sequence 
    print(f"nmutations_per_round:{nmutations_per_round}, population_size:{population_size}, n_top:{n_top}, descendents_per_sequence:{descendents_per_sequence}")

    with open(out_file, 'w') as f: # start a new file
        f.write(f"igeneration,iStep,reward,max_reward,pKd,foldincrease\n")

    with open("directedevolution_rankings.json", 'w') as f:
        f.write("generation,rank,sites,score\n")

    if fget_mutants == get_mutants_DEplusBaselineAlgo:
        with open(f'WTimportances_{complexName}.pkl', 'rb') as f:
            importancesAll = pickle.load(f)
            maxImportancesAll,mAAperSite = process_importancesAll(importancesAll)
    else:
        importancesAll = None
        maxImportancesAll,mAAperSite = None,None

    ltried = [] 
    iStep = 0
    nmutations = 0
    lnmutations = []
    lfoldincrease = []
    for igeneration in tqdm(range(n_generations)):
        new_population_scores = []
        new_population = []
        reward_bestPerGen = 0
        for start_sequence in get_top_sequences(population, population_scores, n_top):
        #for start_sequence in get_top_sequences2(population, population_scores, n_top, wildtype, igeneration, complexName, method):
            #for iVariant,sequence in enumerate(fget_mutants(start_sequence,descendents_per_sequence,nmutations_per_round,importancesAll,new_population)):
            for iVariant,sequence in enumerate(fget_mutants(start_sequence,descendents_per_sequence,nmutations_per_round,maxImportancesAll,mAAperSite,ltried)):
                new_population.append(sequence)
                ltried.append( mutation_representation(sequence, wildtype) )
                # get reward
                reward = get_reward(sequence, smile)
                #print(f"iVariant:{iVariant}, reward:{reward}",[i for i in range(len(sequence)) if sequence[i] != wildtype[i]])
                new_population_scores.append(reward)
                # save to file
                max_reward = max(max_reward, reward) # overall
                reward_bestPerGen = max(reward_bestPerGen, reward)
                iStep += 1
        with open(out_file, 'a') as f: # only best of generation
            sresult = f"{igeneration},{iStep},{reward_bestPerGen},{max_reward},{affinity_to_pKd(reward_bestPerGen)},{get_fold_increase(reward_bestPerGen, wildtype_reward)}"
            f.write(sresult+"\n")
        print(sresult)
        # update population
        population = new_population
        population_scores = new_population_scores
        nmutations += nmutations_per_round
        lnmutations.append(nmutations)
        lfoldincrease.append(get_fold_increase(reward_bestPerGen, wildtype_reward))
    return lnmutations, lfoldincrease


#if __name__ == '__main__':
if False: # compare with agent
    # 1. we start with the wildtype
    config_file = "conf_default.yaml" # located in /usw/rl-enzyme-engineering/
    hydra.initialize(config_path="..") # must be relative to this files directory, not cwd!
    cfg = hydra.compose(config_name=config_file.split('.')[0])
    wildtype, smile = cfg.experiment.wildtype_AA_seq, cfg.experiment.ligand_smile
    #
    n_generations = 40
    complexName = 'default'
    out_file = f"out_directed_evolution_{complexName}_{timestamp}.csv"
    run_directed_evolution_agent(n_generations, wildtype, smile, out_file)

if False: # compare with baseline
    #complexName = '3ebp'
    #complexName = '3PRS'
    #complexName = '2BRB'
    #complexName = "1bcu" 
    #complexName = "3dxg" 
    #complexName = "1p1q"
    complexName = "1o0H"

    n_generations = 40
    out_file = f"out_directed_evolution_{complexName}_{timestamp}.csv"

    method = 'DE'
    #method = 'DEplusBaselineAlgo'
    #backMutations = True -> always in DE , for buildUp in analysis/buildUp_compareMutationSets.py
    print(f"complexName:{complexName}, method:{method}")

    sresults_file = f'lfoldincrease_array_{complexName}_{method}.pkl'
    if os.path.exists(sresults_file):
        with open(sresults_file, 'rb') as f:
            lfoldincrease_array = pickle.load(f)
    else:
        start_time = time.time()   
        print(f"start_time:{start_time}")

        wildtype, smile = get_wildtype_smile(complexName=complexName) 
        
        if method == 'DE':
            fget_mutants = get_mutants_DE
            n_runs = 5 # was 5
        elif method == 'DEplusBaselineAlgo':
            fget_mutants = get_mutants_DEplusBaselineAlgo
            n_runs = 1 # since deterministic 

        lnmutations_runs = []
        lfoldincrease_runs = []
        for _ in range(n_runs):
            lnmutations, lfoldincrease = run_directed_evolution_baseline(n_generations, wildtype, smile, out_file, fget_mutants, complexName, method)
            lnmutations_runs.append(lnmutations)
            lfoldincrease_runs.append(lfoldincrease)

        # Convert lfoldincrease_runs to a NumPy array
        lfoldincrease_array = np.array(lfoldincrease_runs)  # Shape: (n_runs, n_steps)
        # dummy lfoldincrease_array to pkl
        with open(sresults_file, 'wb') as f:
            pickle.dump(lfoldincrease_array, f)

        end_time = time.time()   
        print(f"end_time:{end_time}, total runtime:{end_time-start_time}")


    # Compute mean and standard deviation across runs
    mean_foldincrease = np.mean(lfoldincrease_array, axis=0)
    std_foldincrease = np.std(lfoldincrease_array, axis=0)

    # Assume lnmutations is the same across runs
    lnmutations = [i+1 for i in range(len(mean_foldincrease))]
    n_runs = len(lfoldincrease_array)

    #sns.set_theme(style="whitegrid")
    #df_all_data = pd.DataFrame(all_data)
    #sns.lineplot(x='step', y='contribution1D', data=df_all_data, hue='complexName',  marker='o')


    # Plot the mean and standard deviation
    plt.figure(figsize=(8, 8))

    if True: # matplotlib 
        plt.plot(lnmutations, mean_foldincrease, label='Mean Fold Increase')
        plt.fill_between(
            lnmutations,
            mean_foldincrease - std_foldincrease,
            mean_foldincrease + std_foldincrease,
            alpha=0.3,
            label=f'Standard Deviation of {n_runs} runs'
        )
    if False: # seaborn, Not better 
        # Create a DataFrame
        data = pd.DataFrame({
            'nmutations': np.tile(lnmutations, n_runs),
            'foldincrease': lfoldincrease_array.flatten(),
            'run': np.repeat(np.arange(n_runs), len(lnmutations))
        })

        if False: # show individual runs
            for run in range(n_runs):
                sns.lineplot(
                    x=lnmutations,
                    y=lfoldincrease_array[run],
                    label=f'Run {run+1}',
                    alpha=0.3
                )

        # Plot mean and std deviation
        sns.lineplot(
            data=data,
            x='nmutations',
            y='foldincrease',
            ci='sd',
            estimator='mean',
            color='blue',
            label='Mean Fold Increase'
        )

    plt.xlabel('Number of Mutations')
    plt.ylabel('Fold Increase')
    plt.title(complexName)
    plt.legend()
    plt.show()

def lfoldincrease_array_TO_cvs(sfile):
    # for plot of DE, fold_increase vs nmutations
    with open(sfile, 'rb') as f:
        lfoldincrease_array = pickle.load(f)
    if len(lfoldincrease_array.shape) == 1:
        print("only one run")
        return 
    for i in range(lfoldincrease_array.shape[0]):
        lfoldincrease = lfoldincrease_array[i]
        lnmutations = [i+1 for i in range(len(lfoldincrease))]
    # mean and std
    lfoldincrease = np.mean(lfoldincrease_array, axis=0)
    lstd = np.std(lfoldincrease_array, axis=0)
    #
    df = pd.DataFrame({'nmutations':lnmutations,'foldincrease':lfoldincrease,'std':lstd})
    name = sfile.split('lfoldincrease_array_')[1].split('.pkl')[0]
    df.to_csv(f"lfoldincrease_array_{name}.csv", index=False)  

#lfoldincrease_array_TO_cvs("lfoldincrease_array_2BRB_DE.pkl")
#lfoldincrease_array_TO_cvs("lfoldincrease_array_3PRS_DE.pkl")
#lfoldincrease_array_TO_cvs("lfoldincrease_array_3ebp_DE.pkl")
#lfoldincrease_array_TO_cvs("lfoldincrease_array_1bcu_DE.pkl")
#lfoldincrease_array_TO_cvs("lfoldincrease_array_1o0H_DE.pkl")
#lfoldincrease_array_TO_cvs("lfoldincrease_array_1p1q_DE.pkl")
#lfoldincrease_array_TO_cvs("lfoldincrease_array_3dxg_DE.pkl")


def analyse_topSeq_ranking(sfile):
    records = []
    with open(sfile, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:  # Skip empty lines
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue  # Skip lines that cannot be decoded
    # Process the records as needed
    for record in records:
        print(record)

    # get top sequence
    ngenerations = max([record['generation'] for record in records])
    top_seq_siteset = set([record for record in records if record['generation']==ngenerations and record['rank']==1][0]['mutations'])

    def is_subset(s1,s2):
        return all([c in s2 for c in s1])

    df = pd.DataFrame(records)
    lgeneration = []
    lrank = []
    for igeneration in range(len(records)):    
        dfT = df[df['generation']==igeneration]
        # loop though all sequences in dfT and check if they are a subset of top_seq 
        for i,row in dfT.iterrows():
            sites = set(row['mutations'])
            #sites = set([int(s) for s in row['sites'][1:-1].split(',')])
            if is_subset(sites,top_seq_siteset):
                print(f"generation:{igeneration}, rank:{row['rank']}") # , sites:{sites}
                lgeneration.append(igeneration)
                lrank.append(row['rank'])
                break
    
    ldeviations=[]
    deviation = 0
    for r in lrank:
        if r!=1:
            if deviation > 0:
                deviation += 1
            else:
                deviation = 1
        else: # r==1
            if deviation > 0:
                ldeviations.append(deviation)
                deviation = 0
            else:
                ldeviations.append(0)
    from collections import Counter
    print(Counter(ldeviations))

    # write csv
    df = pd.DataFrame({'generation':lgeneration,'rank':lrank})
    timestamp = sfile.split('directedevolution_rankings_')[1].split('.json')[0]
    df.to_csv(f"topSeq_ranking_{timestamp}.csv", index=False)

#analyse_topSeq_ranking("directedevolution_rankings_20241123_085653_3ebp_DE.json")
#analyse_topSeq_ranking("directedevolution_rankings_20241122_161312_3ebpMaybe.json") # is DEplusBaselineAlgo
#analyse_topSeq_ranking("directedevolution_rankings_DEplusBaselineAlgo_2BRB_20241123_101916.json")
#analyse_topSeq_ranking("directedevolution_rankings_DE_3ebp_20241123_125516.json")
#analyse_topSeq_ranking("directedevolution_rankings_DEplusBaselineAlgo_3ebp_20241123_202254.json")



