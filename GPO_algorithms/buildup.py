import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from utils import load_or_run, get_wildtype_smile, get_reward, print_status, mutation_representation, get_fold_increase, get_deltapKd
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

complexName = config['protein']['complex']

def process_importancesAll(importancesAll):
    maxImportancesAll = []
    mAAperSite = []
    for mimportances in importancesAll:
        maxI = max([e[1] for e in mimportances])
        maxImportancesAll.append(maxI) 
        mAAperSite.append( [aa for aa,I in mimportances if I==maxI][0] )
    return maxImportancesAll, mAAperSite

def select_topk_aas_perSite(lsites_highMagnitude, importances, k):
    lsitemutatoins = []
    for isite in lsites_highMagnitude:
        l = [(aa,importance) for aa,importance in importances[isite]]
        l = sorted(l, key=lambda x: x[1], reverse=True)
        l = l[:k]
        lsitemutatoins.extend([(isite,aa) for aa,_ in l])
    return lsitemutatoins

def select_sites(importances,lsites,n, lsitesToExclude=[], n_probes=None): # lsites gives the site index , much match importances
    lmax_importances = [(lsites[i],max([abs(importance) for aa,importance in mimportance])) for i,mimportance in enumerate(importances)]  
    if lsitesToExclude:
        lmax_importances = [(isite,importance) for isite,importance in lmax_importances if isite not in lsitesToExclude]
    l = sorted(lmax_importances, key=lambda x: x[1], reverse=True)
    l1 = l[:n]
    if n_probes is not None:
        # l2 = l[(n+1):(n+1+n_probes)] # take the highest, but this may be biased
        l2 = random.sample(l[(n+1):], n_probes) # instead randomly select n_probes from the rest
    else:
        l2 = []
    return [isite for isite,_ in l1], [isite for isite,_ in l2]


def get_initial_importances_sites(sequence, lsites, sequence_reward, smile, total_importance=None): # if we add one mutation
    print(f"get_initial_importances_sites: calculating importance for {len(lsites)*20} mutations")
    importances = []
    for i in lsites:
        mimportances = []
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            if aa != sequence[i]:
                new_sequence = sequence[:i] + aa + sequence[i+1:]
                importance = get_reward(new_sequence,smile) - sequence_reward
                if total_importance is not None:
                    importance = 100*importance/total_importance 
                mimportances.append( (aa, importance) ) 
        mimportances = sorted(mimportances, key=lambda x: x[1], reverse=True)
        importances.append(mimportances)
    return importances


def get_initial_importances_siteMutations(sequence, lsiteMutations, reference_reward, smile, total_importance, asFoldIncrease=False, asDeltapKd=False):
    importances = []
    for isite,aa in lsiteMutations:
        #print("get_initial_importances_siteMutations: site,aa",isite,aa)
        new_sequence = sequence[:isite] + aa + sequence[isite+1:]
        reward = get_reward(new_sequence,smile)
        if asFoldIncrease:
            importance = get_fold_increase(reward, reference_reward)
        elif asDeltapKd:
            importance = get_deltapKd(reward, reference_reward)
        else:
            importance = reward - reference_reward
            if total_importance is not None:
                importance = 100*importance/total_importance 
        importances.append( (isite,aa,importance) )
    return importances


def get_initial_importances2D_siteMutations(sequence, lsitemutatoins, reference_reward, smile, total_importance=None, asFoldIncrease=False):
    importances = []
    for isite,aai in lsitemutatoins:
        for jsite,aaj in lsitemutatoins:
            if isite >= jsite:
                continue
            new_sequence = sequence[:isite] + aai + sequence[isite+1:jsite] + aaj + sequence[jsite+1:]
            reward = get_reward(new_sequence,smile)
            if asFoldIncrease:
                importance = get_fold_increase(reward, reference_reward)
            else:
                importance = reward - reference_reward
                if total_importance is not None:
                    importance = 100*importance/total_importance 
            importances.append( (isite,aai,jsite,aaj,importance) )
    return importances


def iterative_buildup_siteMutations_continue(lsitemutatoins, wildtype, smile, maxImportancesAll, mAAperSite, sizeMutationSet, nSteps, brefine_sitemutations=True, lprobeSites=None): 
    backMutations = True
    do2D = False
    updateImportances = True
    aaFineTune = False
    banaylze_alternativeAAs = True

    wildtype_reward = get_reward(wildtype,smile)
    #total_importance = topsequence_reward - wildtype_reward
    total_importance = None # will omit printing effect_remaining, in print_status()

    print(f"Will continue adding mutations.")

    #lsitesAll = [i for i in range(len(wildtype))]
    sequence = wildtype
    lnmutations = []
    lfoldincrease = []
    lsequences = []
    #lapplied = []
    ltried = set() 
    for iStep in range(nSteps): # go through all of WT !
        sequence_reward = get_reward(sequence,smile)
        if do2D:
            importances = get_initial_importances2D_siteMutations(sequence, lsitemutatoins, sequence_reward, smile, total_importance)
            i = np.argsort([t[4] for t in importances])[-1] # choose highest importance, topk=1
            #print("choosen importance rank:",i,"max importance",importances[i][4])
        else:
            importances = get_initial_importances_siteMutations(sequence, lsitemutatoins, sequence_reward, smile, total_importance)
            i = np.argsort([t[2] for t in importances])[-1] # choose highest importance, topk=1
            #print("choosen importance rank:",i,"max importance",importances[i][2])
            if updateImportances:
                for isite,_,importance in importances:
                    maxImportancesAll[isite] = importance

        if lprobeSites is not None:
            lsitemutatoins_probes = [(i,aa) for i in lprobeSites for aa in 'ACDEFGHIKLMNPQRSTVWY']
            importances_probeSites = get_initial_importances_siteMutations(sequence, lsitemutatoins_probes, sequence_reward, smile, total_importance, asFoldIncrease=False)
            # append importances to a csv file
            file = 'importances_probeSites.csv'
            importances_with_n_mutations = [(iStep+1, isite, aa, importance) for isite, aa, importance in importances_probeSites]
            pd.DataFrame(importances_with_n_mutations, columns=['n_mutations','isite', 'aa', 'importance']).to_csv(file, mode='a', header=not os.path.isfile(file), index=False)

        if do2D:
            isite = importances[i][0]
            aai = importances[i][1]
            jsite = importances[i][2]
            aaj = importances[i][3]
            # adding TWO mutations !!!!
            sequence = sequence[:isite] + aai + sequence[isite+1:jsite] + aaj + sequence[jsite+1:]
        else:
            sequence = sequence[:lsitemutatoins[i][0]] + importances[i][1] + sequence[lsitemutatoins[i][0]+1:]

        if banaylze_alternativeAAs:
            importancesOtherAA = get_initial_importances_siteMutations(sequence, [(lsitemutatoins[i][0],aa) for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa!=lsitemutatoins[i][1]], get_reward(sequence,smile), smile, None, asFoldIncrease=False, asDeltapKd=True)
            with open(f"importancesOtherAA_{complexName}.txt", 'a') as f:
                f.write(str(iStep+1)+","+",".join([f"{t[2]:.6f}" for t in importancesOtherAA])+"\n")

        fold_increase = print_status(sequence, iStep+1, wildtype, smile, wildtype_reward, total_importance)
        lnmutations.append(iStep+1)
        lfoldincrease.append(fold_increase)
        lsequences.append(sequence)
        ltried.add( mutation_representation(sequence,wildtype) )

        # append to csv
        sfile = f"foldincrease_vs_steps_{complexName}_highImportanceContinue_{sizeMutationSet}.csv"
        df = pd.DataFrame({'nmutations':lnmutations,'foldincrease':lfoldincrease, 'sequence':lsequences})
        df.to_csv(sfile)

        if fold_increase == max(lfoldincrease):
            with open(f"topsequence_{complexName}_highImportanceContinue_{sizeMutationSet}.txt", 'w') as f:
                f.write(sequence)

        if brefine_sitemutations:
            # New here: for next iteration, redefine lsitemutatoins !
            lsites_notChoosenYet = [i for i in range(len(wildtype)) if sequence[i] == wildtype[i]]
            lsites_highMagnitude = sorted([(i,maxImportancesAll[i]) for i in lsites_notChoosenYet], key=lambda x: x[1], reverse=True)[:sizeMutationSet]
            lsitemutatoins = [(i,mAAperSite[i]) for i,_ in lsites_highMagnitude]
            #
            #topk = 1
            #lsites_notChoosenYet = [i for i in lsitesAll if i not in lapplied]
            #limportancesAll_notChoosenYet = [e for i,e in enumerate(importancesAll) if i not in lapplied]
            #lsites_highMagnitude, _ = select_sites( limportancesAll_notChoosenYet, lsites_notChoosenYet, sizeMutationSet//topk )
            #lsitemutatoins = select_topk_aas_perSite( lsites_highMagnitude, importancesAll, k=topk)
            if backMutations:
                #lbackMutations = [(i,aa) for i,aa in enumerate(wildtype) if i in lapplied]
                lbackMutations = [(i,aa) for i,aa in enumerate(wildtype) if sequence[i] != wildtype[i]]
                for i,aa in lbackMutations: # avoid infinite loop !!! by going back to seq on trajectory
                    if mutation_representation(sequence[:i] + aa + sequence[i+1:],wildtype) not in ltried:
                        lsitemutatoins.append( (i,aa) )

            if aaFineTune and (iStep+1) % 10 == 0:
                aa_finetune(sequence, wildtype, smile)

    return lnmutations, lfoldincrease


def aa_finetune(sequence, wildtype, smile):
    print(f"aa_finetune on {sum([sequence[i] != wildtype[i] for i in range(len(sequence))])} sites")
    sequence_reward = get_reward(sequence,smile)

    lsites = [i for i in range(len(wildtype)) if sequence[i] != wildtype[i]]

    # 1) loock at sites independently
    mAAperSite = [np.nan]*len(wildtype)
    maxImportancesAll = [-999]*len(wildtype)
    lsitemutatoins = []
    lnumPotentialAAs = []
    for isite in lsites:
        lrewards = []
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            if aa == sequence[isite]:
                lrewards.append((aa,sequence_reward))
                continue
            new_sequence = sequence[:isite] + aa + sequence[isite+1:]
            new_sequence_reward = get_reward(new_sequence,smile)
            lrewards.append((aa,new_sequence_reward))
        lrewards = sorted(lrewards, key=lambda x: x[1], reverse=True)
        print(f"site {isite}", [f"{100*(r-sequence_reward)/sequence_reward:4.2f}" for aa,r in lrewards])
        if lrewards[0][1] > sequence_reward:
            lsitemutatoins.append( (isite,lrewards[0][0]) )
            lnumPotentialAAs.append(sum([r > sequence_reward for aa,r in lrewards]))
            maxImportancesAll[isite] = lrewards[0][1] - sequence_reward
            mAAperSite[isite] = lrewards[0][0]

    print("lsitemutatoins",lsitemutatoins)
    if len(lsitemutatoins)>0:
        lnmutations,lfoldincrease = iterative_buildup_siteMutations_continue(lsitemutatoins, sequence, smile, maxImportancesAll, mAAperSite, len(lsitemutatoins), len(lsitemutatoins), brefine_sitemutations=False)
        with open(f"aafinetunePotential2_{complexName}.txt", 'a') as f:
            #f.write(f"{len(lsites)},{len(lsitemutatoins)},{max([100*importance/sequence_reward for importance in maxImportancesAll])},{max(lfoldincrease)}\n")
            f.write(f"{len(lsites)},{len(lsitemutatoins)},{sum(lnumPotentialAAs)},{max([100*importance/sequence_reward for importance in maxImportancesAll])},{max(lfoldincrease)}\n")
    print("aa_finetune done")
    return 


def foldevolution_highImportance(nmutations_goal, sizeMutationSet, complexName, lsites_preselected=None, lsitesToExclude=None): 
    start_time = time.time()   

    wildtype, smile = get_wildtype_smile(complexName) 

    # 2) highest WT-importances 
    print(f"n_mutations_goal = {nmutations_goal}.") # number of mutations to apply = number of steps
    sequence = wildtype
    lsites = [i for i in range(len(wildtype))]
    #total_importance = 1/100 # to get absolute value instead of percentage, BUG
    total_importance = None # Bug Fix, None == 100 
    wildtype_reward = get_reward(wildtype,smile)

    importances = load_or_run(f'WTimportances_{complexName}.pkl', get_initial_importances_sites, sequence, lsites, wildtype_reward, smile, total_importance)
    meanmagnitude = np.mean([abs(importance) for mimportance in importances for aa,importance in mimportance])
    print(f"meanmagnitude WTimportanceAll = {meanmagnitude}")
    if meanmagnitude > 1: # BUG fix, work around 
        print("Warning: importances are in percentage, not absolute values, will apply correction factor.")
        importances = [[(aa,importance/10000) for aa,importance in mimportance] for mimportance in importances]
    maxImportancesAll, mAAperSite = process_importancesAll(importances)

    n_probes = 0 # 6 # only for ananlysis
    topk = 1
    if lsites_preselected is None:
        lsites_highMagnitude, lsites_probes = select_sites(importances,lsites,sizeMutationSet//topk, lsitesToExclude, n_probes)
    else:
        lsites_highMagnitude = lsites_preselected
        lsites_probes = []

    lsitemutatoins = select_topk_aas_perSite(lsites_highMagnitude, importances, k=topk)
    lsitemutatoins_probes = [(isite,random.choice('ACDEFGHIKLMNPQRSTVWY')) for isite in lsites_probes] # instead take random aa 

    lnmutations,lfoldincrease = iterative_buildup_siteMutations_continue(lsitemutatoins, wildtype, smile, maxImportancesAll, mAAperSite, sizeMutationSet, nmutations_goal) # considers all AAs 
    print(f"run time: {time.time()-start_time}")

    # save to csv ; Maybe not needed? already written in iterative_buildup_siteMutations_continue
    df = pd.DataFrame({'nmutations':lnmutations,'foldincrease':lfoldincrease})
    sfile = f"foldincrease_vs_steps_{complexName}_highImportance_{sizeMutationSet}.csv"
    if lsitesToExclude:
        sfile = sfile.replace('.csv','_exclude.csv')
    if lsites_preselected:
        sfile = sfile.replace('.csv','_preselected.csv')
    df.to_csv(sfile)

    # calculate importances for topsequence (used in analyze_siteVariance.py)
    with open(f"topsequence_{complexName}_highImportanceContinue_{sizeMutationSet}.txt", 'r') as f:
        topsequence = f.read()
    # Note: has to be deleted before running, to be created new !!
    load_or_run(f'TSimportances_{complexName}.pkl', get_initial_importances_sites, topsequence, lsites, wildtype_reward, smile, total_importance)

    # line plot
    fig, ax = plt.subplots()
    ax.plot(lnmutations, lfoldincrease)
    plt.show()
    return
