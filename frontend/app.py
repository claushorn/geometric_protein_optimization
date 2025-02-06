#from re import L
import panel as pn  # GUI
#import datetime
#import random
import matplotlib.pyplot as plt
#import pickle
#import numpy as np
#import pandas as pd
import os
#from regex import R
import yaml
import glob
from buildup import run_buildup

pn.extension('ipywidgets') # for matplotlib 
#pn.extension('tabulator')


text_FAQ = """
#### How many choices should I make?

As many as you like. The more the better.

#### What if I am done?

You can just close the page. Your answers are saved after each choice.
"""

display_FAQ = pn.pane.Markdown(text_FAQ, width=100, min_height=377, sizing_mode='stretch_both')

intro_text = """
GPO is an AI-native approach that optimizes protein function by combining a large number of mutations from diverse locations alon the sequence to fine-tune the global geometry of the protein.

The GPO algorithm BuildUp performs a sequence of discrete gradient ascent steps to optimize a protein's function, applying one mutation at a time.

In this tool it is used for protein-ligand binding affinity optimization.

To start choose a protein-ligand complex and click fetch.
"""

protein = pn.widgets.StaticText(name='Protein', value='...')
ligand = pn.widgets.StaticText(name='Ligand', value='...')

txt_output = pn.widgets.StaticText(name='Optimized Variants', value='...')

def fetch_PLdata(complexName):
    # get data from the server
    return None

def get_PLdata(complexName): 
    " cache data locally "
    if os.path.exists(f"{complexName}.yaml"):
        with open(f"{complexName}.yaml") as f:
            cfg = yaml.safe_load(f)
            return cfg['wildtype'], cfg['smile']
    else:
        wildtype, smile = fetch_PLdata(complexName)
        if wildtype is not None and smile is not None:
            protein.value = wildtype
            ligand.value = smile
            with open(f"{complexName}.yaml", 'w') as f:
                yaml.dump({'wildtype': wildtype, 'smile': smile}, f)
        return wildtype, smile

lknownPLComplexes = [s.split()[0] for s in glob.glob('*.yaml')]


#info = pn.pane.Alert(intro_text, alert_type="warning", sizing_mode='stretch_width')

def make_plot(x, y):
    fig, ax = plt.figure()
    ax.plot(x, y)
    ax.set_title("BuildUp Results")
    return fig

buildUp_foldincrease = pn.pane.Matplotlib(make_plot([],[]), tight=True, format="svg", fixed_aspect=True, interactive=True,width=500, height=500, sizing_mode='stretch_both') # dpi=144, 

def update_results(results):
    # resilts = list of tuples (sequence, reward)
    x = [i for i in range(len(results))]
    y = [r[1] for r in results]
    buildUp_foldincrease.object = make_plot(x, y)
    #
    txt_output.value += f"\n {results[-1][0]}"
    return

settings = pn.Column(
    pn.widgets.Select(name='Select known PL-Complex', options=lknownPLComplexes),
    pn.Row(pn.widgets.TextInput(name='OR fetch from PDB', width=170, placeholder='', value="3ebp"), pn.widgets.Button(name="fetch", icon="", button_type="primary",align='center')),
)

buildup_info = """
Number of steps: Usually improvements are small after 30-40 steps.\n 
Number of mutations evaluated per step: 10 should give ok results; 30 is usually better, but slower.\n
Run time: scales linearly with 'number of steps' and 'number of mutations evaluated per step'.
"""

buildup_params = pn.Column(
    "### BuildUp Parameters",
    pn.widgets.IntInput(name='Number of steps', value=10),
    pn.widgets.IntInput(name='Number of mutations evaluated per step', value=10),
    buildup_info
)

button_run = pn.widgets.Button(name="Run BuildUp", icon="", button_type="primary",align='center')
button_abort = pn.widgets.Button(name="Abort run", icon="", button_type="primary",align='center')
controls = pn.Row(button_run, button_abort, align='center')

babort = False
def on_run(event):
    global babort
    babort = False
    for results in run_buildup(protein.value, ligand.value, buildup_params[1].value, buildup_params[0].value):
        update_results(results)
        if babort:
            break
    return 
button_run.on_click(on_run)

def on_abort(event):
    global babort
    babort = True
    return
button_abort.on_click(on_abort)

# Define the variables
wildtype = protein.value
smile = ligand.value
sizeMutationSet = buildup_params[1].value
nSteps = buildup_params[0].value

# pn.layout.Divider(), select_dataset
pn.template.FastGridTemplate(
    title="Geometric Protein Optimization - Welcome!",
    sidebar= pn.Column( "## Settings", settings, pn.layout.Divider(), buildup_params ), #, "### FAQ", display_FAQ, pn.layout.Divider(), "### Feedback", "Please let us know any suggestion for improvements you may have.", pn.widgets.TextAreaInput(name='', width=300, height=200, placeholder='...'), button_submitt_feedback), 
    main= pn.Column(intro_text, protein, ligand, controls),  #, pn.Row(option_pannelA, score_panelB, height=450), sizing_mode='stretch_both'), 
    accent = "blue",
    main_layout = None, 
    prevent_collision=True
).servable()

