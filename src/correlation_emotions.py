'''
Calculate correlations between emotions
'''

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sn
import ndjson
from summarize_models import get_emotion_distribution

def get_emo(d: dict) -> list:
    '''
    returns BERT emotion distribution
    '''
    return d['emo_prob']

def evaluate_correlation(r:float, upper:float = 0.5, lower:float = -0.5):
    """Evaluate correlation direction and strength based on limits.
    r is considered a positive correlation if r>=upper, and negative if r <= lower

    Args:
        r (float): correlation coefficient
        upper (float, optional): Upper floor for positive correlations. Defaults to 0.5.
        lower (float, optional): Lower ceiling for negative correlations. Defaults to -0.5.

    Returns:
        text (str or bool): text describing correlation (if found), otherwise None
    """    
    if r >= upper and r<1:
        text = f"positively correlated: r = {r}"
    elif r <= lower:
        text = f'negatively correlated: r = {r}'
    else:
        text = None
    return text


if __name__ == "__main__":
    filename = os.path.join("summarized_emo", "tweets_recol_emo_date_sd.ndjson")
    with open(filename) as f:
        emo_file = ndjson.load(f)
    # df = pd.DataFrame(emo_file)

    labels = ["Glæde/Sindsro",
              "Tillid/Accept",
              "Forventning/Interrese",
              "Overasket/Målløs",
              "Vrede/Irritation",
              "Foragt/Modvilje",
              "Sorg/trist",
              "Frygt/Bekymret"]

    # Extract emotions and zip in lists
    emo_lists = list(map(get_emo, emo_file))
    zipped = list(zip(*emo_lists))

    # Add to df
    data = {labels[i]: zipped[i] for i in range(len(labels))}
    df = pd.DataFrame(data)

    # Calculate correlations
    corr_matrix = df.corr()

    # Show in heatmap
    sn.heatmap(corr_matrix, annot=True, cmap="PiYG")
    plt.show()

    corrs = set() # Set for storing already processed correlations
    limit = 0.5 # Assuming upper and lower limits are mirrors

    for r_i, row in corr_matrix.iterrows(): # Looping through rows
        for c_i, value in row.items(): # Looping through columns/values
            label_tuple = tuple(sorted((r_i, c_i)))
            text = evaluate_correlation(value, upper=limit, lower=-limit) 

            if text and label_tuple not in corrs: # Print not already printed correlations
                corrs.add(label_tuple)
                print(f'{r_i} and {c_i} are {text}')
    