'''
Smoothing EmoDynamics signals
'''

import pandas as pd
import os
import sys
import argparse

#Code from newsFluxus Github. 
path = os.path.join("newsFluxus", "src")
sys.path.append(path)
import news_uncertainty


def main(filename:str, span:int):
    '''
    Reads in a csv with emoDynamics extracted. Smooths the signals
    Writes to csv and returns df
    '''
    filepath = os.path.join("idmdl", f"{filename}.csv")
    if not os.path.exists(os.path.join("idmdl", "smoothed")):
        os.makedirs(os.path.join("idmdl", "smoothed"))
    out_path = os.path.join("idmdl", "smoothed", f"{filename}_smoothed_{span}.csv")

    df = pd.read_csv(filepath)
    df["smoothed_transience"] = news_uncertainty.adaptive_filter(df["transience"], span=span)
    df["smoothed_novelty"] = news_uncertainty.adaptive_filter(df["novelty"], span=span)
    df["smoothed_resonance"] = news_uncertainty.adaptive_filter(df["resonance"], span=span)
    df.to_csv(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True,
                        help='Name of the file with novelty, transience, and resonance scores')
    parser.add_argument('--span', type=int, required=False, default=56,
                        help='The span used for smoothing')
    args = parser.parse_args()

    print(f'''Running smoothing.py with arguments:
            filename = {args.filename},
            span = {args.span}''')
    main(filename=args.filename, span=args.span)

