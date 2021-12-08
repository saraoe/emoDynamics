# Emotion dynamics on tweets during COVID19

## Abstract

## Project Organization
The organization of the project is as follows:

```
├── README.md                  <- The top-level README for this project.
├── fig                        
├── idmdl                      <- csv-files with novelty/transience/resonance
│   └── ...
├── logs                       
├── newsFluxus                 <- the repo newsFluxus from CHCAA github
├── plot_notebooks             <- notebooks for plotting      
│   └── ...
├── src                        <- main scripts
│   └── ...
├── summarized_emo             <- ndjson-files with summarized scores of emotion distributions
│   └── ...
├──  requirement.txt            <- A requirements file of the required packages.
└──  run.sh                     <- bash script for reproducing results
```

## Pipeline

| DO | File| Output placement |
|-----------|:------------|:--------|
Run BERT models | Either using ```news_bert.py```, ```tweets_bert.py```, or with another script | ```../data/```
Summarize the emotion distributions | ```summarize_models.py``` | ```summarized_emo/```
Run newsFluxus pipeline | ```emotionsFluxus``` | ```idmdl/```

## Run the code
To reproduce the results clone this repository and run the following command
```
bash run.sh
```

*NB: This only runs ```emotionFluxus.py``` and not the rest of the pipeline, as the tweets used for this is not shared on git!* 

## Acknowledgments

Centre for Humanities Computing Aarhus for creating [newsFluxus](https://github.com/centre-for-humanities-computing/newsFluxus).