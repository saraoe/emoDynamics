# Emotion dynamics on tweets during COVID19

## Abstract

## Project Organization
The organization of the project is as follows:

```
├── README.md                  <- The top-level README for this project.
├── fig                        
├── idmdl                      <- csv-files with novelty/transience/resonance
│   └── smoothed               <- csv-files with smoothed signal
├── logs                       
├── newsFluxus                 <- the repo newsFluxus from CHCAA github
├── notebooks                  <- notebooks for plotting      
│   └── ...
├── src                        <- main scripts
│   ├── tweets_bert.py
│   ├── summarize_models.py
│   ├── emotionFluxus.py
│   └── ...
├──  summarized_emo            <- ndjson-files with summarized scores of emotion distributions
├──  requirement.txt           <- A requirements file of the required packages.
└──  run.sh                    <- bash script for reproducing results
```

## Pipeline

| Do | File| Output placement |
|-----------|:------------|:--------|
Run BERT models | ```src/tweets_bert.py```  | ```../data/```
Summarize the emotion distributions | ```src/summarize_models.py``` | ```summarized_emo/```
Run newsFluxus pipeline | ```src/emotionsFluxus.py``` | ```idmdl/```
Smooth the signal | ```src/smoothing.py``` | ```idmdl/smoothed/```

To see what input the different scripts need run e.g. ```src/tweets_bert.py -h```.

## Run the code
To reproduce the results clone this repository and run the following command
```
bash run.sh
```

*NB: This only runs ```src/emotionFluxus.py``` and ```src/smoothing.py``` but not the rest of the pipeline, as the tweets used for this is not shared on git!* 

## Acknowledgments

Centre for Humanities Computing Aarhus for creating [newsFluxus](https://github.com/centre-for-humanities-computing/newsFluxus).