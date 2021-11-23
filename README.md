# Emotion dynamics on tweets during COVID19

## Project Organization
The organization of the project is as follows:

```
├── README.md                  <- The top-level README for this project.
├── fig                        
├── idmdl                      <- csv-files with novelty/transience/resonance
├── logs                       
├── newsFluxus                 <- the repo newsFluxus from CHCAA github
├── plot_notebooks             
│   ├── plot_emotion_distr.ipynb <- plots of each emotion from BERT emotion 
│   └── vis_emotionFLuxus.ipynb  <- plots of resonance and novelty
├── src 
│   ├── emotionFluxus.py       <- runs newsFluxus pipeline on the summarized emotion distributions
│   ├── news_bert.py           <- pipeline for running BERT models on news fronpages
│   ├── preprocess.py          <- functions for preprocessing tweets
│   ├── smoothing.py           <- for smoothing the emoFluxus signal
│   ├── tweets_bert.py         <- pipeline for running BERT models on tweets
│   ├── summarize_models.py    <- summarizes emotion distributions (e.g. date/hour/paper)
│   ├── tensor_fix.py          <- generator function for splitting text into right size
│   └── tweets_bert.py         <- pipeline for running BERT models on tweets
├── summarized_emo             <- ndjson-files with summarized scores of emotion distributions
├── requirement.txt            <- A requirements file of the required packages.
├── count_emo.py               <- script for counting the emotional tweets
└── tab_nontab_split.py        <- script for splitting the news into tabloid and non-tabloid
```

## Pipeline

| DO | File|
|-----------|:-----------|
Run BERT models | Either from ```news_bert.py```, ```tweets_bert.py```, or from previous code
Summarize the emotion distributions | ```summarize_models.py```
Run newsFluxus pipeline | ```emotionsFluxus```





*Notes:* How many lines are in each of the 004_twitter-stopwords folder
|file   | n_lines      | size
|-------|--------------|---------|
part 1  |  10.372.005  |   30G
part 2  |  13.101.766  |   38G
part 3  |   2.907.322  |   8.5G