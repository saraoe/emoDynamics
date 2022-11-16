pip install -r requirements.txt
git clone https://github.com/centre-for-humanities-computing/newsFluxus.git
pip install -r newsFluxus/requirements.txt

python src/emotionFluxus.py --filenames tweets_emo_date tweets_pol_date --window 3

python src/emotionFluxus.py --filenames tweets_emo_date --extract_emotions emo
python src/emotionFluxus.py --filenames tweets_pol_date --extract_emotions pol

python src/smoothing.py --filename tweets_emo_date_W3 --span 150
python src/smoothing.py --filename tweets_pol_date_W3 --span 150

python src/changepoints.py --in_file tweets_emo_date_W3