#!/usr/bin/env bash

docker run -v $PWD/temp/autophrasein:/autophrase/data -v $PWD/temp/autophraseout:/autophrase/models \
       -e RAW_TRAIN=data/autophraseinput.txt \
       -e ENABLE_POS_TAGGING=1 \
       -e MIN_SUP=30 -e THREAD=10 \
       -e MODEL=models/MyModel \
       -e TEXT_TO_SEG=data/autophraseinput.txt \
       --rm \
       remenberl/autophrase ./auto_phrase.sh
