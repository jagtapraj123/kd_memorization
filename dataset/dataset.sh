#!/bin/bash

kaggle datasets download hereisburak/pins-face-recognition
unzip pins-face-recognition.zip

python data_init.py -n 32