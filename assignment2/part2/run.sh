#!/bin/bash

bash stopservers.sh
bash startservers.sh
#dstat --time --top-cpu --top-mem -n --output ./dstat.csv &
python -m AlexNet.scripts.train --mode cluster2
