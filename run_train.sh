#!/bin/bash
source activate mla36
echo "start"

python script.py

echo "end"
source deactivate
