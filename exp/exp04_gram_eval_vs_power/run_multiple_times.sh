#!/bin/sh

# Delete existing output file
rm -f output.txt

# Run ``run.py`` multiple times, append output to ``output.txt``
for i in 1 2 3 4 5
do
   echo "Run $i"
   python run.py >> output.txt
done
