import json
import sys
import csv

in_file = sys.argv[1]
out_file = sys.argv[2]

labels = {}
with open(in_file, 'r') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        id = row['path'].lstrip('/mnt/md0/beck/datasets/benchmarks/faceforensics_benchmark_images/')
        pad_id = str(id).rjust(8, '0')
        print(pad_id)
        labels[pad_id] = row['label']

with open(out_file, 'x') as o:
     json.dump(labels, o, sort_keys=True)

    
