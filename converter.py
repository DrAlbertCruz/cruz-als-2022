import json
import pickle
from os import listdir

for file in [path for path in listdir() if path.endswith('pk1')]:
        with open(file, "rb") as fp:
                    with open(file.split('.')[0] + ".json", "wt") as fp_json:
                                    json.dump(pickle.load(fp), fp_json)
                                    
