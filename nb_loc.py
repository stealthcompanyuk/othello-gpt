from glob import glob
from json import load
from pprint import pprint

def loc(nb):
    cells = load(open(nb))["cells"]
    return sum(len(c["source"]) for c in cells)

root_folder = "."
locs = {}
for f in glob(root_folder+"/**/*.ipynb", recursive=True):
    locs[f] = loc(f)

pprint(locs)
