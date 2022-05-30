import os
import glob
import pandas as pd

os.chdir("/storage2/dvandijk/Agent2D-DataExtractor/dataset7/")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
path = '/storage2/dvandijk/modified_data/fullstate/unum/1match.csv'
combined_csv.to_csv(path, index=False)
print("done")
