from os import listdir
import pandas as pd
from os.path import isfile, join
#dirlist = [f for f in listdir('emerson') if isfile(join('emerson', f))]



dirlist = [f for f in listdir('pickle_emerson') if isfile(join('pickle_emerson', f))]
print(len(dirlist))

#for i in dirlist:
#    temp_data = pd.read_csv('emerson/{0}'.format(i), engine='c')
#    temp_data.to_pickle('pickle_emerson/{0}.pickle'.format(i.split('.')[0]))
    
    
#for i in dirlist:
#    t = pd.read_csv('.emerson\{0}'.format(i), encoding='utf-8')
#    t.to_pickle('emerson\{0}.pickle'.format(i.split('.')[0]))
    