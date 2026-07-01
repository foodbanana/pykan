from utils import save, datadir
import pandas as pd
import os
os.makedirs(os.path.join(datadir, 'temp'), exist_ok=True)
d = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
save(os.path.join(datadir, 'temp', 'test.dill'), d, verbose=True)
