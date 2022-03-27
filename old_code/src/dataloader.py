import numpy as np
from config import NEW_DATA


DATA = None

if NEW_DATA:
    from dataloader_new import DATA, DATA_TEST
else:
    from dataloader_old import DATA

print("Dataset {} loaded, containing image preprocessing of size{} ".format(DATA.name, np.shape(DATA.images)))

