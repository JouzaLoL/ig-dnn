from insta import getData
import numpy as np

data = getData("reknihy")
data = data.astype(int)
np.savetxt("reknihy.csv", data, delimiter=",")