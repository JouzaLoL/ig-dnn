from insta import getData
import numpy as np
import sys
import os


profileName = sys.argv[1] if len(sys.argv) > 1 != None else 'reknihy'
fileName = profileName + ".csv"
print("Getting data from profile: " + profileName)
data = getData(profileName)
data = data.astype(int)

if os.path.exists(fileName):
  os.remove(fileName)
  print("File already exists, removing")

np.savetxt(fileName, data, delimiter=",")
print("Data saved in " + fileName)