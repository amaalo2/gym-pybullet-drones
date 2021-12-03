import numpy as np
import sys
data = np.load("save-flight-10.27.2021_13.00.24.npy")
print(data.files)
row = data.files
np.set_printoptions(threshold=np.inf)
#print(data['arr_0'])
sys.stdout=open("test.txt","w")
for i in row:
    print("--------------------------")
    print(data[i])
sys.stdout.close()