import numpy as np

prediction = -2
DOA = 0

# error = (((prediction - DOA) * np.pi / 180) + np.pi / 2) % np.pi - np.pi / 2
error = ((((prediction - DOA) * np.pi / 180) + np.pi) % (2 * np.pi)) - np.pi 
# error = ((((p - DOA) * np.pi / 180)) % (2 * np.pi)) 
prmse = (1 / np.sqrt(1)) * np.linalg.norm(error)

print("error" , error)
print("prmse" , prmse)