import numpy as np
fSamp = 10
f_array = []
for i in range(int(fSamp)):
                # mapping from index i to frequency f
                if i > int(fSamp) // 2:
                                f = - int(fSamp) + i
                else: f = i
                f_array.append(f)
freq_axis_pos = np.linspace(0, fSamp // 2 , fSamp // 2 + 1, endpoint=True)
freq_axis_neg = np.linspace(-(fSamp // 2) + 1 , 0, fSamp // 2 - 1, endpoint=False)
freq_axis = np.concatenate([freq_axis_pos, freq_axis_neg])
print("end")