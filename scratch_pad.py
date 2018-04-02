import numpy as np
from random import shuffle

a = np.array([1,2,3,4,5,6,7,8,9])

b = a + 10
a = a-1
print (b)


c = shuffle(a)
print (b[a])