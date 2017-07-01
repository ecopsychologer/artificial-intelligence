# Artificial Intelligence Stuff

### Neural Network Code
```python
import numpy as np


def nonlin(x, deriv=False):
    if(deriv == True):
        return x * (1-x)

    return 1 / (1+np.exp(-x))

# input data
X = np.array([[0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0]])

# output data
y = np.array([[0], [1], [1], [1], [0], [0]])

np.random.seed(1)

# synapses
syn0 = 2*np.random.random((4, 6)) - 1
syn1 = 2*np.random.random((6, 6)) - 1
syn2 = 2*np.random.random((6,6)) - 1
syn3 = 2*np.random.random((6, 1)) - 1

# training step
for j in range(60000):
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    l3 = nonlin(np.dot(l2, syn2))
    l4 = nonlin(np.dot(l3, syn3))

    l4_error = y - l4

    if(j % 10000) == 0:
        print("Error Rate: " + str(np.mean(np.abs(l4_error))))

    l4_delta = l4_error * nonlin(l4, deriv=True)

    l3_error = l4_delta.dot(syn3.T)

    l3_delta = l3_error * nonlin(l3, deriv=True)

    l2_error = l3_delta.dot(syn2.T)

    l2_delta = l2_error * nonlin(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(l1, deriv=True)

    # update synapse weights
    syn3 += l3.T.dot(l4_delta)
    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print("Output after training")
print(l4)

n = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0]])

j0 = n
j1 = nonlin(np.dot(j0, syn0))
j2 = nonlin(np.dot(j1, syn1))
j3 = nonlin(np.dot(j2, syn2))
j4 = nonlin(np.dot(j3, syn3))


print("Output after testing")
print(j4)

```
