import numpy as np

X_filenames = [
    'model/X.npy',
    'X9_20181226150350.npy',
    'X_20181226171732.npy',
    'X_20181226172317.npy',
    'X_20181226172954.npy',
    'X_20181226173432.npy',
    'X_20181228114849.npy',
    'X_20181228140937.npy',
    'X_20181228140937.npy'
]

Y_filenames = [
    'model/Y.npy',
    'Y9_20181226150350.npy',
    'Y_20181226171732.npy',
    'Y_20181226172317.npy',
    'Y_20181226172954.npy',
    'Y_20181226173432.npy',
    'Y_20181228114849.npy',
    'Y_20181228140937.npy',
    'Y_20181228140937.npy'
]

X = None
Y = None
for i in range(len(X_filenames)):
    X_file = X_filenames[i]
    Y_file = Y_filenames[i]

    new_X = np.load(X_file)
    new_Y = np.load(Y_file)

    print('X_file', X_file)
    print('new_X.shape', new_X.shape)
    print('new_Y.shape', new_Y.shape)

    if X is None:
        X = new_X
    else:
        X = np.concatenate((X, new_X), axis=0)

    if Y is None:
        Y = new_Y
    else:
        Y = np.concatenate((Y, new_Y), axis=0)

print('X.shape', X.shape)
print('Y.shape', Y.shape)

np.save(open('model/X_combine.npy','wb'), X)
np.save(open('model/Y_combine.npy','wb'), Y)