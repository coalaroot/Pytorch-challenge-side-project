import numpy as np
import pandas as pd

X_filenames = [
    'model/X.npy',
    'X9_20181226150350.npy',
    'X_20181226171732.npy',
    'X_20181226172317.npy',
    'crc_X_20181226172954.npy',
    'crc_X_20181226173432.npy',
    'X_20181228114849.npy',
    'X_20181228140937.npy',
    'X_20181228140937.npy',
    'X_20181230102906.npy',
    'crc_X_20181230142850.npy'
]

Y_filenames = [
    'model/Y.npy',
    'Y9_20181226150350.npy',
    'Y_20181226171732.npy',
    'Y_20181226172317.npy',
    'crc_Y_20181226172954.npy',
    'crc_Y_20181226173432.npy',
    'Y_20181228114849.npy',
    'Y_20181228140937.npy',
    'Y_20181228140937.npy',
    'Y_20181230102906.npy',
    'crc_Y_20181230142850.npy'
]

label_index_to_label_dict = {
    1:0,
    4:1,
    8:2,
    7:3,
    6:4,
    9:5,
    3:6,
    2:7,
    5:8,
    0:9
}

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

    if new_X.max() > 1:
        new_X = (new_X / 255).astype(np.float32)
        
    if X is None:
        X = new_X
    else:
        X = np.concatenate((X, new_X), axis=0)

    if Y is None:
        Y = new_Y
    else:
        Y = np.concatenate((Y, new_Y), axis=0)

    labels = np.argmax(new_Y, axis=1)

    for i in range(len(labels)):
        labels[i] = label_index_to_label_dict[labels[i]]
    print(pd.DataFrame({'label':labels}).groupby('label')['label'].count())
    print()

print('X.shape', X.shape)
print('Y.shape', Y.shape)

np.save(open('model/X_combine.npy','wb'), X)
np.save(open('model/Y_combine.npy','wb'), Y)