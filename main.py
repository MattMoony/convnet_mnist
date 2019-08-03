import gzip
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def read_dataset(in_fname, num_imgs, img_size=28):
    f = gzip.open(in_fname)
    f.read(16)

    buf = f.read(img_size * img_size * num_imgs)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_imgs, img_size, img_size)

    return np.asarray(data)

def read_labels(in_fname, num_labels, label_size=1):
    f = gzip.open(in_fname)
    f.read(8)

    buf = f.read(num_labels * label_size)
    labl = np.frombuffer(buf, dtype=np.uint8).astype(np.uint8)
    labl = labl.reshape(num_labels, label_size)

    return np.asarray(labl)

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def plot_img(x, colorbar=False):
    plt.figure()
    plt.imshow(x, cmap='gray')

    if colorbar:
        plt.colorbar()
    plt.show()

def show_random_example(X, colorbar=False):
    plt.figure()
    plt.imshow(X[np.random.choice(np.arange(X.shape[0]), 1)].squeeze(), cmap='gray')

    if colorbar:
        plt.colorbar()
    plt.show()

def plot_j_epoch(past_J):
    plt.figure()
    plt.title('J/Epoch-Graph')

    plt.plot(np.arange(pastJ.shape[0]), pastJ, c='darkslategray', linestyle='-')

    plt.xlabel('Epoch')
    plt.ylabel('Cost (J)')
    plt.legend(['J/Epoch-Graph'])

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def shuffle(X, y):
    inds = np.arange(X.shape[0])
    np.random.shuffle(inds)

    return X[inds], y[inds]

def normalize(mu, sigma, *args):
    res = []
    for arg in args:
        res.append((arg - mu) / sigma)
    return res

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def convolve(x, W):
    fshape = tuple(np.subtract(x.shape[:-1], W.shape[1:-1]) + 1)
    sm = np.lib.stride_tricks.as_strided(x, (*fshape, *W.shape[1:-1], x.shape[-1]), tuple(x.strides[:-1]) + tuple(x.strides))
    
    res = np.sum(sm * W[0], (4,3,2))
    for w in W[1:]:
        res = np.dstack((res, np.sum(sm * w, (4,3,2))))
    return res

def ReLU(z):
    return np.maximum(0, z)

def predict(x, W1, W2, W3, b1, b2, b3):
    z1 = convolve(x, W1) + b1
    a1 = ReLU(z1)
    z2 = convolve(a1, W2) + b2
    a2 = ReLU(z2).flatten()
    z3 = a2.dot(W3) + b3
    a3 = z3

    return a3

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def main():
    # ----------------------------------------------------------------------------------------- #
    
    x_train = read_dataset('train-images-idx3-ubyte.gz', 60000)[:,:,:,np.newaxis]
    y_train = np.where(np.ones((x_train.shape[0], 10)) * np.arange(10) == read_labels('train-labels-idx1-ubyte.gz', 60000), 1, 0)
    x_test = read_dataset('t10k-images-idx3-ubyte.gz', 10000)[:,:,:,np.newaxis]
    y_test = np.where(np.ones((x_test.shape[0], 10)) * np.arange(10) == read_labels('t10k-labels-idx1-ubyte.gz', 10000), 1, 0)

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    x_val, y_val = x_train[-x_test.shape[0]:], y_train[-y_test.shape[0]:]
    x_train, y_train = x_train[:-x_test.shape[0]], y_train[:-y_test.shape[0]]

    # ----------------------------------------------------------------------------------------- #

    mu          = np.mean(x_train, 0)
    sigma       = np.max(x_train, 0) - np.min(x_train, 0) + 1e-8

    x_train, x_val, x_test = normalize(mu, sigma, x_train, x_val, x_test)

    # ----------------------------------------------------------------------------------------- #

    # -------------------------------- #
    # MODEL: 2 conv-layers, 1 fc-layer #
    # - 1st conv-layer:                #
    #   . Filter-Size: 8x8             #
    #   . Stride: 1                    #
    #   . #Filters: 4                  #
    #   . Activation: ReLU             #
    # - 2nd conv-layer:                #
    #   . Filter-Size: 4x4             #
    #   . Stride: 1                    #
    #   . #Filters: 2                  #
    #   . Activation: ReLU             #
    # - 1st fc-layer:                  #
    #   . #Weights: 18*18*3 = 972      #
    #   . #Outputs: 10                 #
    # -------------------------------- #

    W1 = np.random.rand(4,8,8,1)
    W2 = np.random.rand(2,4,4,4)
    W3 = np.random.rand(648,10)

    b1 = np.random.rand(4)
    b2 = np.random.rand(2)
    b3 = np.random.rand(1)

    # ----------------------------------------------------------------------------------------- #

    print(predict(x_train[0], W1, W2, W3, b1, b2, b3))
    print(np.argmax(predict(x_train[0], W1, W2, W3, b1, b2, b3)))
    
    print(y_train[0])
    print(np.argmax(y_train[0]))

    # ----------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    main()