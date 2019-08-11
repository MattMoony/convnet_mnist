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

    plt.plot(np.arange(len(past_J)), past_J, c='darkslategray', linestyle='-')

    plt.xlabel('Epoch')
    plt.ylabel('Cost (J)')
    plt.legend(['J/Epoch-Graph'])

def show_conv_weights(Ws):
    for i, W in enumerate(Ws):
        plt.figure()
        plt.suptitle('Conv-Layer #{:}'.format(i))

        for y, f in enumerate(W):
            for x, d in enumerate(f.T, 1):
                plt.subplot(W.shape[0], W.shape[-1], y * W.shape[-1] + x)
                plt.imshow(d.T, cmap='gray')

    plt.show()

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

# def convolve_s(x, W):
#     fshape = tuple(np.subtract(x.shape[:-1], W.shape[1:-1]) + 1)
#     sm = np.lib.stride_tricks.as_strided(x, (*fshape, *W.shape[1:-1], x.shape[-1]), tuple(x.strides[:-1]) + tuple(x.strides))
    
#     res = np.sum(sm * W[0], (4,3,2))
#     for w in W[1:]:
#         res = np.dstack((res, np.sum(sm * w, (4,3,2))))
#     return res

def convolve(x, W):
    fshape = tuple(np.subtract(x.shape[1:-1], W.shape[1:-1]) + 1)
    sm = np.lib.stride_tricks.as_strided(x, (x.shape[0], *fshape, *W.shape[1:-1], x.shape[-1]), tuple(x.strides[:-1]) + tuple(x.strides[1:]))
    res = np.sum(sm * W[0], (5,4,3))[:,:,:,np.newaxis]
    for w in W[1:]:
        res = np.concatenate((res, np.sum(sm * w, (5,4,3))[:,:,:,np.newaxis]), axis=3)
    return res

def deconvolve(a, g, W):
    fshape = tuple(np.subtract(a.shape[1:-1], W.shape[1:-1]) + 1)
    sm = np.lib.stride_tricks.as_strided(a, (a.shape[0], *fshape, *W.shape[1:-1], a.shape[-1]), tuple(a.strides[:-1]) + tuple(a.strides[1:]))
    # print(a.shape)
    # print(g.shape)
    # print(W.shape)
    # print(sm.shape)
    # print('-'*24)
    res = np.sum(sm * np.repeat(g[:,:,:,0], np.prod(W.shape[1:]), axis=2).reshape(sm.shape), axis=(2,1,0))
    for i in range(1, g.shape[-1]):
        res = np.vstack((res, np.sum(sm * np.repeat(g[:,:,:,i], np.prod(W.shape[1:]), axis=2).reshape(sm.shape), axis=(2,1,0))))
    return res.reshape(W.shape)

def de_con_actv(o, w):
    u = np.repeat(o, np.prod(w.shape[1:]), axis=3).reshape(*o.shape, *w.shape[1:]) * w.reshape(1,1,1,*w.shape)
    u = np.sum(u, axis=3)
    oshape = (o.shape[0], *np.add(np.subtract(o.shape[1:-1], 1), w.shape[1:-1]), w.shape[-1])
    res = np.zeros(oshape)
    for y in range(o.shape[1]):
        for x in range(o.shape[2]):
            res[:,y:y+w.shape[1],x:x+w.shape[2],:] += u[:,y,x,:,:,:]
    return res

def ReLU(z):
    return np.maximum(0, z)

def ReLU_grad(z, *args, **kwargs):
    grad = np.maximum(0, z)
    grad[grad != 0] = 1
    return grad

def softmax(Z):
    Z = Z.copy()
    Z -= np.max(Z, 1, keepdims=True)
    return np.exp(Z) / np.sum(np.exp(Z), 1)[:,np.newaxis]

def softmax_grad(Z, y, *args, **kwargs):
    grad = softmax(Z)
    grad[np.arange(grad.shape[0]), y] -= 1
    return grad

def predict(x, conv_ws, fc_ws, bs):
    zl = 0
    al = x
    wi = 0

    for w, actf, actf_g in conv_ws:
        zl = convolve(al, w) + bs[wi]
        al = actf(zl)

        wi+=1

    al = al.reshape((al.shape[0], np.prod(al.shape[1:])))

    for w, actf, actf_g in fc_ws:
        zl = al.dot(w) + bs[wi]
        al = actf(zl)

        wi+=1

    return al

def predict_s(x, conv_ws, fc_ws, bs):
    return predict(x[np.newaxis,:], conv_ws, fc_ws, bs)

def loss(X, y, conv_ws, fc_ws, bs, lamb=1e-3):
    preds = predict(X, conv_ws, fc_ws, bs)
    y = y.copy().squeeze()

    l = np.sum(-np.log(preds[np.arange(X.shape[0]), y]))
    l /= X.shape[0]

    l += (lamb / (2 * X.shape[0])) * np.sum(np.asarray([np.sum(np.asarray(x[0]) ** 2) for x in conv_ws + fc_ws]))
    return l

def compute_gradient(X, y, conv_ws, fc_ws, bs, lamb=1e-3):
    # -- FORWARD PASS ------------------------------------------------------------------------- #

    zl = 0
    al = X
    wi = 0

    zl_s = []
    al_s = [al]

    for w, actf, actf_g in conv_ws:
        zl = convolve(al, w) + bs[wi]
        al = actf(zl)

        zl_s.append(zl)
        al_s.append(al)

        wi+=1

    last_conv_shape = al.shape

    al = al.reshape((al.shape[0], np.prod(al.shape[1:])))
    al_s[-1] = al

    for w, actf, actf_g in fc_ws:
        zl = al.dot(w) + bs[wi]
        al = actf(zl)

        zl_s.append(zl)
        al_s.append(al)

        wi+=1

    # -- BACKWARD PASS ------------------------------------------------------------------------ #

    conv_grads = []
    fc_grads = []
    bs_grads = []

    al_grad = np.ones((al_s[-1].shape))
    wi -= 1

    m = X.shape[0]

    for w, actf, actf_g in reversed(fc_ws):
        ac_grad = al_grad * actf_g(zl_s[wi], y.squeeze())

        bs_grads.append(np.sum(ac_grad) / m)
        fc_grads.append(al_s[wi].T.dot(ac_grad) / m + lamb * (w / m))

        al_grad = ac_grad.dot(w.T)
        wi -= 1

    al_grad = al_grad.reshape(last_conv_shape)

    for w, actf, actf_g in reversed(conv_ws):
        ac_grad = al_grad * actf_g(zl_s[wi], y.squeeze())

        bs_grads.append(np.sum(ac_grad, axis=(0,1,2)) / m)
        conv_grads.append(deconvolve(al_s[wi], ac_grad, w) / m + lamb * (w / m))

        al_grad = de_con_actv(ac_grad, w)
        wi -= 1

    # -- RETURN VALUES ------------------------------------------------------------------------ #

    conv_grads = list(reversed(conv_grads))
    fc_grads = list(reversed(fc_grads))
    bs_grads = list(reversed(bs_grads))

    return conv_grads, fc_grads, bs_grads

    # -- FIN ---------------------------------------------------------------------------------- #

def sgd(X, y, conv_ws, fc_ws, bs, lamb=0.001, iters=10, alpha=1, batch_size=16):
    conv_ws = conv_ws.copy()
    for i in range(len(conv_ws)):
        conv_ws[i][0] = conv_ws[i][0].copy()
    fc_ws = fc_ws.copy()
    for i in range(len(fc_ws)):
        fc_ws[i][0] = fc_ws[i][0].copy()
    bs = bs.copy()
    for i in range(len(bs)):
        bs[i] = bs[i].copy()

    a_inds = np.arange(X.shape[0])
    inds = np.random.choice(a_inds, batch_size)

    past_Js = [loss(X[inds], y[inds], conv_ws, fc_ws, bs, lamb)]

    for i in range(iters):
        inds = np.random.choice(a_inds, batch_size)
        conv_g, fc_g, bs_g = compute_gradient(X[inds], y[inds], conv_ws, fc_ws, bs, lamb)

        for j in range(len(conv_ws)):
            conv_ws[j][0] -= alpha * conv_g[j]
        for j in range(len(fc_ws)):
            fc_ws[j][0] -= alpha * fc_g[j]
        for j in range(len(bs)):
            bs[j] -= alpha * bs[j]

        inds = np.random.choice(a_inds, batch_size)
        past_Js.append(loss(X[inds], y[inds], conv_ws, fc_ws, bs, lamb))

        print('[iter#{:04d}] Loss: {:f} ... '.format(i, past_Js[-1]))

    return [conv_ws, fc_ws, bs, past_Js]

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def main():
    # -- DATASET SETUP ------------------------------------------------------------------------ #
    
    x_train = read_dataset('train-images-idx3-ubyte.gz', 60000)[:,:,:,np.newaxis]
    y_train = read_labels('train-labels-idx1-ubyte.gz', 60000)
    x_test = read_dataset('t10k-images-idx3-ubyte.gz', 10000)[:,:,:,np.newaxis]
    y_test = read_labels('t10k-labels-idx1-ubyte.gz', 10000)

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    x_val, y_val = x_train[-x_test.shape[0]:], y_train[-y_test.shape[0]:]
    x_train, y_train = x_train[:-x_test.shape[0]], y_train[:-y_test.shape[0]]

    # -- DATASET PREPARATION ------------------------------------------------------------------ #

    mu          = np.mean(x_train, 0)
    sigma       = np.max(x_train, 0) - np.min(x_train, 0) + 1e-8

    x_train, x_val, x_test = normalize(mu, sigma, x_train, x_val, x_test)

    # -- MODEL SETUP -------------------------------------------------------------------------- #

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
    #   . #Weights: 18*18*2 = 648      #
    #   . #Outputs: 10                 #
    #   . Activation: Softmax          #
    # -------------------------------- #

    W1 = np.random.rand(4,8,8,1)
    W2 = np.random.rand(2,4,4,4)
    W3 = np.random.rand(648,10)

    b1 = np.zeros(4) + 1e-3
    b2 = np.zeros(2) + 1e-3
    b3 = np.zeros(1) + 1e-3

    conv_ws = [[W1, ReLU, ReLU_grad], [W2, ReLU, ReLU_grad]]
    fc_ws = [[W3, softmax, softmax_grad]]
    bs = [b1, b2, b3]

    # -- MODEL TRAINING ----------------------------------------------------------------------- #

    n_conv_ws, n_fc_ws, n_bs, past_Js = sgd(x_train, y_train, conv_ws, fc_ws, bs, 
        lamb=0.5, iters=1024, alpha=1, batch_size=32)
    
    # -- MODEL EVALUATION --------------------------------------------------------------------- #
    


    # -- MODEL VISUALIZATION ------------------------------------------------------------------ #

    plt.ioff()
    plot_j_epoch(past_Js)

    plt.show()

    # -- FIN ---------------------------------------------------------------------------------- #

if __name__ == '__main__':
    main()