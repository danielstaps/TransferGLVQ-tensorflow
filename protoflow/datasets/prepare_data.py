import numpy as np


def shuffle_data(data, label, s_label):
    indices = np.arange(len(label))
    np.random.shuffle(indices)
    data, label = data[indices], label[indices]
    if type(s_label) == bool:
        return data, label, False
    else:
        s_label = s_label[indices]
        return data, label, s_label


def k_split_dataset(x, yc, ys, k):
    indices = np.arange(len(yc))
    indices = np.array_split(indices, k, axis=0)
    x = [x[ind] for ind in indices]
    yc = [yc[ind] for ind in indices]
    if type(ys) == bool:
        return x, yc, False
    else: 
        ys = [ys[ind] for ind in indices]
        return x, yc, ys


def check_distribution(y, nclasses, exclude_axis):
    if exclude_axis:
        for col in exclude_axis:
            y = [np.delete(y[i], col, axis=1) for i in range(len(y))]
    hist = np.asarray([np.histogram(split, range=(0, nclasses-1), bins=nclasses)[0] for split in y])
    mean = np.mean(hist, axis=1)
    hist = np.subtract(hist.T, mean).T
    std = np.std(hist, axis=1)
    hist = np.divide(hist.T, std).T
    separation = [1 for split in std if split < 1.5]
    if len(y) == sum(separation):
        return True
    else:
        return False


def prepare_data_for_crossvalidation(x, y_classes, nclasses, k, 
                                     exclude_axis=False, 
                                     no_check=False,
                                     y_sources=False):
    print("Search for a good separated k-split ...")
    while(True):
        x_, yc_, ys_ = shuffle_data(x, y_classes, y_sources)
        x_, yc_, ys_ = k_split_dataset(x_, yc_, ys_, k) 
        if check_distribution(yc_, nclasses, exclude_axis) or no_check:
            break
    return x_, yc_, ys_


def perturbate_data(x, y_classes, i, k, y_sources=False):
    test_indices = i
    train_indices = np.delete(np.arange(0,k), i, axis=0)
    x_train = np.concatenate([x[j] for j in train_indices])
    yc_train = np.concatenate([y_classes[j] for j in train_indices])
    if y_sources:
        ys_train = np.concatenate([y_sources[j] for j in train_indices])
    #x_test  = np.concatenate([x[j] for j in test__indices])
    #y_test  = np.concatenate([y[j] for j in test__indices])
    x_test = x[i]
    yc_test = y_classes[i]
    if y_sources:
        ys_test = y_sources[i]   
        return ((x_train, x_test), (yc_train, yc_test), (ys_train, ys_test))
    else:
        return ((x_train, x_test), (yc_train, yc_test))
