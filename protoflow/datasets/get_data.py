import csv
import numpy as np

"""Spiral dataset."""
def make_spirals(n_samples=500, spiral_noise=0.3):
    def get_samples(n, delta_t):
        f = np.random.randn(1)*5
        points = []
        for i in range(n):
            r = i / n_samples * 5
            t = 1.75 * i / n * 2 * np.pi + delta_t
            x = r * np.sin(t) + np.random.rand(1) * 0.5
            y = r * np.cos(t) + np.random.rand(1) * 0.5
            z = spiral_noise + np.random.rand(1) * 0.5
            #points.append([x, y, (f-np.cos(x)**2)*(1-f)+(f-np.cos(y)**2)*(f-1)+spiral_noise])
            points.append([x, y, z])
        return points
    n = n_samples // 2
    positive = get_samples(n=n, delta_t=0)
    negative = get_samples(n=n, delta_t=np.pi)
    x = np.concatenate([np.array(positive).reshape(n, -1), 
                        np.array(negative).reshape(n, -1)],
                        axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return x, y


def get_spiral_dataset(spiral_noise=[-0.5,0.5], concatenate=True):
    n_samples = 1000
    nsources = len(spiral_noise)

    for i, sn in enumerate(spiral_noise):
        x_spirals, y_spirals = make_spirals(n_samples=n_samples, 
                                            spiral_noise=sn)
        if i == 0:
            x = [x_spirals]
            yc = [y_spirals]
            ys = [np.zeros(n_samples)]
        else:
            x.append(x_spirals)
            yc.append(y_spirals)
            ys.append(np.full(n_samples,i))
    nclasses = len(list(set(yc[0])))
    if concatenate:
        x = np.concatenate(x)
        yc = np.concatenate(yc)
        ys = np.concatenate(ys)

    return x, yc, ys, nclasses, nsources


""" Bonbon-dataset """
def z_score(data, axis=0):
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    data = (data-mean)/std
    return data


def get_bonbon_dataset(concatenate=True):
    data_sources = ['protoflow/datasets/measurement1.csv',
                    'protoflow/datasets/measurement2.csv',
                    'protoflow/datasets/measurement3.csv']
    nsources = 3

    x_train, y_train = [], []
    for source_f in data_sources:
        x_train.append([])
        y_train.append([])
        with open(source_f) as sf:
            sf_data = csv.reader(sf, delimiter=',', quotechar='|')
            for row in sf_data:
                x_train[-1].append(row[:-1])
                y_train[-1].append(row[-1])
            x_train[-1] = np.asarray(x_train[-1], dtype=float)
            y_train[-1] = np.asarray(y_train[-1], dtype=float) - 1

    nclasses = np.amax(np.asarray([len(set(y)) for y in y_train]))

    ys_train = []
    for i in range(len(data_sources)):
        ys_train.append(np.full(y_train[i].shape[0], i))

    if concatenate:
        x_train = np.concatenate(x_train)
        ys_train = np.concatenate(ys_train)
        yc_train = np.concatenate(y_train)
        """
        one_shot = np.full(y_train[1].shape[0], -1)
        for i in range(25):
            one_shot[i*2] = y_train[1][i*2]
        yc_train = np.concatenate([y_train[0], one_shot])
        """
        x_train = z_score(x_train)
    else:
        x_train = np.asarray([z_score(x) for x in x_train])
        yc_train = np.asarray(y_train)
        ys_train = np.asarray(ys_train)

    return x_train, yc_train, ys_train, nclasses, nsources
    


def get_data(dataset="spirals", concatenate=True):
    DATASETS = {"spirals":get_spiral_dataset,
                "bonbons":get_bonbon_dataset}
    
    return DATASETS[dataset](concatenate=concatenate)
