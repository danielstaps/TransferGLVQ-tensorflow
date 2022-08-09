import gc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

TIMER = 100000
PLOTS = False


def close_event():
    plt.close()

def get_eigValVec(data):
    data_cov_mat = np.cov(data.T)
    eigVal, eigVec = np.linalg.eig(data_cov_mat)
    eigVal = eigVal/np.sum(eigVal)
    eigVec = np.asarray(eigVec, dtype='float32')
    idx = np.abs(eigVal).argsort()[::-1] # sorts descending
    eigVal, eigVec = eigVal[idx], eigVec[idx]
    return eigVal, eigVec

def pca(x, dims, eigVec=False):
    if type(eigVec) == bool:
        eigVal, eigVec = get_eigValVec(x)
        eigVec = eigVec[:, :dims]
        eigVec = eigVec.T
    return np.matmul(eigVec, x.T), eigVec


def plot_3d(x, y, title):
    if x.shape[1] > 3:
        x, _ = pca(x, 3)
    else:
        x = x.T

    fig = plt.figure()
    timer = fig.canvas.new_timer(interval = TIMER)
    timer.add_callback(close_event)
    ax = fig.add_subplot(111, projection='3d')

    colors = ['gold','orange','khaki','yellow','yellowgreen','gray']
    c = [colors[int(yc)] for yc in y]
    print(x.shape, len(c))
    ax.scatter(*x, c=c, marker='o', alpha=1)

    ax.set_title(title)

    ax.set_xlabel('1')
    ax.set_ylabel('2')
    ax.set_zlabel('3')
    
    timer.start()
    if PLOTS:
        plt.show()

    plt.clf()
    return 


def mapping(smi_gmlvq, x, space="null"):
    if space == "null":
        mapper = smi_gmlvq.ComplementaryOrthogonalProjector().numpy()
    if space == "row":
        mapper = smi_gmlvq.omega_layer.omega_matrix.numpy()
    input_omega_map = np.matmul(
            x,
            mapper.T)
    proto_omega_map = np.matmul(
            smi_gmlvq.source_prototype_layer.get_weights()[0],
            mapper.T)
    proto_COP_map = np.matmul(
            smi_gmlvq.class_prototype_layer.get_weights()[0],
            mapper.T)
    return input_omega_map, proto_omega_map, proto_COP_map



def plot_pca(x, y, title, mapping=False):
    if mapping:
        x, _, _ = mapping(smi_gmlvq, x)
    x, _ = pca(x, 2)        

    colors = ['gold','orange','khaki','yellow','yellowgreen','gray']
    c = [colors[int(yc)] for yc in y]
 
    fig = plt.figure()
    timer = fig.canvas.new_timer(interval = TIMER)
    timer.add_callback(close_event)
   
    plt.scatter(*x, c=c)
    plt.title(title)
    
    timer.start()
    if PLOTS:
        plt.show()

    plt.clf()

    return 


def pca_space(x, y_classes, smi_gmlvq, dims, space, anchor='data'):
    input_omega_map, proto_omega_map, proto_COP_map = mapping(
            smi_gmlvq, x, space)
    if space == 'null':
        input_omega_map = smi_gmlvq.omega_layer_2(input_omega_map)
        proto_omega_map = smi_gmlvq.omega_layer_2(proto_omega_map)
        proto_COP_map = smi_gmlvq.omega_layer_2(proto_COP_map)

    if input_omega_map.shape[1] == 1:
        input_plot = np.concatenate([input_omega_map, np.zeros(input_omega_map.shape)], axis=1).T
        sourc_plot = np.concatenate([proto_omega_map, np.zeros(proto_omega_map.shape)], axis=1).T
        class_plot = np.concatenate([proto_COP_map, np.zeros(proto_COP_map.shape)], axis=1).T
    elif dims == x.shape[1]:
        input_plot = input_omega_map.T
        sourc_plot = proto_omega_map.T
        class_plot = proto_COP_map.T
    else:
        if anchor == 'data':
            _, eigVec = pca(input_omega_map, dims)
        if anchor == 'protos':
            if space == 'null':
                _, eigVec = pca(proto_COP_map, dims)
            if space == 'row':
                _, eigVec = pca(proto_omega_map, dims)
        if anchor == 'all':
            if space == 'null':
                _, eigVec = pca(np.concatenate([proto_COP_map, input_omega_map]), dims)
            if space == 'row':
                _, eigVec = pca(np.concatenate([proto_omega_map, input_omega_map]), dims)
        input_plot, _ = pca(input_omega_map, dims, eigVec)
        sourc_plot, _ = pca(proto_omega_map, dims, eigVec)
        class_plot, _ = pca(proto_COP_map, dims, eigVec)
    return input_plot, sourc_plot, class_plot


def plot_space(
        x, y_classes, smi_gmlvq, 
        space="null", img_name=False, img_path=None, title="", dims=2, acc=None):

    input_plot, sourc_plot, class_plot = pca_space(
            x, y_classes, smi_gmlvq, dims, space, anchor='data')
 
    plt.close('all')
    fig = plt.figure()
    if PLOTS:
        timer = fig.canvas.new_timer(interval = TIMER)
        timer.add_callback(close_event)

    if dims==3:
        ax = fig.add_subplot(111, projection='3d')
    if dims==2:
        ax = fig.add_subplot(111)

    if space == "row":
        #data_colors = ['royalblue','springgreen']
        #prot_colors = ['navy','green']
        data_colors = ['orange','gray']
        prot_colors = ['darkred','black']
    if space == "null":
        #colors = ['fuchsia','crimson','deeppink','violet','pink']
        #colors = ['gold','orange','khaki','yellow','yellowgreen','gray']
        #colors = ['sienna','darkorchid','darkgreen','royalblue','chocolate']
        
        #data_colors = ['orange','gray']
        #prot_colors = ['darkred','black']
        data_colors = ['royalblue','springgreen']
        prot_colors = ['navy','green']

    c = [data_colors[int(yc)] for yc in y_classes]
    #print(input_plot.shape, len(c))
    ax.scatter(*input_plot, c=c, marker='.', alpha=0.15, zorder=-1)
    
    if space == "row":
        #colors = ['black', 'blue']
        c = [prot_colors[int(ys)] for ys in smi_gmlvq.source_prototype_layer.get_weights()[1]]
        ax.scatter(*sourc_plot, c=c, marker='D', alpha=1, zorder=1)

    if space == "null":
        #colors = ['fuchsia','crimson','deeppink','violet','pink']
        c = [prot_colors[int(yc)] for yc in smi_gmlvq.class_prototype_layer.get_weights()[1]]
        ax.scatter(*class_plot, c=c, marker='D', alpha=1, zorder=1)
    
    ax.set_xlabel('1')
    ax.set_ylabel('2')
    if dims == 3:
        ax.set_zlabel('3')

    if PLOTS:
        timer.start()
        plt.show()

    if acc:
        text = "{:.2f} %".format(acc*100)
        plt.text(0.5, 0.0, text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
   
    if img_name:
        plt.axis('off')
        plt.title(title)
        plt.savefig(f"{img_path}/{img_name}", dpi=300, bbox_inches='tight')

    plt.close('all')    
    gc.collect()
    return 

def print_paras(smi_gmlvq):
    print(smi_gmlvq.source_prototype_layer.get_weights()[0])
    print(smi_gmlvq.class_prototype_layer.get_weights()[0])
    print(smi_gmlvq.omega_layer.get_weights()[0])
    print(smi_gmlvq.omega_layer.classification_correlation_matrix().numpy()[0])
    print(smi_gmlvq.ComplementaryOrthogonalProjector()[0])



def gen_gifs(dirs=False):
    if dirs:
        dirs = [dirs]
    else:
        dirs = os.listdir("imgs/")
    for d in dirs:
        gifpath = "gifs/"
        gifname = f"{d}.gif"
        filepath = f"imgs/{d}/"

        filepath_sources = f"{filepath}sources/"
        filepath_classes = f"{filepath}classes/"

        if not os.path.isfile(gifpath+gifname):
            imgnames_sources = os.listdir(filepath_sources)
            imgnames_classes = os.listdir(filepath_classes)

            if len(imgnames_sources) <= 1 or len(imgnames_classes) <= 1:
                pass
            else:
                imgnames_sources.sort()
                imgnames_classes.sort()

                if len(imgnames_sources) != len(imgnames_classes):
                    gif_length = max(len(imgnames_sources), len(imgnames_classes)) - 1
                else:
                    gif_length = -1

                #compress = False

                images = []
                for s, c in zip(imgnames_sources[:gif_length], imgnames_classes[:gif_length]):
                    s = imageio.imread(filepath_sources+s)
                    c = imageio.imread(filepath_classes+c)
                    #if compress:
                    #    s = s[::2,::2]
                    #    c = c[::2,::2]
                    images.append(np.concatenate([s, c], axis=1))

                #if compress:
                #    imageio.mimsave(gifpath+"compressed-"+gifname, images)
                #else:
                imageio.mimsave(gifpath+gifname, images)

                """ streaming approach
                with imageio.get_writer('path.gif', mode="I") as writer:
                    for filename in filenames:
                        image = imageio.imread(filename)
                        writer.append_data(image)
                """

        if d == '2021-08-30_17-49-24':
            imgnames_sources = os.listdir(filepath_sources)
            imgnames_classes = os.listdir(filepath_classes)
            imgnames_sources.sort()
            imgnames_classes.sort()

            if len(imgnames_sources) != len(imgnames_classes):
                idx = min(len(imgnames_sources), len(imgnames_classes)) - 1
            else:
                idx = -2

            img = np.concatenate(
                    [imageio.imread(filepath_sources+imgnames_sources[idx]),
                     imageio.imread(filepath_classes+imgnames_classes[idx])],
                    axis=1)
            imageio.imsave('train-end-comb.png', img)


