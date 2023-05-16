import pickle, fnmatch, os
import numpy as np

def mp_np(x):
    return np.array(x.tolist(),dtype=float)



def find(pattern, path, nosub = False):
    '''
    Returns list of filenames containing pattern in path.
    '''
    result = []
    if nosub:
        all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for name in all_files:
            if fnmatch.fnmatch(name,pattern):
                result.append(os.path.join(path, name))
    else:
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
    return result


def mysave(path,name,data):
    if os.path.exists(path):
        if os.path.exists(path+name):
            os.remove(path+name)
        with open(path+name, "wb") as fp:   #Pickling
            pickle.dump(data, fp)
    else:
        os.mkdir(path)
        with open(path + name, "wb") as fp:   #Pickling
            pickle.dump(data, fp)

def myload(name):
    with open(name, "rb") as fp:   # Unpickling
        all_stats = pickle.load(fp)
    return all_stats
