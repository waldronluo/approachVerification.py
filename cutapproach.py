import numpy
import scipy
import os

from scipy import io as sio
from matplotlib import pyplot as plt

#from derivative import derivative

def get_derivative(series):
    deriArr = []
    for i in range(series.shape[0] - 1):
        deri = numpy.divide(
                numpy.subtract(series[i+1][1:], series[i][1:]),
                numpy.subtract(series[i+1][0], series[i][0]))
        deri = numpy.insert(deri, 0, (series[i+1] + series[i]) / 2)
        deriArr.append(deri)
    return numpy.array(deriArr)

def get_features(contactSig):
    return numpy.concatenate([numpy.max(contactSig, axis=0) - numpy.min(contactSig, axis=0),
                             numpy.mean(contactSig, axis=0)])

def get_mask(derivative):
    absolute_derivative = numpy.absolute(derivative)[:, 1:]
    std_derivative = numpy.std(absolute_derivative, axis=0)
    mean_derivative = numpy.mean(absolute_derivative, axis=0)
    filter = numpy.add(mean_derivative, std_derivative)
    result = numpy.greater(absolute_derivative, filter)
    for i in range(numpy.shape(result)[0]):
        if not numpy.equal(numpy.sum(result[i]), 0):
            return (derivative[i, 0], derivative[-1,0])
    return derivative(-1, 0)

def get_sig(datfile, mask):
    return datfile[(datfile[:,0] >= mask[0])
            & (datfile[:,0] <= mask[1])]

def cut_approach(dataRootPath, dataBuildPath):
    dirs = os.listdir(dataRootPath)
    os.system("mkdir " + dataBuildPath + " -p")

    result = []
    result_group = []

    for dir in dirs:
        if(dir.startswith(r'exp') or dir.startswith(r'FC')
                or dir.startswith(r'success')):
            dataPath = os.path.join(dataRootPath, dir)
            datfile = numpy.loadtxt(fname=(dataPath + r'/Torques.dat'))
            statefile = numpy.loadtxt(fname=(dataPath + r'/State.dat'))

            get_sig(datfile, statefile[0:2])
            approachSig = get_sig(datfile, statefile[0:2])

            secondDerivative = get_derivative(get_derivative(approachSig))
            mask = get_mask(secondDerivative)
            contactSig = get_sig(datfile, mask)
            f = file(dataBuildPath + "/" + dir + ".npy", "w")
            numpy.save(f, contactSig)
            f.seek(0)
            features = get_features(contactSig[:, 1:])
            result.append(features)
            result_group.append(1 if dir.startswith(r'success') else 0)
            #for der in secondDerivative:
            #    print(der)
            #    raw_input("<ENTER> ->")

    for features in result:
        print(features)

if __name__ == "__main__":
    dataRootPath = r"../Error DAta/ErrorCharac/"
    dataBuildPath = r"./buildData"
    cut_approach(dataRootPath, dataBuildPath)
