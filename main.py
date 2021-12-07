import TrainingSet as trs
import numpy as np
import matplotlib.pyplot as plt


def aufg_c():
    arrn = np.array([2, 3, 9])
    x_vec = np.arange(0, 6.01, 0.01)
    for n in arrn:
        trainclass = trs.ImportedTraining(n)
        trainclass.run()
        plt.plot(x_vec, trainclass.get_y_for_x_vector(x_vec))
        plt.xlabel("x")
        plt.ylabel("y")
    trainclass = trs.ImportedTraining(1)
    plt.scatter(trainclass.get_x(), trainclass.get_data(), c='r')
    plt.legend(["n=2", "n=3", "n=9"])
    plt.show()
    return


def aufg_d():
    rmse = []
    for n in range(9):
        t = trs.ImportedTraining(n+1)
        t.run()
        rmse.append(t.get_rmse_own_data())
    plt.bar(np.arange(1, 10), rmse, width=0.5, color='r')
    #plt.scatter(np.arange(1, 10), rmse, c='r')
    plt.ylabel("rsme")
    plt.xlabel("features n")
    plt.xticks(range(1, 10))
    plt.legend(["rsme of learned model"], loc='best')
    plt.show()
    return

def aufg_e():
    rmse = []
    rmseval = []
    for n in range(9):
        t = trs.ImportedTraining(n + 1)
        t.run()
        rmse.append(t.get_rmse_own_data())
        rmseval.append(t.get_rmse_validation_data())
    plt.bar(np.arange(1+0.125, 10+0.125), rmseval, width=-0.5, color='b', align='edge')
    #plt.scatter(np.arange(1-0.125, 10-0.125), rmseval, c='b')
    plt.bar(np.arange(1-0.125, 10-0.125), rmse, width=0.5, color='r', align='edge')
    #plt.scatter(np.arange(1+0.125, 10+0.125), rmse, c='r')
    plt.ylabel("rsme")
    plt.xlabel("features n")
    plt.xticks(range(1, 10))
    plt.legend(["rsme of learned model", "rmse val. data"], loc='best')
    plt.show()
    return

def aufg_f():
    mean_arr = []
    std_arr = []
    for n in range(1, 10):
        llo = trs.LooValidation(n)
        mean, var = llo.llo()
        mean_arr.append(mean)
        std_arr.append(var)

    plt.errorbar(range(1, 10), mean_arr, yerr=std_arr)
    plt.xlabel("features n")
    plt.ylabel("rsme")
    plt.legend(["rsme with error"])
    plt.show()
    return

def aufg_h():
    ker = trs.KernelRegression(30)
    print(ker.get_rmse_validation_data())
    plt.plot(ker.get_x(), ker.get_y_for_x_vector(ker.get_x()), color='r')
    plt.scatter(ker.get_x_saved(), ker.get_data_saved())
    #plt.scatter(ker.get_data_x(), ker.get_data(), c='r')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["predicted y", "data points"])
    plt.show()

plt.figure()
#aufg_c()
#aufg_d()
#aufg_e()
#aufg_f()
aufg_h()






