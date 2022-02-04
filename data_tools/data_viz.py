import matplotlib.pyplot as plt

def val_epochs(history, name,cut = 10):
    plt.figure()
    plt.title(name + ' Epoch Training')
    plt.ylabel('OME Test Error')
    plt.xlabel('Epochs')
    for hist in history:
        plt.plot(range(cut,len(hist.history['val_loss'])), hist.history['val_loss'][cut:])
    plt.legend(['Fold' + str(i) for i in range(len(history))], loc = (1.04,0))

def calc_slopes_Mw(X,Y):
    a_2 = (Y[-1] - Y[-5])/(X[-1] - X[-5])
    a_1 = (Y[5] - Y[0])/(X[5] - X[0])
    return a_1, a_2