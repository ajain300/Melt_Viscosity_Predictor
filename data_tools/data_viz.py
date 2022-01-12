import matplotlib.pyplot as plt

def val_epochs(history, name,cut = 10):
    plt.figure()
    plt.title(name + ' Epoch Training')
    plt.ylabel('MSE Test Error')
    plt.xlabel('Epochs')
    for hist in history:
        plt.plot(range(cut,len(hist.history['val_loss'])), hist.history['val_loss'][cut:])
    plt.legend(['Fold' + str(i) for i in range(len(history))], loc = (1.04,0))
