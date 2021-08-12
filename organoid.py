import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

class Organoid:
    def __init__(self, saveDir, saveLog = False):
        self.saveDir = saveDir
        self.saveLog = saveLog

    def coloring(self, X, Y, colors):
        assert(len(X) == len(Y) == len(colors))
        DataLog = []

        figColor = plt.figure()
        for i, (x,y) in enumerate(zip(X,Y)):
            plt.scatter(x, y, c = colors[i], picker=5)

        def onclick_color(event):
            print()
            print('*** MODIFYING COLOR ***')
            val = input("Enter value g/r/b: ")
            event.artist.set_color(val)
            figColor.canvas.draw()
            data = np.frombuffer(figColor.canvas.tostring_rgb(), dtype=np.uint8) #sep=''
            data = data.reshape(figColor.canvas.get_width_height()[::-1] + (3,))
            DataLog.append(data)
            print(len(DataLog))
            print(DataLog[-1].shape)
            plt.savefig(os.path.join(self.saveDir, "outputFig.png"), bbox_inches=0, pad_inches = 0)

        cid = figColor.canvas.mpl_connect('pick_event', onclick_color)

        plt.show()

    def removal(self, X, Y):
        assert(len(X) == len(Y))
        DataLog = []

        figRm = plt.figure()
        for i, (x,y) in enumerate(zip(X,Y)):
            plt.scatter(x, y, c = 'b', picker=5)

        def onclick_remove(event):
            event.artist.remove()
            figRm.canvas.draw()
            data = np.frombuffer(figRm.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(figRm.canvas.get_width_height()[::-1] + (3,))
            DataLog.append(data)
            plt.savefig(os.path.join(self.saveDir, "outputFig.png"), bbox_inches=0, pad_inches = 0)

        cid = figRm.canvas.mpl_connect('pick_event', onclick_remove)

        plt.show()

x = [1,2,3,1,2,3,1,2,3,3,4,5]
y = [2,2,2,4,4,4,6,6,6,8,8,8]
colors = ['b', 'b', 'b', 'g','g', 'g','r', 'r','r','b', 'b', 'b']
tubes = Organoid(saveDir='') # NEED TO WORK ON: saveLog=True

# TRY IT OUT
#--------------------
tubes.coloring(x,y,colors)
#tubes.removal(x,y)
