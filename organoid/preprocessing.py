import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
import cv2
import pandas as pd
from collections import Counter
import re

class Alignment:
    _closedWindows = 0

    # color pattern to be used
    colors = ['m','g','b','y','r']
    
    _colorVec = []
    _index = 0

    def __init__(self, saveDir):
        self.saveDir = saveDir

    def matching(self, initCoords, finalCoords, saveDirName, expName, validation = False):
        # main function for matching two patterns
        
        dir_path = os.getcwd()
        saveDir = os.path.join(dir_path, saveDirName)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        if validation:
            valDir = os.path.join(saveDir, "validationIms_" + expName)
            if not os.path.exists(valDir):
                os.makedirs(valDir)
        
        yValCnts = self._numOrgs(initCoords)

        initGrpArr, finalGrpArr, colorArr = self._groupOrgs(initCoords, finalCoords, yValCnts)

        assert(initGrpArr.shape[0] == finalGrpArr.shape[0] == len(colorArr))

        #annotColor = self._coloring(initGrpArr, finalGrpArr, colorArr)

        """ corrections
        resArrs = np.array([initGrpArr[:,0], initGrpArr[:,1], colorArr, finalGrpArr[:,0], finalGrpArr[:,1], annotColor])
        cNames = ['xInit', 'yInit', 'colorInit', 'xPheno', 'yPheno', 'colorPheno']
        df = pd.DataFrame(data = resArrs.T, columns = cNames)

        # make color vector from matched color plots
        cnts = list(Counter(df.sort_values(by='yInit').yInit.values).values())
        colors = ['m','g','b','y','r']
        nCols = len(colors)
        colorsOrder = [colors[i % nCols] for i, _ in enumerate(cnts)]

        dfSorted_InitY = df.sort_values(by='yInit')
        initIx = []
        start = 0
        for cnt in cnts:
            subDf = dfSorted_InitY.iloc[start:start+cnt,:]
            initInds = subDf.sort_values(by='xInit').index.values
            initIx.extend(initInds)
            start += cnt

        dfSorted_PhenoY = df.sort_values(by='yPheno').reset_index()
        dfSorted_PhenoY["temp"] = [0]*dfSorted_PhenoY.shape[0]
        
        phenoIx = []
        for i, cnt in enumerate(cnts):
            c = colorsOrder[i]
            subDf = dfSorted_PhenoY[(dfSorted_PhenoY.colorPheno == c) & (dfSorted_PhenoY.temp == 0)].iloc[:cnt,:]
            
            # add indices to list
            phenoInds = subDf.sort_values(by='xPheno')["index"].values
            phenoIx.extend(phenoInds)
            
            # mark the indices which have been recorded and update
            temp = dfSorted_PhenoY.temp.values
            for ix in subDf.index.values:
                temp[ix] = 1    
            dfSorted_PhenoY["temp"] = temp 


        dfINIT = df.loc[initIx].iloc[:, 1:4].copy()
        dfPHENO = df.loc[phenoIx].iloc[:, 4:7].copy()
        
        init = dfINIT.reset_index(drop=True)
        final = dfPHENO.reset_index(drop=True)
        dfPaired = init.join(final)


        assert(np.all(dfPaired.colorInit.values == dfPaired.colorPheno.values))

        if validation:
            for _ in range(10):
                num = np.random.randint(low = 0, high = df.shape[0], size=1)

                fig, axes = plt.subplots(1,2, figsize=(16,8))
                axes[0].scatter(dfINIT.xInit.values, dfINIT.yInit.values, c=dfINIT.colorInit.values)
                axes[0].scatter(dfINIT.xInit.values[num], dfINIT.yInit.values[num], c='black', s=100)

                axes[1].scatter(dfPHENO.xPheno.values, dfPHENO.yPheno.values, c=dfPHENO.colorPheno.values)
                axes[1].scatter(dfPHENO.xPheno.values[num], dfPHENO.yPheno.values[num], c='black', s=100)

                fig.savefig(os.path.join(valDir, "organoid" + str(num)))

        dfPaired.to_csv(os.path.join(saveDir, 'dfPaired_'+expName+".csv"), index=False)
        """

    def _numOrgs(self, initCoords):
        #  return counts of unique y values in sorted order

        y = initCoords[:,1]
        OrgsCnts = Counter(y)
        # list of tuples of key and value: y value and associated count
        yValCnts = OrgsCnts.items()
        yValCntsSorted = sorted(yValCnts)
        return np.array(yValCntsSorted)[:,1]

    def _groupOrgs(self, initCoords, finalCoords, yValCounts):
        # sort coordinates by y values then x values
        # associate each unique y value with a color

        initCoordsYsort = sorted(initCoords, key = lambda x: x[1])
        finalCoordsYsort = sorted(finalCoords, key = lambda x: x[1])

        colorsUsed = [0,0,0,0,0,0]

        nColors = len(self.colors)
        start = 0

        initGrpdCoordList = []
        finalGrpdCoordList = []
        colors = []

        for ix, count in enumerate(yValCounts):
            colorIx = ix % nColors
            hi1 = self._xSort(initCoordsYsort, start, start + count)
            hi2 = self._xSort(finalCoordsYsort, start, start + count)
            initGrpdCoordList.append(hi1)
            finalGrpdCoordList.append(hi2)

            currColor = self.colors[colorIx]
            colors.extend([currColor] * count)

        return np.vstack(initGrpdCoordList), np.vstack(finalGrpdCoordList), colors
    
    def _xSort(self, coords, start, end):
        # extract and sort values by x axis
        groupByY = np.array(coords[start:end])
        groupXsorted = sorted(groupByY, key = lambda x: x[0])
        return groupXsorted

    def _coloring(self, initCoords, finalCoords, colors):
        # Manually annotate to align color coding of two input patterns

        X = finalCoords[:,0]
        Y = finalCoords[:,1]
        numCoords = len(X)

        self._colorVec = colors.copy()

        # plot reference and annotation image
        refFig = plt.figure(1)
        plt.scatter(initCoords[:,0], initCoords[:,1], c = colors)
        plt.title("REFERENCE IMAGE")

        figColor = plt.figure(2, figsize=(8,8))
        for i, (x,y) in enumerate(zip(X,Y)):
            plt.scatter(x, y, c = colors[i], picker=5)

        def onclick(event):
            # get click location
            xclick = np.round(event.mouseevent.xdata,3)
            yclick = np.round(event.mouseevent.ydata, 3)
            clickCoord = np.array((xclick, yclick))
            print(xclick, yclick)

            print('Color mod options: magenta/green/blue/yellow/red')
            val = input("Enter value m/g/b/y/r: ")

            # record changes 
            clickVec = np.tile(clickCoord, (numCoords, 1))
            diff = clickVec - finalCoords
            sumSqDiff = np.sum(diff*diff, axis=1)
            ix = np.argmin(sumSqDiff)
            self._index = ix
            print("Current Color is {}.".format(self._colorVec[ix]))
            self._colorVec[ix] = val
            print("New Color is {}.".format(self._colorVec[ix]))

            # redraw with changes
            event.artist.set_color(val)
            figColor.canvas.draw()
            
        def on_close(event):
            figNum = event.canvas.figure.number
            print("Figures {} has been closed.".format(figNum))
            self._closedWindows += 1

        cid = figColor.canvas.mpl_connect('pick_event', onclick)
        refFig.canvas.mpl_connect('close_event', on_close)
        figColor.canvas.mpl_connect('close_event', on_close)
        
        plt.axis('off')
        plt.show()

        # if both windows close, end return annotation
        if self._closedWindows == 2:
            return self._colorVec

    

    

    