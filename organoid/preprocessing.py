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
    colors = ['m','g','b','y','r']
    # testing
    _colorVec = []
    _index = 0

    def __init__(self, saveDir):
        self.saveDir = saveDir

    def coloring(self, coordsTrue, colorsTrue, coordsPheno, colorsPheno):
        # x,y coordinates of phenotype images
        X = coordsPheno[:,0]
        Y = coordsPheno[:,1]
        numCoords = len(X)
        assert(numCoords == len(Y) == len(colorsPheno))

        self._colorVec = colorsPheno.copy()

        refFig = plt.figure(1)
        plt.scatter(coordsTrue[:,0], coordsTrue[:,1], c = colorsTrue)
        plt.title("REFERENCE IMAGE")

        figColor = plt.figure(2, figsize=(8,8))
        for i, (x,y) in enumerate(zip(X,Y)):
            plt.scatter(x, y, c = colorsPheno[i], picker=5)

        def onclick(event):
            # get click location
            xclick = np.round(event.mouseevent.xdata,3)
            yclick = np.round(event.mouseevent.ydata, 3)
            clickCoord = np.array((xclick, yclick))
            print(xclick, yclick)

            # get user input
            print('Color mod options: magenta/green/blue/yellow/red')
            val = input("Enter value m/g/b/y/r: ")

            # record changes 
            clickVec = np.tile(clickCoord, (numCoords, 1))
            diff = clickVec - coordsPheno
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

        if self._closedWindows == 2:
            return self._colorVec


    # TO FIX LATER
    def removal(self, coords):
        '''
        X = coords[:,0]
        Y = coords[:,1]
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
        '''

    def matching(self, coordsTrue, coordsPheno, saveDirName, expName, validation = False):

        dir_path = os.getcwd()
        saveDir = os.path.join(dir_path, saveDirName)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        valDir = os.path.join(saveDir, "validationIms_" + expName)
        if not os.path.exists(valDir):
            os.makedirs(valDir)
        
        orgNumsYaxisAscending = self._numOrgs(coordsTrue)
        
        # outputs dictionary of color keys and associated coordinates
        groupsTrue = self._groupOrgs(coordsTrue, orgNumsYaxisAscending)
        groupsPheno = self._groupOrgs(coordsPheno, orgNumsYaxisAscending)
        
        # returns array of coordinates and list of colors
        initCoord, initColor = self._colorDict2Arr(groupsTrue)
        phenoCoord, phenoColor = self._colorDict2Arr(groupsPheno)

        newColor = self.coloring(initCoord, initColor, phenoCoord, phenoColor)

        resArrs = np.array([initCoord[:,0], initCoord[:,1], initColor, phenoCoord[:,0], phenoCoord[:,1], newColor])
                            #, np.array(newColor)]
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

    def _colorDict2Arr(self, colsDict):
        keys = list(colsDict.keys())

        coordList = []
        colList = []
        for key in keys:
            col = re.sub(r'[0-9]+', '', key)
            colList.extend([col for _ in range(len(colsDict[key]))])
            coordList.append(np.array(colsDict[key]))

        return np.vstack(coordList), colList

    def _numOrgs(self, designCoords):
        # get counts of organoids with y axis that have the same values
        OrgsCnts = Counter(designCoords[:,1])
        # sort y axis keys in ascending order
        OrgsCntsSorted = sorted(OrgsCnts.items())
        # return sorted counts
        return np.array(OrgsCntsSorted)[:,1]

    def _groupOrgs(self, coords, cnts):
        # sort y coordinates in ascending order 
        coordsYsort = sorted(coords, key = lambda x: x[1])

        # make dictionary for grouping organoids
        patternDict = {}
        colors_used = [0,0,0,0,0,0]

        # Colors used for plot and grouping organoids
         #, 'brown'

        num_col = len(self.colors)
        start = 0
        # loop through y axis layers, plot groups and save in dictionary
        for ix, num in enumerate(cnts):
            col_ix = ix % num_col
            vals = np.array(coordsYsort[start:start+num])
            valsXsort = sorted(vals, key = lambda x: x[0])
            currColor = self.colors[col_ix]

            # if color not in dictionary, add color
            # else denote its index e.g. blue3
            if currColor not in patternDict:
                patternDict[currColor] = valsXsort
            else:
                key_num = colors_used[col_ix]
                newKey_num = key_num + 1
                key_name = currColor + str(newKey_num)
                patternDict[key_name] = valsXsort
                colors_used[col_ix] = newKey_num
            
            start = start+num
        
        return patternDict