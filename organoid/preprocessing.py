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

    def grouping(self, coordsTrue, coordsPheno, plot=False):
        
        orgNumsYaxisAscending = self._numOrgs(coordsTrue)
        
        # outputs dictionary of color keys and associated coordinates
        groupsTrue = self._groupOrgs(coordsTrue, orgNumsYaxisAscending, showPlot = plot)
        groupsPheno = self._groupOrgs(coordsPheno, orgNumsYaxisAscending, showPlot = plot)
        
        # returns array of coordinates and list of colors
        c1,col1 = self._colorDict2Arr(groupsTrue)
        c2,col2 = self._colorDict2Arr(groupsPheno)
        return c1, col1, c2, col2

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

    def _groupOrgs(self, coords, cnts, showPlot = False):
        # sort y coordinates in ascending order 
        coordsYsort = sorted(coords, key = lambda x: x[1])

        # make dictionary for grouping organoids
        patternDict = {}
        colors_used = [0,0,0,0,0,0]

        # Colors used for plot and grouping organoids
        colors = ['m','g','b','y','r'] #, 'brown'
        
        if showPlot:
            plt.figure(figsize=(8,8))

        num_col = len(colors)
        start = 0
        # loop through y axis layers, plot groups and save in dictionary
        for ix, num in enumerate(cnts):
            col_ix = ix % num_col
            vals = np.array(coordsYsort[start:start+num])
            valsXsort = sorted(vals, key = lambda x: x[0])
            currColor = colors[col_ix]

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

            if showPlot:
                plt.scatter(np.array(vals)[:,0], np.array(vals)[:,1], c = currColor)
            
            start = start+num
        
        if showPlot:
            # flip y axis for plotting purposes
            plt.gca().invert_yaxis()
            plt.show()
        
        return patternDict