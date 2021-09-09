import matplotlib.pyplot as plt
import numpy as np
import os
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

    def _numOrgs(self, initCoords):
        #  return counts of unique y values in sorted order

        y = initCoords[:,1]
        OrgsCnts = Counter(y)
        # list of tuples of key and value: y value and associated count
        yValCnts = OrgsCnts.items()
        yValCntsSorted = sorted(yValCnts)
        return np.array(yValCntsSorted)[:,1]

    def _initColors(self, initCoords, finalCoords, yValCnts):
        initGrpd, colors = self._groupAndColor(initCoords, yValCnts)
        finalGrpd, finalColors = self._groupAndColor(finalCoords, yValCnts)
        assert(colors == finalColors)
        return initGrpd, finalGrpd, colors
    
    def _groupAndColor(self, coords, yValCnts):
        # group coordinates by y values, sort x values within groups, assign colors to groups
        coordsYsort = sorted(coords, key = lambda column: column[1])

        patternDict = {}
        colors_used = [0,0,0,0,0,0]

        nColors = len(self.colors)
        start = 0

        for ix, cnt in enumerate(yValCnts):
            colIx = ix % nColors
            vals = np.array(coordsYsort[start:start+cnt])
            valsXsort = sorted(vals, key = lambda x: x[0])
            currColor = self.colors[colIx]
            currColor = self.colors[colIx]

            if currColor not in patternDict:
                patternDict[currColor] = valsXsort
            else:
                keyNum = colors_used[colIx]
                newKeyNum = keyNum + 1
                keyName = currColor + str(newKeyNum)
                patternDict[keyName] = valsXsort
                colors_used[colIx] = newKeyNum

            start = start + cnt

        return self._colorDict2Arr(patternDict)

    def _colorDict2Arr(self, colsDict):
        keys = list(colsDict.keys())
        coordList = []
        colList = []
        for key in keys:
            col = re.sub(r'[0-9]+', '', key)
            colList.extend([col for _ in range(len(colsDict[key]))])
            coordList.append(np.array(colsDict[key]))
        return np.vstack(coordList), colList

    def _fixColors(self, initCoords, finalCoords, colors):
        # Manually annotate to align color coding of two input patterns
        X = finalCoords[:,0]
        Y = finalCoords[:,1]
        xyVec = np.array([X,Y]).T
        numCoords = len(X)
        self._colorVec = colors.copy()

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
            diff = clickVec - xyVec
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
            print("### Exiting manual annotation ###")
            print()
            return self._colorVec
    
    def _colorArr2Dict(self, colors, yValCnts):
        colorDict = {}
        colors_used = {'m':0,'g':0,'b':0,'y':0,'r':0}

        nColors = len(self.colors)
        start = 0

        for ix, cnt in enumerate(yValCnts):
            currColor = colors[start]

            if currColor not in colorDict:
                colorDict[currColor] = cnt
            else:
                colors_used[currColor] += 1
                keyName = currColor + str(colors_used[currColor])
                colorDict[keyName] = cnt

            start = start + cnt

        return colorDict

    def _sortX(self, coords, yValCnts):
        reOrderCoords = []
        
        start = 0
        for cnt in yValCnts:
            subCoords = coords[start: start + cnt]
            sortedSubCoords = sorted(subCoords, key = lambda column: column[0])
            reOrderCoords.append(sortedSubCoords)
        
            start = start + cnt

        return np.vstack(reOrderCoords)
    
    def _updateFinalCoords(self, coords, oldColors, newColors,  yValCnts):

        colorDict = self._colorArr2Dict(oldColors, yValCnts)
        newColorsCopy = newColors.copy() 
        correctIx = []
        for oldColor, cnt in colorDict.items():
            oldColor = col = re.sub(r'[0-9]+', '', oldColor)
            counter = 0
            for ix, newColor in enumerate(newColorsCopy):
                if newColor == oldColor:
                    correctIx.append(ix)
                    newColorsCopy[ix] = 'Taken'
                    counter += 1
                if counter == cnt:
                    break
        
        assert(correctIx != list(range(len(oldColors))))

        correctedNewColors = [newColors[ix] for ix in correctIx]
        correctedFinalCoords = [coords[ix] for ix in correctIx]
        
        assert(len(correctIx) == len(oldColors))
        assert(oldColors == correctedNewColors)

        correctedFinalCoords = self._sortX(correctedFinalCoords, yValCnts)

        return correctedFinalCoords

    def _getLocalities(self, df, searchLen, normScalar):
        x = df[:, 0].astype(int) // normScalar
        y = df[:, 1].astype(int) // normScalar
        
        xRange = np.max(x) - np.min(x)
        yRange = np.max(y) - np.min(y)
        maxRange = np.max((xRange, yRange))
        
        normSearchLen = searchLen // normScalar
        
        maskDim =  maxRange + 4*normSearchLen
        mask = np.zeros((maskDim, maskDim))
        
        centroids = []
        rewards = []
        localities = []
        
        # draw 1 by 1 organoids on mask with padding
        for ix, coord in enumerate(zip(x, y)):
            cx = coord[0] + 2*normSearchLen
            cy = coord[1] + 2*normSearchLen
            centroids.append((cx,cy))
            mask[cx, cy] = 1
            
            rewards.append(df[:, -1][ix])

        # extract localities 
        for ix, c in enumerate(centroids):
            cx, cy = c[0], c[1]
            locality = mask[cx-normSearchLen: cx+normSearchLen+1, cy-normSearchLen: cy+normSearchLen+1]
            localities.append(np.append(locality.flatten(), rewards[ix]))
            
        return localities

    def matching(self, initCoords, finalCoords, expName, searchLen, normScalar, validation = False):
        # main function for matching two patterns
        
        dir_path = os.getcwd()
        saveDir = os.path.join(dir_path, "datasets", self.saveDir)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        if validation:
            valDir = os.path.join(saveDir, "validationIms_" + expName)
            if not os.path.exists(valDir):
                os.makedirs(valDir)
        
        yValCnts = self._numOrgs(initCoords)

        initGrpCoords, finalGrpCoords, colors = self._initColors(initCoords, finalCoords, yValCnts)

        assert(initGrpCoords.shape[0] == finalGrpCoords.shape[0] == len(colors))

        annotColors = self._fixColors(initGrpCoords, finalGrpCoords, colors)

        correctedFinalCoords = np.array(self._updateFinalCoords(finalGrpCoords, colors, annotColors, list(yValCnts)))

        if validation:
            print("Saving validation images in subdirectory...")
            for _ in range(30):
                num = np.random.randint(low = 0, high = len(correctedFinalCoords)-1, size=1) 

                fig, axes = plt.subplots(1,2, figsize=(16,8))
                axes[0].scatter(initCoords[:,0], initCoords[:,1], c=colors)
                axes[0].scatter(initCoords[:,0][num], initCoords[:,1][num], c='black', s=100)

                axes[1].scatter(correctedFinalCoords[:,0], correctedFinalCoords[:,1], c=colors)
                axes[1].scatter(correctedFinalCoords[:,0][num], correctedFinalCoords[:,1][num], c='black', s=100)

                fig.savefig(os.path.join(valDir, "organoid" + str(num)))

        df = np.hstack((initCoords, correctedFinalCoords))

        print("Saving matched dataframe...")
        np.save(os.path.join(saveDir, "matchDF"), df)

        localities = self._getLocalities(df, searchLen, normScalar)
        print("Saving localities and rewards...")
        np.save(os.path.join(saveDir, "localsAndRewards"), localities)
