from typing import final
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

    def _numOrgs(self, designCoords):
        #  return counts of unique y values in sorted order

        y = designCoords[:,1]
        OrgsCnts = Counter(y)
        # list of tuples of key and value: y value and associated count
        yValCnts = OrgsCnts.items()
        yValCntsSorted = sorted(yValCnts)
        return np.array(yValCntsSorted)[:,1]

    def _designColors(self, designCoords, finalCoords, yValCnts):
        designGrpd, colors = self._groupAndColor(designCoords, yValCnts)
        finalGrpd, finalColors = self._groupAndColor(finalCoords, yValCnts)
        return designGrpd, finalGrpd, colors, finalColors
    
    def _groupAndColor(self, coords, yValCnts):
        # group coordinates by y values, sort x values within groups, assign colors to groups
        coordsYsort = sorted(coords, key = lambda column: column[1])

        patternDict = {}
        colors_used = [0,0,0,0,0,0]

        nColors = len(self.colors)
        start = 0

        for ix, cnt in enumerate(yValCnts):
            cnt = int(cnt)
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
            if len(np.array(colsDict[key])) > 0:
                coordList.append(np.array(colsDict[key]))
        return np.vstack(coordList), colList

    def _fixColors(self, designCoords, finalCoords, colors):
        # Manually annotate to align color coding of two input patterns
        X = finalCoords[:,0]
        Y = finalCoords[:,1]
        xyVec = np.array([X,Y]).T
        numCoords = len(X)
        self._colorVec = colors.copy()

        refFig = plt.figure(1)
        plt.scatter(designCoords[:,0], designCoords[:,1], c = colors)
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
        
        #plt.axis('off')
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
            cnt = int(cnt)
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
            cnt = int(cnt)
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
            cnt = int(cnt)
            oldColor = col = re.sub(r'[0-9]+', '', oldColor)
            counter = 0
            for ix, newColor in enumerate(newColorsCopy):
                if newColor == oldColor:
                    correctIx.append(ix)
                    newColorsCopy[ix] = 'Taken'
                    counter += 1
                if counter == cnt:
                    break

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

    
    def removal(self, designCoords, finalCoords):

        designNum = len(designCoords)
        finalNum = len(finalCoords)

        yValCnts = self._numOrgs(designCoords)

        designGrpCoords, finalGrpCoords, dcolors, fcolors = self._designColors(designCoords, finalCoords, yValCnts)


        if designNum == finalNum:
            print("Move on to matching step")
            return designCoords, finalCoords
        elif designNum > finalNum:
            print("Remove from design, holding final fixed")
            new_designCoords = self._removeCentroids(finalGrpCoords, designGrpCoords, fcolors, dcolors, dfirst = False)
            return new_designCoords, finalCoords
        else:
            print("Remove from final, holding design fixed")
            new_finalCoords = self._removeCentroids(designGrpCoords, finalGrpCoords, dcolors, fcolors, dfirst=True)
            return designCoords, new_finalCoords
        

    def _removeCentroids(self, refCoords, coordsToRemove, colsRef, colsMod, dfirst): # add colors later?
        # Manually annotate to align color coding of two input patterns
        X = coordsToRemove[:,0]
        Y = coordsToRemove[:,1]
        xyVec = np.array([X,Y])
        numCoords = len(X)
        colorVec = ['b']*len(colsMod)
        removeLog = []

        if dfirst:
            refname = "Reference Image (Design)"
            modname = "Fix Image (Final)"

        else:
            refname = "Reference Image (Final)"
            modname = "Fix Image (Design)"

        refFig = plt.figure(1, figsize=(8,8))
        X_ref, Y_ref = refCoords[:, 0], refCoords[:, 1]
        plt.scatter(X_ref, Y_ref, c = colsRef) 
        #text = list(range(len(X_ref)))
        #for i, v in enumerate(text):
        #    plt.annotate(v, (X_ref[i], Y_ref[i] + 0.2))
        plt.title(refname)

        fixFig = plt.figure(2, figsize=(8,8))
        for i, (x, y) in enumerate(zip(X,Y)):
            plt.scatter(x, y,  picker=5, c = colorVec[i]) #colorVec[i]
        #plt.gca().invert_xaxis()
        #text = list(range(len(X)))
        #for i, v in enumerate(text):
        #    plt.annotate(v, (X[i], Y[i] + 0.2))
        plt.title(modname)

        def onclick(event):
            # redraw with changes
            if event.mouseevent.button == 1:
                event.artist.set_color('g')
                fixFig.canvas.draw()
            if event.mouseevent.button == 3:
                event.artist.set_color('r')
                fixFig.canvas.draw()

        def onclick_remove(event): 
            if event.dblclick:
                if event.button == 3:
                    # get click location
                    clickCoord = np.array((event.xdata, event.ydata))
                    print("removing coord at {}".format(clickCoord))
                    # find index of closest centroid
                    clickVec = np.tile(clickCoord, (numCoords, 1))
                    diff = clickVec - xyVec.T
                    sumSqDiff = np.sum(diff*diff, axis=1)
                    ix = np.argmin(sumSqDiff)
                    removeLog.append(ix)

        def on_close(event):
            figNum = event.canvas.figure.number
            print("Figures {} has been closed.".format(figNum))
            self._closedWindows += 1

        cid = fixFig.canvas.mpl_connect('button_press_event', onclick_remove)
        fixFig.canvas.mpl_connect('pick_event', onclick)
        refFig.canvas.mpl_connect('close_event', on_close)
        fixFig.canvas.mpl_connect('close_event', on_close)
        #plt.axis('off')
        plt.show()

        # if both windows close, end return annotation
        if self._closedWindows == 2:
            print("Exiting removal...")
            newCoords = coordsToRemove.copy()
            newCoords = np.delete(newCoords, removeLog, axis = 0)
            self._closedWindows = 0
            return newCoords


    def matching(self, designCoords, finalCoords, searchLen = None, normScalar = None, validation = False):
        # main function for matching two patterns

        assert(len(designCoords) == len(finalCoords))
        
        dir_path = os.getcwd()
        saveDir = os.path.join(dir_path, "datasets", self.saveDir)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        # if true, save images to a validation directory to be checked manually
        if validation:
            valDir = os.path.join(saveDir, "validationIms_" + self.saveDir)
            if not os.path.exists(valDir):
                os.makedirs(valDir)
        
        yValCnts = self._numOrgs(designCoords)

        designGrpCoords, finalGrpCoords, colors, _ = self._designColors(designCoords, finalCoords, yValCnts)

        assert(len(designGrpCoords) == len(finalGrpCoords) == len(colors))
        annotColors = self._fixColors(designGrpCoords, finalGrpCoords, colors)
        correctedFinalCoords = np.array(self._updateFinalCoords(finalGrpCoords, colors, annotColors, list(yValCnts)))
        
        if validation:
            print("Saving validation images in subdirectory...")
            for _ in range(30):
                num = np.random.randint(low = 0, high = len(correctedFinalCoords)-1, size=1) 

                fig, axes = plt.subplots(1,2, figsize=(16,8))
                axes[0].scatter(designGrpCoords[:,0], designGrpCoords[:,1], c=colors)
                axes[0].scatter(designGrpCoords[:,0][num], designGrpCoords[:,1][num], c='black', s=100)

                axes[1].scatter(correctedFinalCoords[:,0], correctedFinalCoords[:,1], c=colors)
                axes[1].scatter(correctedFinalCoords[:,0][num], correctedFinalCoords[:,1][num], c='black', s=100)

                fig.savefig(os.path.join(valDir, "organoid" + str(num)))
                plt.close(fig)

        df = np.hstack((designCoords, correctedFinalCoords))

        print("Saving matched dataframe...")
        np.save(os.path.join(saveDir, "matchDF"), df)

        #localities = self._getLocalities(df, searchLen, normScalar)
        #print("Saving localities and rewards...")
        #np.save(os.path.join(saveDir, "localsAndRewards"), localities)
