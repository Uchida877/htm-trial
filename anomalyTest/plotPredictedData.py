# -*- coding: utf-8 -*-

#実行方法
#python plotPredictedData.py sine(../data/*.csv) plotPredictedSine(predicted/*.csv)

import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def getCsvData(fileName):
    _EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.join(_EXAMPLE_DIR, os.pardir, "data/", argvs[1] + ".csv")
    #filePath = "data/raw/" + argvs[1] + "/" + fileName

    with open(filePath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # ヘッダーを読み飛ばしたい時
        reader.next()
        reader.next()

        data = []
        for row in reader:
            data.append(row)
    return data

def getCsvDataPredicted(fileName):
    filePath = "predicted/" + argvs[2] + ".csv"

    with open(filePath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # ヘッダーを読み飛ばしたい時
        reader.next()
        reader.next()

        data = []
        for row in reader:
            data.append(row)
    return data

def plotData(data):
    fig = plt.figure(figsize=(10,6))
    ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
    ax3 = plt.subplot2grid((2,2), (1,0), colspan=2)

    y = []
    y1 = [[]]
    y2 = [[]]
    csvFileNum = len(data)
    windowSize_x = 200
    countsForWindow = 0

    colorList = ["greenyellow", "aqua", "gold", "darkorange", "navy"]
    legendList = ["Raw data", "Predicted data", "Anomaly Score"]

    for i in range(len(data[0])):
        ax1.set_title('HTMTest')
        ax1.set_xlabel('Seconds')
        ax1.set_ylabel('Predicted')
        #ax1.legend(loc='lower right')

        ax1.set_xlim(0, windowSize_x)
        ax1.set_ylim(-1, 1)
        ax1.grid(True)


        #ax3.set_title('anomalyScore')
        ax3.set_xlabel('Seconds')
        ax3.set_ylabel('anomalyScore')
        ax3.set_xlim(0, windowSize_x)
        ax3.set_ylim(0, 1)
        ax3.grid(True)

        """if countsForWindow < windowSize_x:
            ax1.set_xlim(0, windowSize_x)
            ax3.set_xlim(0, windowSize_x)
        else:
            ax1.set_xlim(countsForWindow-windowSize_x, countsForWindow)
            ax3.set_xlim(countsForWindow-windowSize_x, countsForWindow)"""



        for j in range(csvFileNum):

            y1[j].append(float(data[j][i][1]))

            if len(y1) != csvFileNum + 1:
                y1.append([])

        y2[0].append(float(data[1][i][2]))

        if countsForWindow > 1200:
            if countsForWindow < windowSize_x:
                for j in range(csvFileNum):

                    ax1.plot( y1[j], linewidth=2, color=colorList[j], label = legendList[j])
                    ax1.legend(loc = 'upper left')

                ax3.plot( y2[0], linewidth=2, color="red", label = legendList[2])
                ax3.legend(loc = 'upper left')



                plt.pause(1e-7)  # 引数はsleep時間


            else:

                for j in range(csvFileNum):

                    ax1.plot( y1[j][countsForWindow-windowSize_x:countsForWindow], linewidth=2, color=colorList[j], label = legendList[j])
                    ax1.legend(loc = 'upper left')
                ax3.plot( y2[0][countsForWindow-windowSize_x:countsForWindow], linewidth=2, color="red", label = legendList[2])
                ax3.legend(loc = 'upper left')
                plt.pause(1e-7)  # 引数はsleep時間


            # 関数の最後で消去しないとうまくプロットされない
            ax1.cla()
            ax3.cla()
            #plt.cla(ax1)  # 現在描写されているグラフを消去

        countsForWindow += 1


if __name__ == "__main__" :

    argvs = sys.argv

    data = []

    data.append(getCsvData(argvs[2]))
    data.append(getCsvDataPredicted(argvs[2]))


    plotData(data)
