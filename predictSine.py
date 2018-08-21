import csv
import datetime
import os
import yaml
import sys
import numpy as np
import struct
import wave
import scipy.fftpack
import math
from itertools import islice
from params.model_params import MODEL_PARAMS
from nupic.frameworks.opf.model_factory import ModelFactory
from nupic.algorithms import anomaly_likelihood

predicted = []

_NUM_RECORDS = 33000
_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_FILE_PATH = os.path.join(_EXAMPLE_DIR, os.pardir, "data/", "sine.csv")
#_PARAMS_PATH = os.path.join(_EXAMPLE_DIR, os.pardir, "params", "model.yaml")



def createModel():
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)
    return ModelFactory.create(modelParams)



def runHotgym(numRecords):
    model = ModelFactory.create(MODEL_PARAMS)
    model.enableInference({"predictedField": "sine"})
    with open(_INPUT_FILE_PATH) as fin:
        reader = csv.reader(fin)
        headers = reader.next()
        reader.next()
        reader.next()

        results = []
        anomalyScore = []
        anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood()
        anomalyProbability = []
        count = 0

        rawData = []
        for record in islice(reader, numRecords):
            print count
            count += 1
            modelInput = dict(zip(headers, record))
            #print modelInput
            #sys.exit()
            modelInput["sine"] = float(modelInput["sine"])
            #modelInput["timestamp"] = datetime.datetime.strptime(modelInput["timestamp"], "%m/%d/%y %H:%M")
            result = model.run(modelInput)
            bestPredictions = result.inferences["multiStepBestPredictions"]
            allPredictions = result.inferences["multiStepPredictions"]
            rawAnomalyScore = result.inferences["anomalyScore"]
            anomalyScore.append(rawAnomalyScore)

            anomalyProbability.append(anomalyLikelihood.anomalyProbability(record[1], rawAnomalyScore))


            oneStep = bestPredictions[1]
            oneStepConfidence = allPredictions[1][oneStep]
            #fiveStep = bestPredictions[5]
            #fiveStepConfidence = allPredictions[5][fiveStep]

            #result = (oneStep, oneStepConfidence * 100, fiveStep, fiveStepConfidence * 100)
            #print "1-step: {:16} ({:4.4}%)\t 5-step: {:16} ({:4.4}%)".format(*result)
            predicted.append(oneStep)
            result = (oneStep, oneStepConfidence * 100)
            #print "1-step: {:16} ({:4.4}%)".format(*result)

            results.append(result)
            rawData.append(record[1])
        return results, anomalyScore, anomalyProbability, rawData

def saveCsv(outputFileName, data):
    file_handler = open(outputFileName + '.csv', 'w')
    writer = csv.writer(file_handler)
    #headers = ["seconds", "data", "data"]
    #types = ["float", "float", "float"]
    #flags = ["", "", ""]
    headers = ["seconds", outputFileName, "anomalyScore", "anomalyLikelihood"]
    types = ["float", "float", "int", "float"]
    flags = ["", "", "", ""]
    writer.writerow(headers)
    writer.writerow(types)
    writer.writerow(flags)
    for i in range(len(data[0])):
        writer.writerow([data[3][i], data[0][i][0], data[1][i], data[2][i]])
    file_handler.close()

def save(data, fs, bit, filename):

    #data = [int(v) for v in data]
    #data = struct.pack("h" * len(data), *data)
    print (fs)
    print (len(data))


    w = wave.Wave_write(filename + ".wav")
    w.setnchannels(1)
    w.setsampwidth(int(bit/8))
    w.setframerate(fs)
    w.writeframes(data)
    w.close()

def invMuLaw(quantized_signal, quantization_steps=256, format="16bit_pcm", sampling_rate=48000):

    print sampling_rate

    quantized_signal = quantized_signal.astype(float)
    normalized_signal = (quantized_signal / quantization_steps - 0.5) * 2.0

    # inv mu-law companding transformation (ITU-T, 1988)
    mu = quantization_steps - 1
    signals_1d = np.sign(normalized_signal) * ((1 + mu) ** np.absolute(normalized_signal)) / mu

    if format == "16bit_pcm":
    	max = 1<<15
    	type = np.int16
    elif format == "32bit_pcm":
    	max = 1<<31
    	type = np.int32
    elif format == "8bit_pcm":
    	max = 1<<8 - 1
    	type = np.uint8
    signals_1d *= max

    audio = signals_1d.reshape((-1, 1)).astype(type)
    #audio = np.repeat(audio, 2, axis=1)
    #wavfile.write("a.wav", sampling_rate, audio)
    save(audio, sampling_rate, 16, "genData2")

if __name__ == "__main__":
    argvs = sys.argv
    result = runHotgym(_NUM_RECORDS)

    saveCsv(argvs[1], result)

    #predicted = np.array(predicted)
    #invMuLaw(predicted, 256, "16bit_pcm", 16000)
