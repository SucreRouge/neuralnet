/* 
 * File:   Layer.h
 * Author: Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 *
 * Created on 2017/05/29, 19:03
 */

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Neuron.h"
#include <fstream>   // ifstream, ofstream

using namespace std;

class Layer {
public:
    Layer(vector<int> topology, int layerNo, double adjusting);
    virtual ~Layer();
    void getLayerOutput(vector<double> outputs);
    void getLayerDelta(vector<double> deltas);
    void printOutput();
    void reviseNetworkWeights();
    void printAllWeights(ofstream &writing_file);
    
    vector<Neuron> neurons;
    vector<double> layerOutput;
    vector<double> layerDelta;
private:
    double adjusting;
    int layerNo;
};

#endif /* LAYER_H */

