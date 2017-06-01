/* 
 * File:   Net.h
 * Author: Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 *
 * Created on 2017/05/29, 6:32
 */

#ifndef NET_H
#define NET_H

#include <vector>
#include "Layer.h"

using namespace std;

class Net {
public:
    Net(vector<int> topology);
    virtual ~Net();
    void getForwardOutput(vector<double> initVal);
    void execBackpropagation(vector<double> deltaE);
    void printForwardOutput();
    Layer getOutputLayer();
    void adjustingPreLayerWeightsOut(int layerNo);
    void reviseAllNetworkWeights();
    void printAllWeights(int iteration, string dataFileName);
    vector<double> getOutput();

    vector<double> outputs;
private:
    vector<Layer> layers;
    vector<int> topology;
    double adjusting;
};

#endif /* NET_H */

