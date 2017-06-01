/* 
 * File:   Layer.cpp
 * Author: Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 * 
 * Created on 2017/05/29, 19:03
 */

#include "Layer.h"
#include "Neuron.h"
#include <iostream>

//#define DEBUG false

using namespace std;

Layer::Layer(vector<int> topology, int layerNo, double adjusting) {
    this->adjusting = adjusting;
    this->layerNo = layerNo;
    // layerNo は層の番号
    for (int j = 0; j < topology.at(layerNo); j++) {
        if (layerNo == 0) {
            // 入力レイヤーの場合、前のレイヤーがない
            Neuron neuron = Neuron(0,topology.at(1),j,layerNo,adjusting);
            neuron.inputNeuron = true;
            neurons.push_back(neuron);
        } else {
            // 出力層の場合
            if(layerNo == topology.size()-1){
                Neuron neuron = Neuron(topology.at(layerNo - 1),0,j,layerNo,adjusting);
                neuron.outputNeuron = true;
                neurons.push_back(neuron);
            }else{
                Neuron neuron = Neuron(topology.at(layerNo - 1),topology.at(layerNo + 1),j,layerNo,adjusting);
                neurons.push_back(neuron);
            }
        }
    }
}

void Layer::printAllWeights(ofstream &writing_file){
    for(int i=0;i<neurons.size();i++){
        Neuron neuron = neurons.at(i);
        neuron.printWeightsIn(writing_file);
    }
}

void Layer::reviseNetworkWeights(){
    for(int i=0;i<neurons.size();i++){
        Neuron neuron = neurons.at(i);
        neuron.adjustingWeightsIn();
        neurons.at(i) = neuron;
    }
}

void Layer::getLayerDelta(vector<double> deltas){
    layerDelta.clear();
    if(DEBUG) cout << "Layer::getLayerDelta この層のニューロン数 = " << neurons.size() << endl;
    if(DEBUG) cout << "Layer::getLayerDelta 前の層のDelta値のサイズ = " << deltas.size() << endl;
    for (int j = 0; j < neurons.size(); j++) {
        //cout << "ニューロン [ " << j << " ] のDeltaを計算 " << endl;
        Neuron neuron = neurons.at(j);
        neuron.getNeuronDelta(deltas);
        //cout << "ニューロン [ " << j << " ] の Delta = " << neuron.delta << endl;
        layerDelta.push_back(neuron.delta);
        neurons.at(j) = neuron;
    }
}

void Layer::getLayerOutput(vector<double> outputs){
    layerOutput.clear();
    if(DEBUG) cout << "Layer::getLayerOutput この層のニューロン数 = " << neurons.size() << endl;
    if(DEBUG) cout << "Layer::getLayerOutput 前の層の出力サイズ = " << outputs.size() << endl;
    for (int j = 0; j < neurons.size(); j++) {
        Neuron neuron = neurons.at(j);
        neuron.getNeuronOutput(outputs);
        layerOutput.push_back(neuron.output);
        neurons.at(j) = neuron;
    }
}

void Layer::printOutput(){
    if(layerOutput.size() != neurons.size()){
        cout << "Layer::printOutput Error: I have no output data! layerOutput.size() = " << layerOutput.size() << " neurons.size() = " << neurons.size() << endl;
        return;
    }
    cout << "Layer::printOutput ";
    for(int i=0;i<neurons.size();i++){
        cout << i << ":" << layerOutput.at(i) << " ";
    }
    cout << endl;
}

Layer::~Layer() {
}

