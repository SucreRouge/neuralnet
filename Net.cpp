/* 
 * File:   Net.cpp
 * Author: Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 * 
 * Created on 2017/05/29, 6:32
 */

#include <iostream>
#include "Net.h"
#include "Layer.h"
#include "Neuron.h"
#include <fstream>
#include <time.h>

using namespace std;

void showVectorVals(string label, vector<double> v);

Net::Net(vector<int> topology) {
    this->topology = topology;
    adjusting = 0.15; // ウェイトを改訂する程度
    // topologyに基づいて、ネットワークを構成する
    for(int i=0;i<topology.size();i++){
        Layer layer(topology,i,adjusting);
        layers.push_back(layer);
    }
    // 入力側のウェイトと出力側のウェイトの整合性を図る
    for(int i=topology.size()-1;i>0;i--){
        adjustingPreLayerWeightsOut(i);
    }    
}

Layer Net::getOutputLayer(){
    return layers.at(topology.size()-1);
}

void Net::printAllWeights(int iteration, string dataFileName) {
    time_t timer;
    struct tm* tm;
    char datetime[20];
    timer = time(NULL);
    tm = localtime(&timer);
    strftime(datetime, 20, "%Y%m%d%H%M%S", tm);
    //cout << "DATETIME = " << datetime << endl;
    string dt = datetime;
    string filename = "weight_"+dt+"."+"wgt";
    ofstream writing_file;
    writing_file.open(filename, ios::out);

    cout << "形成されたウェイトをファイル[ " << filename <<  " ]に書き出します" << endl;
    writing_file << "File name: [ " << filename << " ]" << endl;
    writing_file << "Iteration: " << iteration << endl;
    writing_file << "InputData: " << dataFileName << endl;
    writing_file << "Topology: ";
    for(int i=0;i<topology.size();i++) writing_file << topology.at(i) << " ";
    writing_file << endl;
    writing_file << "Adjusting: " << adjusting << endl;
    writing_file << "Label: Pre_neuron No." << " -> " << " Layer No. : Neuron No. = Weight" << endl;
    for(int i=1;i<topology.size();i++){
        Layer layer = layers.at(i);
        layer.printAllWeights(writing_file);
    }    
}

void Net::adjustingPreLayerWeightsOut(int layerNo){
    Layer this_layer = layers.at(layerNo);
    vector<Neuron> this_neurons = this_layer.neurons;
    Layer pre_layer = layers.at(layerNo-1);
    vector<Neuron> pre_neurons = pre_layer.neurons;
    for(int i=0;i<pre_neurons.size();i++){
        Neuron pre_neuron = pre_neurons.at(i);
        for(int j=0;j<this_neurons.size();j++){
            Neuron this_neuron = this_neurons.at(j);
            // 一つ前のレイヤー(i番目)の出力側j番目のニューロンに向けてのウェイトは、
            // このレイヤー(j番目)の入力側第i番目のニューロンからのウェイトに同じ
            // だから、コピーする
            if(DEBUG) cout << "Net::adjustingPreLayerWeightsOut L:N[" << layerNo-1 << ":" << i << "] ==>> " << "L:N[" << layerNo << ":" << j << "] 出力ウェイト調整 = " << this_neuron.weightsIn.at(i) <<endl;
            pre_neuron.weightsOut.at(j) = this_neuron.weightsIn.at(i);
        }
        // 書き戻しを忘れない
        pre_neurons.at(i) = pre_neuron;
    }
    // 書き戻しを忘れない
    pre_layer.neurons = pre_neurons;
    layers.at(layerNo-1) = pre_layer;
}

void Net::reviseAllNetworkWeights(){
    for(int i=1;i<topology.size();i++){
        Layer layer = layers.at(i);
        layer.reviseNetworkWeights();
        layers.at(i) = layer;
    }
}

void Net::execBackpropagation(vector<double> deltaE){
    // バックプロパゲーションの実行
    Layer layerOut = layers.at(topology.size()-1);
    layerOut.layerDelta.clear();
    if(DEBUG) cout << "Net::execBackpropagation 出力レイヤーのDelta値を計算する:他と計算方法が異なるので" << endl;
    for(int i=0;i<layerOut.neurons.size();i++){
        Neuron neuron = layerOut.neurons.at(i);
        neuron.getNeuronDelta(deltaE);
        layerOut.layerDelta.push_back(neuron.delta);
        layerOut.neurons.at(i) = neuron;
    }
    layers.at(topology.size()-1) = layerOut;
    if(DEBUG) showVectorVals("Net::execBackpropagation 出力層のDelta値",layerOut.layerDelta);
    // 逆方向に計算を進める
    // 入力層については計算しない：ウェイトは2層分しかないので
    // i=0 はない
    for(int i=topology.size()-2;i>=1;i--) {
        Layer layer = layers.at(i);
        Layer prev = layers.at(i+1);
        if(DEBUG) showVectorVals("Net::execBackpropagation 一つ前の層のDelta",prev.layerDelta);
        if(DEBUG) cout << "Net::execBackpropagation 層 [ " << i << " ] のDelta計算" << endl;
        layer.getLayerDelta(prev.layerDelta);
        layers.at(i) = layer;
        adjustingPreLayerWeightsOut(i);
        if(DEBUG) showVectorVals("Net::execBackpropagation このレイヤーのDelta",layer.layerDelta);
    }
    // ウェイトの変更は、デルタを全て作り終えてからにしなければならない
    // ウェイトの完全変更
    reviseAllNetworkWeights();
    // 入力側のウェイトと出力側のウェイトの整合性を図る
    for(int i=topology.size()-1;i>0;i--){
        adjustingPreLayerWeightsOut(i);
    }    
}

void Net::printForwardOutput(){
    // 順伝搬出力の表示
    // 最終レイヤーを取り出す
    Layer layer = layers.at(layers.size()-1);
    layer.printOutput();
}

vector<double> Net::getOutput(){
    // 最終レイヤーを取り出す
    Layer layer = layers.at(layers.size()-1);
    return layer.layerOutput;
}

void Net::getForwardOutput(vector<double> initVal) {
    if(DEBUG) cout << "Net::getForwardOutput Network Topology Size = " << topology.size() << endl;
    // 入力を直接受け取るレイヤーの出力値を計算する
    Layer layer0 = layers.at(0);
    layer0.layerOutput.clear();
    if(DEBUG) cout << "Net::getForwardOutput 入力層の出力は、入力データそのまま" << endl;
    for(int i=0;i<layer0.neurons.size();i++){
        Neuron neuron = layer0.neurons.at(i);
        layer0.layerOutput.push_back(neuron.output = initVal.at(i));
        layer0.neurons.at(i) = neuron;
    }
    layers.at(0) = layer0;
    if(DEBUG) showVectorVals("Net::getForwardOutput 入力層からの出力",layer0.layerOutput);
    for(int i=1;i<topology.size();i++) {
        Layer layer = layers.at(i);
        Layer prev = layers.at(i-1);
        if(DEBUG) showVectorVals("Net::getForwardOutput 一つ前の層からの出力",prev.layerOutput);
        if(DEBUG) cout << "Net::getForwardOutput 層 [ " << i << " ] の出力計算" << endl;
        layer.getLayerOutput(prev.layerOutput);
        layers.at(i) = layer;
    }    
}

Net::~Net() {
}
