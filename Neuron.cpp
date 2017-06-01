/* 
 * File:   Neuron.cpp
 * Author: Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 * 
 * Created on 2017/05/29, 5:06
 */

#include "Neuron.h"
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <iomanip>

//#define DEBUG false
extern vector< vector< vector<double> > > reserved_weights;

using namespace std;

Neuron::Neuron(int inputNum, int outNum, int neuronNo, int layerNo, double adjusting) {
    if (inputNum > 0) {
        if (!reserved_weights.empty() && reserved_weights.size() > 0) {
            // 保存されていたウェイトを使用する場合
            for (int i = 0; i < inputNum; i++) {
                if(DEBUG) cout << "Layer[" << layerNo << "] Neuron [" << neuronNo << "] のNo.[" << i << "]のweight[" << reserved_weights[layerNo-1][neuronNo][i] << "]をセット" << endl;
                weightsIn.push_back(reserved_weights[layerNo-1][neuronNo][i]);
            }
        } else {
            for (int i = 0; i < inputNum; i++) {
                // ウェイトの初期値は乱数で与える
                // スケールが大きすぎると不都合が発生するのでinputNumで割る
                weightsIn.push_back((double) rand() / ((double) RAND_MAX * (double) inputNum));
            }
        }
    }
    if(outNum > 0){
        // 出力側のウェイトの初期値はなんでも良い
        // バックプロパゲーションの際に上の層のニューロンによって書き換えられる
        for (int i = 0; i < outNum; i++) {
            weightsOut.push_back(1.0);
        }
    }
    inputNeuron = false;
    outputNeuron = false;
    this->neuronNo = neuronNo;
    this->layerNo = layerNo;
    this->adjusting = adjusting;
}

void Neuron::printWeightsIn(ofstream &writing_file){
    // ウェイトのファイルへの書き出し
    for(int i=0;i<weightsIn.size();i++){
        writing_file << "Weight: " << i << " -> " << layerNo << " : " << neuronNo << " = " << fixed << setprecision(10) << weightsIn.at(i) << endl;
    }
}

void Neuron::adjustingWeightsIn(){
    for(int i=0;i<weightsIn.size();i++){
        if(DEBUG) cout << "Neuron::adjustingWeightsIn prev_output.at(i) = " << prev_output.at(i) << " delta = " << delta << " org weight = " <<  weightsIn.at(i) <<  endl;
        weightsIn.at(i) = weightsIn.at(i)-adjusting*delta*prev_output.at(i);
        if(DEBUG) cout << "Neuron::adjustingWeightsIn L:N[" << layerNo << ":" << neuronNo << "] 入力ウェイト No.["<< i << "]を調整 weight = " << weightsIn.at(i) << endl;
    }
}

void Neuron::getNeuronDelta(vector<double> prev_delta){
    if(prev_delta.empty() || prev_delta.size() == 0){
        cout << "Neuron::getNeuronDelta エラー: ニューロンのデルタ値の計算に必要な入力データがありません" << endl;
        return;
    }
    if(outputNeuron){
        // 出力ニューロンだった場合
        double ev = exp(-value);
        delta = ev/((1+ev)*(1+ev));
        delta = delta*prev_delta.at(neuronNo); // 損失関数の微分値
        if(DEBUG) cout << "Neuron::getNeuronDelta 出力ニューロン のDeltaを計算 value = " << value << " prev_delta.at(neuronNo) = " << prev_delta.at(neuronNo) << endl;
    }else{
        //cout << "ニューロン のDeltaを計算（１） prev_delta.size() = " << prev_delta.size() << endl;
        delta = 0.0;
        for(int i=0;i<prev_delta.size();i++){
           delta += prev_delta.at(i)*weightsOut.at(i); 
           if(DEBUG) cout << "Neuron::getNeuronDelta prev_delta.at(" << i << ") = " << prev_delta.at(i) 
                   << " weightsOut.at(" << i << ") = " << weightsOut.at(i) << endl;
        }
        double ev = exp(-value);
        delta = delta*(ev/((1+ev)*(1+ev)));
        //cout << "ニューロン のDeltaを計算（２）終了" << endl;
    }
    if(DEBUG) cout << "Neuron::getNeuronDelta レイヤー No.[ " << layerNo << " ]  出力ニューロン No.[ " << neuronNo << " ] の 保存されていた value = " << value << " Delta = " << delta << endl;
    //adjustingWeightsIn();
}

void Neuron::getNeuronOutput(vector<double> prev_output){
    this->prev_output = prev_output;
    if(prev_output.empty() || prev_output.size() == 0){
        cout << "Neuron::getNeuronOutput エラー: ニューロンの出力値の計算に必要な入力データがありません" << endl;
        return;
    }
    value = 0.0;
    for (int i = 0; i < weightsIn.size(); i++) {
        if(DEBUG) cout << "Neuron::getNeuronOutput L:N[" << layerNo << ":" << neuronNo << "] weightsIn.at(" << i << ") = " << weightsIn.at(i)
                << " prev_output.at(" << i << ") = " << prev_output.at(i) << endl;
        value += prev_output.at(i) * weightsIn.at(i);
    }
    output = 1 / (1 + exp(-value));
    //cout << "レイヤー No.[ " << layerNo << " ] ニューロン No. [ " << neuronNo << " ] の value = " << value << " output = " << output << endl;
}

Neuron::~Neuron() {
}

