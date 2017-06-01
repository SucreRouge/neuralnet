/* 
 * File:   Neuron.h
 * Author: Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 * (C) 2017 Toyoaki WASHIDA
 *
 * Created on 2017/05/29, 5:06
 */

#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <fstream> 

using namespace std;

class Neuron {
public:
    Neuron(int inputNum, int outNum, int neuronNo, int layerNo, double adjusting);
    void getNeuronOutput(vector<double> prev_output);
    void getNeuronDelta(vector<double> prev_delta);
    void adjustingWeightsIn();
    void printWeightsIn(ofstream &writing_file);
    virtual ~Neuron();

    double output;
    double delta;
    bool outputNeuron;
    bool inputNeuron;
    // 入力側ニューロンからのウェイト
    vector<double> weightsIn;
    // 出力側ニューロンへのウェイト
    vector<double> weightsOut;
private:
    double value;
    // このニューロンのレイヤーの中での位置
    int neuronNo;
    int layerNo;
    double adjusting;
    // 一つ前からの出力を保存しておく
    vector<double> prev_output;
};

#endif /* NEURON_H */

