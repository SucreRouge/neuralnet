/* 
 * File:   main.cpp
 * Author: Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 *
 * Created on 2017/05/28, 23:50
 */

#include <cstdlib>
#include "Net.h"
#include "Data.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <stdlib.h>

using namespace std;

// レイヤー、ニューロン、ウェイト
vector< vector< vector<double> > > reserved_weights;

int openWeightFile(string weight_path){
    vector<string> terms;
    if(weight_path.length() > 0){
        boost::algorithm::split(terms, weight_path, boost::is_any_of("."));
        if(terms.at(terms.size()-1) != "wgt"){
            cout << "正しいweightファイルではありません：中止" << endl;
            return 0;
        }
    }
    ifstream weight_file;
    weight_file.open(weight_path, ios::in);
    if (weight_file.fail()){
        cout << "weightファイルを開くのに失敗しました" << endl;
        return 0;
    }
    string line;
    int layerNo = 1;
    int neuronNo = 0;
    vector<double> weights;
    vector<vector<double> > neuron_weights;
    int iteration = 0;
    cout << "Layer [ " << layerNo << " ] を読み込みます" << endl;
    while (getline(weight_file, line)){
        boost::algorithm::split(terms, line, boost::is_any_of(" "));
        // ウェイトは、きちんと順番に入っているはず
        if(terms.at(0) == "Weight:"){
            int lnum = stoi(terms.at(3));
            int nnum = stoi(terms.at(5));
            double weight = stod(terms.at(7));
            if(lnum != layerNo){
                // レイヤー番号が変わった
                // 当然ニューロン番号も変わっているはず
                neuron_weights.push_back(weights);
                if(DEBUG) cout << "読み込んだウェイト数 = " << weights.size() << endl; 
                if(DEBUG) cout << "読み込んだニューロン数 = " << neuron_weights.size() << endl; 
                reserved_weights.push_back(neuron_weights);
                neuron_weights.clear();
                weights.clear();
                weights.push_back(weight);
                layerNo = lnum;
                neuronNo = nnum;
                if(DEBUG) cout << "Layer [ " << layerNo << " ] を読み込みます" << endl;
            }else if(nnum != neuronNo){
                // ニューロン番号だけ変わった
                if(DEBUG) cout << "読み込んだウェイト数 = " << weights.size() << endl; 
                neuron_weights.push_back(weights);
                weights.clear();
                weights.push_back(weight);
                neuronNo = nnum;
            }else{
                // どっちも変わらない
                weights.push_back(weight);
            }
        }else if(terms.at(0) == "Iteration:"){
            boost::algorithm::split(terms, line, boost::is_any_of(" "));
            iteration = stoi(terms.at(1));
            if(DEBUG) cout << "weightファイルから読み込んだ繰り返し数 = " << iteration << endl;
        }
    }
    // 最後のレイヤーのウェイト
    if(DEBUG) cout << "読み込んだウェイト数 = " << weights.size() << endl;
    neuron_weights.push_back(weights);
    if(DEBUG) cout << "読み込んだニューロン数 = " << neuron_weights.size() << endl;
    reserved_weights.push_back(neuron_weights);
    //
    cout << "weightファイルの読み込みを終了しました" << endl;
    return iteration;
}

void showVectorVals(string label, vector<double> v) {
    cout << label << " ";
    for (int i = 0; i < v.size(); ++i) {
        cout << i << ":" << v[i] << " ";
    }
    cout << endl;
}

int main(int argc, char** argv) {  
    string weightFile = "";
    string dataFileName = "trainingData.txt";
    int iteration = 0;
    int maxIteration = -1;
    bool testPhase = false;
    for(int i=1;i<argc;i++) {
        if (strcmp(argv[i], "-weights") == 0) {
            if(i+1 >= argc){
                cout << "コマンドの引数が不足しています [ " << argv[i] << "]" << endl;
                return 1;
            }
            string path = argv[++i];
            cout << "weightsファイルを読み込みます file = " << path << endl;
            iteration = openWeightFile(path);
            if (iteration == 0) {
                cout << "weightsファイルを読み込みに失敗しました" << endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-data") == 0) {
            if(i+1 >= argc){
                cout << "コマンドの引数が不足しています [ " << argv[i] << "]" << endl;
                return 1;
            }
            dataFileName = argv[++i];
            cout << "dataFileNameを変更します 新しいfile = " << dataFileName << endl;
        } else if (strcmp(argv[i], "-test") == 0) {
            cout << "バックプロパゲーションを行わずテストを実行します " << endl;
            testPhase = true;
        } else if (strcmp(argv[i], "-maxiter") == 0) {
            if(i+1 >= argc){
                cout << "コマンドの引数が不足しています [ " << argv[i] << "]" << endl;
                return 1;
            }
            maxIteration = atoi(argv[++i]);
            cout << "最大繰り返し数を [ " << maxIteration << " ] に変更します" << endl;
        } else if (strcmp(argv[i], "-help") == 0) {
            cout << "有効なオプション: -weights -data -test -maxiter -help" << endl;
            return 0;
        }
    }
    Data trainData(dataFileName);
    vector<int> topology;
    trainData.getTopology(topology);
    if(topology.size() == 0){
        cout << "トポロジーの読み込みに失敗しました 強制終了" << endl;
        return 1;
    }
    cout << "ネットワークトポロジー" << endl;
    for(int i=0;i<topology.size();i++){
        cout << "Layer [ " << i << " ] Neuron num = " << topology.at(i) << endl;
    }
    Net net(topology);

    vector<double> inputVals;
    vector<double> targetOutputVals;
    int errorNum = 0;
    int correctNum = 0;
    vector<int> correctHist(1000); // 正解のヒストグラム
    for(int i=0;i<correctHist.size();i++) correctHist.at(i) = 0;
    while(!trainData.isEof()) {
        if(iteration%100 == 0) cout << "--No." << iteration << "--" << endl;
        // 最大繰り返し数が指定されていたらそれで中止する
        if(maxIteration >= 0 && iteration >= maxIteration) break;
        int num = trainData.getNextInputs(inputVals);
        //cout << "データ [ " << num << " ] 個、読み込みました" << endl;
        if(num == topology.at(0)){
            // データが入力ニューロン数と一致している場合に限る
            if(DEBUG) if(iteration%100 == 0) showVectorVals("入力データ", inputVals);
            net.getForwardOutput(inputVals);
            if(iteration%100 == 0) net.printForwardOutput();
        }
        num = trainData.getTargetOutputs(targetOutputVals);
        if(num == topology.at(topology.size()-1)){
            // データが出力ニューロン数と一致している場合に限る
            if(iteration%100 == 0) showVectorVals("main: target ", targetOutputVals);
            if(testPhase){
                vector<double> output = net.getOutput();
                double recogval = -100.0;
                int recognum;
                for(int i=0;i<output.size();i++){
                    if(output.at(i) > recogval){
                        recognum = i;
                        recogval = output.at(i);
                    }
                }
                if(targetOutputVals.at(recognum) > 0.5){
                    // 1か0なので、0.5で分ければ良い
                    // 最大値が、答えに一致していれば正解とする
                    correctNum++;
                    // 0から1の間を1000刻んで、ヒストグラムを得る
                    for(int i=correctHist.size()-1;i>=0;i--){
                        if(recogval > (double)i*0.001){
                            int res = correctHist.at(i);
                            res++;
                            correctHist.at(i) = res;
                            break;
                        }
                    }
                }else{
                    errorNum++;
                }
            }else {
                // バックプロパゲーションの実行
                // targetOutputValsをセミバッチにすべきか
                vector<double> deltaE;
                double loss = 0.0;
                for (int i = 0; i < topology.at(topology.size() - 1); i++) {
                    Layer layer = net.getOutputLayer();
                    deltaE.push_back(layer.layerOutput.at(i) - targetOutputVals.at(i));
                    loss += (layer.layerOutput.at(i) - targetOutputVals.at(i))*(layer.layerOutput.at(i) - targetOutputVals.at(i));
                }
                loss *= 0.5;
                if (iteration % 100 == 0) cout << "誤差: " << loss << endl;
                if (DEBUG) showVectorVals("損失関数微分値 deltaE ", deltaE);
                net.execBackpropagation(deltaE);
            }
        }
        iteration++;
    }
    // トレーニングを実行した場合は、最終ウェイトをファイルに出力する
    if(testPhase){
        cout << "****** テストの結果 *********" << endl;
        double rate = (double)correctNum/((double)correctNum+(double)errorNum);
        cout << "正解数 = " << correctNum << " 不正解数 = " << errorNum << " 正解率 = " << rate << endl;
        cout << "ヒストグラム（正解の出力値）" << endl;
        for(int i=correctHist.size()-1;i>=0;i--){
            double fromRange = (double)(i+1)*0.001;
            double toRange = (double)i*0.001;
            cout << fromRange << " - " << toRange << " : " << correctHist.at(i) << endl;
        }
    }else{
        net.printAllWeights(--iteration, dataFileName);
    }
    //
    cout << endl << "Done" << endl;
    //
    return 0;
}

