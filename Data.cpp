/* 
 * File:   Data.cpp
 * Author: Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 * 
 * Created on 2017/05/29, 6:34
 */

#include "Data.h"
#include <sstream>
#include <boost/algorithm/string.hpp>

using namespace std;

Data::Data(const string filename){
	m_trainingDataFile.open(filename.c_str());
}

void Data::getTopology(vector<int> &topology) {
    // ネットワークのトポロジーだけを得る
    string line;
    string label;
    getline(m_trainingDataFile, line);
    boost::trim (line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }
    // stringstreamにすると、空白を自動的に読み飛ばすので
    // splitなどの処理が入らなくなる
    // 事前に、他の区切り文字を空白に変えておくこともいい
    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }
    return;
}

int Data::getNextInputs(vector<double> &inputVals){
    inputVals.clear();
    string line;
    getline(m_trainingDataFile, line);
    boost::trim (line);
    stringstream ss(line);
    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }
    return inputVals.size();
}

int Data::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();
    string line;
    getline(m_trainingDataFile, line);
    boost::trim (line);
    stringstream ss(line);
    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }
    return targetOutputVals.size();
}

Data::~Data() {
}

