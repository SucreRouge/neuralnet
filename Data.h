/* 
 * File:   Data.h
 * Author: Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 *
 * Created on 2017/05/29, 6:34
 */

#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include <string>
#include <fstream>
#include <vector>

using namespace std;

class Data {
public:
    Data(const string filename);
    ~Data();

    bool isEof(void) {
        return m_trainingDataFile.eof();
    }
    
    void getTopology(vector<int> &topology);

    // Returns the number of input values read from the file:
    int getNextInputs(vector<double> &inputVals);
    int getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

#endif /* TRAININGDATA_H */

