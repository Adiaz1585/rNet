#include <Rcpp.h>
#include "Network.hpp"
//#include <iostream>
//#include <ctime> 
//#include <cstdlib>
#include <vector>
#include <Rmath.h>
#include <unistd.h>
#include <Rinterface.h>
//#include <fstream>
//#include <sstream>
using namespace std;

Rcpp::StringVector Network::getCategories(){
  return this->categories;
}

void Network::setCategories(Rcpp::StringVector categories){
  this->categories = categories;
}


void Network::setPredicted(Rcpp::NumericMatrix &TempMat, int index){
  //Rcpp::Rcout<< predicted.nrow() << "\n";
  Rcpp::NumericVector Temp;
  
  
  for(int i = 0; i < net.back().size()-1; i++){
      Temp.push_back(net.back()[i].getVal());
  }
  
  TempMat.row(index) = Temp;
}

void Network::predict(Rcpp::NumericMatrix predictVals){
//What do we need in order to predict
  Rcpp::NumericMatrix TempMat(predictVals.nrow(), categories.length());
  
  for(int nrows = 0; nrows < predictVals.nrow(); nrows++){
    Rcpp::NumericVector predictValsVect = predictVals(nrows, Rcpp::_ );
    forwardFeed(predictValsVect);
    setPredicted(TempMat,nrows);
  }
  
  predicted = TempMat;
  float max = 0;
  int index;

  for(int j = 0; j< predicted.nrow(); j++){
    max = 0;
    for(int k = 0; k < predicted.ncol(); k++){
      
      if(max < predicted(j,k)){
        max = predicted(j,k);
        index = k;
        //Rcpp::Rcout << predicted.ncol() << endl;
        //Rcpp::Rcout << index << endl;
      }
      //Rcpp::Rcout << predicted(j,k) << ' ';
    } 
    
    Rcpp::Rcout << categories[index] << endl;
  }
  
// the inputs we want to predict on.
// forward feed through all layers.
// a way to interpret the prediction
// a print screen.
// For R a way return thes prdictions.
// make predects capable of do many at a time.
}
void Network::forwardFeed(Rcpp::NumericVector &predictVals){
  for(int i = 0; i < net[0].size()-1;i++){
    net[0][i].setVal(predictVals[i]);
  }
  
  int numlay = (int)net.size();
  
  for(int lay = 1; lay < numlay; lay++){
    layers beforeLay = net[lay-1];
    for(int i = 0; i < net[lay].size()-1; i++){
      if(lay == numlay-1){
        net[lay][i].forwardFeedL(beforeLay);
      }
      else{
        net[lay][i].forwardFeed(beforeLay);
      }
    } 
  }
}

void Network::backProp(Rcpp::NumericVector &inputs ,Rcpp::NumericVector &trueVals, int inter, float alpha, int epoch){
  //sets input to check error.
  R_CheckUserInterrupt();
  forwardFeed(inputs, trueVals);
  accuracy(inputs, trueVals, inter, epoch);
  float delt, weight;
  
  //this is the the back prop for just the last layer.
  //First step is to calculate the errors for the last terms.
  for(int layer = (int)net.size()-1; layer >= 1; layer--){
    if(layer < net.size()-1){
      for(int i = 0; i < net[layer].size()-1; i++){
        net[layer][i].hiddenDelta(net[layer - 1]);
      }
    }
    else{
      for(int i = 0; i < net.back().size() - 1; i++){
        net.back()[i].outputDelta(net[layer - 1]);
      }
    }
  }
  
  for(int layer = 0; layer < net.size() - 1; layer++){
    
    for(int j = 0; j < net[layer].size(); j++){
      //cout<< "j: " << net[1][j].getConnections().size() << '\n';
      //average change
      
      for(int k = 0; k < net[layer][j].getConnections().size(); k++){
        delt =0;
        delt = net[layer][j].getConnections()[k].delta;
        weight = net[layer][j].getConnections()[k].weight;
        //cout << "delt: " << weight << '\n';
        net[layer][j].setConnectionsWeight(weight - (alpha * (net[layer][j].getVal() *  delt + .0 * weight)), k);
      }
    }
  }
}

void Network::forwardFeed(Rcpp::NumericVector &inputs, Rcpp::NumericVector &trueVals){
  //this will set the input layer
  for(int i = 0; i < net[0].size()-1;i++){
    net[0][i].setVal(inputs[i]);
  }
 
  int numlay = (int)net.size();
  
  for(int lay = 1; lay < numlay; lay++){
    layers beforeLay = net[lay-1];
    for(int i = 0; i < net[lay].size()-1; i++){
      if(lay == numlay-1){
        net[lay][i].forwardFeedL(beforeLay, trueVals[i]);
      }
      else{
        net[lay][i].forwardFeed(beforeLay);
      }
    } 
  }
}

void Network::accuracy(Rcpp::NumericVector &inputs, Rcpp::NumericVector &trueVals, int inter, int epoch){
  
  float predict = 0., acc;
  int index = 0;
  
  for(int i = 0; i < net.back().size()-1; i++){
    //out << net.back()[i].getVal() << endl;
    if(predict < net.back()[i].getVal()){
      predict = net.back()[i].getVal();
      index = i;
    }
  }
  
  if(trueVals[index] == 1){
    count++;
  }
  acc = count / ((float)inter +1);
  printProgBar(acc *100,epoch, count, inter+1);
  //print();
}

void Network::print(){
  
  /*for(int layer = 0; layer < net.size(); layer++){
  for(int node = 0; node < net[layer].size(); node++){
  cout << net[layer][node].getVal() << " \n";
  }
  cout << '\n';
}*/
  
  //for(int i = 0; i < net[1].size();i++){
  
  cout << net[2].back().getConnections()[1].weight << endl;
  //}
  
  //cout << '\n';
  }

void Network::train(int nEpoch, float alpha, Rcpp::NumericMatrix trainData, Rcpp::NumericMatrix trueVals){
  count = 0;
  
  for(int epoch = 0; epoch < nEpoch; epoch++){
    for(int i = 0; i < trainData.nrow(); i++){
      Rcpp::NumericVector trainDataVect = trainData(i, Rcpp::_);
      Rcpp::NumericVector trueValsVect = trueVals(i, Rcpp::_);
      backProp(trainDataVect, trueValsVect, i, alpha, epoch+1);
    }
    count = 0;
  }
  //Rcpp::Rcout<< trainData(0,0) << endl;
}

Network::Network(vector<int> topology){
  count =0;
  //This will set the topology.
  int numOfconnections;
  int numOfLayers = (int) topology.size();
  
  for(int i = 0; i < numOfLayers; i++){
    net.push_back(layers());
    
    if(i == numOfLayers - 1){
      numOfconnections =  0;
    }
    else{
      numOfconnections = topology[i+1];
    }
    
    for(int j = 0; j <=topology[i]; j++){
      net.back().push_back(Nodes(numOfconnections, j));
    }
    net.back().back().setVal(1.0);
  }
  for(int lay = 0; lay< net.size(); lay++){
    if(lay == 0){
      Rcpp::Rcout << "Input Layer Size: " << net[lay].size()-1 << endl;
    }
    else if(lay ==net.size()-1){
      Rcpp::Rcout << "Output Layer Size: " << net[lay].size()-1 << endl;
    }
    else{
      Rcpp::Rcout << "Hidden Layer " << lay << " Size: " << net[lay].size()-1 << endl;
    }
  }
}


void printProgBar( int percent, int epoch, int count, int inter ){
  string bar;
  
  for(int i = 0; i < 50; i++){
    if( i < (percent/2)){
      bar.replace(i,1,"=");
    }else if( i == (percent/2)){
      bar.replace(i,1,">");
    }else{
      bar.replace(i,1," ");
    }
  }
  
  Rcpp::Rcout<< "\r" "[" << bar << "] ";
  Rcpp::Rcout.width( 3 );
  Rcpp::Rcout<<  " | Accuracy: " << percent << "%   " << count << "/" << inter << "     |    Epoch:  " << epoch << "  ";
 
  #if !defined(WIN32) && !defined(__WIN32) && !defined(__WIN32__)
  R_FlushConsole();
  #endif
}