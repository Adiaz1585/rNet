/*
 * Author/Maintainer: Austin Diaz
 * File: Nodes implementation file "nodes.cpp"
 * Abstract: Implements the node class methods. The nodes class is for the individual nodes in the network.
 * Requirements: Uses ++11 standards. tested by compiling using: g++ g++ -std=c++11 -
 */


#include "nodes.hpp"
//#include <iostream>
//#include <ctime> 
//#include <cstdlib>
#include <vector>
#include <Rmath.h>
#include <Rcpp.h>
//#include <fstream>
//#include <sstream>

using namespace std;

void Nodes::setError(float error){
  this->error = error;
}

void Nodes::setVal(float value){
  this->value = value;
}
//This sets the delta for the output connections with resepect to the output nodes.
void Nodes::outputDelta(layers &beforeLay){
  for(int node = 0; node < beforeLay.size(); node++){
    beforeLay[node].setConnectionsDelt(error * sigmoid(outputSum) * (1 - sigmoid(outputSum)), index);
  }
}

//This sets the delta for the hidden connections with resepect to the output nodes.
void Nodes::hiddenDelta(layers &beforeLay){
  float newDelt = 0.0;
  for(int i = 0; i < getConnections().size(); i++){
    newDelt += getConnections()[i].weight *getConnections()[i].delta;
  }
  //	cout << "OutputSum: " << outputSum << '\n';
  newDelt *= ReLuDerivative(outputSum);
  
  for(int node = 0; node < beforeLay.size(); node++){
    beforeLay[node].setConnectionsDelt(newDelt, index);
  }
}

//used for hidden layers
float Nodes::ReLuDerivative(float val){
  if(val > 0.0){
    return 1.0;
  }
  else return 0.0;
}


float Nodes::sigmoid(float sum){
  return 1/(1+exp(-sum));
}

float Nodes::ReLu(float sum){
  if(sum > 0.0){
    return sum;
  }
  else return 0.0;
}

void Nodes::forwardFeedL(layers &beforeLay, float trueVal){
  float sum = 0.0;
  this->outputSum = 0.0;
  for(int i = 0; i < beforeLay.size(); i++){
    sum += beforeLay[i].getVal() * beforeLay[i].getConnections()[index].weight;
  }
  this->outputSum = sum;
  //setVal(Nodes::sigmoid(sum));
  value = Nodes::sigmoid(sum);
  this->error = value - trueVal;
}

void Nodes::forwardFeedL(layers &beforeLay){
  float sum = 0.0;
  this->outputSum = 0.0;
  for(int i = 0; i < beforeLay.size(); i++){
    sum += beforeLay[i].getVal() * beforeLay[i].getConnections()[index].weight;
  }
  this->value = Nodes::sigmoid(sum);
}

void Nodes::forwardFeed(layers &beforeLay){
  float sum = 0.0;
  this->outputSum = 0.0;
  for(int i = 0; i < beforeLay.size(); i++){
    //cout << index << " " <<beforeLay[i].getConnections()[index].weight<< endl;
    sum += beforeLay[i].getVal() * beforeLay[i].getConnections()[index].weight;
  }
  this->outputSum = sum;
  setVal(Nodes::ReLu(sum));
}

Nodes::Nodes(int numOfconnections, int index){
  
  GetRNGstate();
  Rcpp::NumericVector ran(1);
  for(int k = 0; k <numOfconnections; k++){
    connections.push_back(Connection());
    ran[0] = R::runif(-1,1);
    
    connections.back().weight = ran[0];
  }
  
  this->index = index;
  PutRNGstate();
}

vector<Connection> Nodes::getConnections(){
  return connections;
}