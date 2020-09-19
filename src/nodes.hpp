/*
 * Author/Maintainer: Austin Diaz
 * File:"nodes.h"
 * Abstract: Header file for the nodes class.
 */

#ifndef NODES_H
#define NODES_H
#include <iostream>
#include <ctime> 
#include <cstdlib>
#include <vector>
#include <cmath>
#include <fstream>
//#include <sstream>
using namespace std;

struct Connection{
  float weight;
  float delta;
};

class Nodes;
typedef vector<Nodes> layers;

class Nodes{
public:
  // This index is used so that we can refer to the correct connection for the node in the layer before.
  int getIndex(){return index;};
  
  // The error is used primarily for the output layers gradient calculation.
  float getError(){return error;};
  void setError(float error);
  
  // This the value the node holds after the activation function is applied.
  float getVal(){return value;};
  void setVal(float value);
  
  /*################### Activation function ########################### */
  // these are the activation functions respective to the layers.
  // sigmoid is used for the output layer
  // ReLu function is used for the hidden outputs
  float sigmoid(float sum);
  float ReLu(float sum);
  /*################################################################### */
  
  /*################### Activation functions Derivitives ###############*/
  //This function calculates the derivatie for the ReLu activation function:
  //her we allow the 0 to be defined as 0 rather than undefined.
  float ReLuDerivative(float val);
  /*################################################################### */	
  
  /*######################### Delta Calculations #######################*/
  //these functions calculate the deltas for the output and hidden layers.
  void outputDelta(layers &beforeLay);
  void hiddenDelta(layers &beforeLay);
  /*####################################################################*/
  
  /*#########################  ##############################*/
  // feed values to the next layer
  // forwardFeedL is to for the last layer in the network
  void forwardFeed(layers &beforeLay);
  void forwardFeedL(layers &beforeLay, float trueVal); //for trining
  void forwardFeedL(layers &beforeLay); //for prediction
  /*####################################################################*/
  
  /*######################### Connections Function ##############################*/
  //Each node will have a vector of struct connection type.
  // each element in the vect will befor a respective node in the following layer m-1.
  vector<Connection> getConnections();
  
  void setConnectionsWeight(float val, int indexx){this->connections[indexx].weight = val;};
  void setConnectionsDelt(float val, int indexx){this->connections[indexx].delta = val;}
  /*####################################################################*/
  
  //constructors
  Nodes(int numOfconnections, int indexs); //done // construct the thing
  Nodes();
  
private:
  float outputSum; // this is handy for calculating derivatives of act func.
  float error; 
  int index; // this is used when looking for weight and delta in conection vect.
  float value; // output value
  vector<Connection> connections; // contains weigths and deltas
};

#endif