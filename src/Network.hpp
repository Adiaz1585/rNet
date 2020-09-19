/*
 * Author/Maintainer: Austin Diaz
 * File:"nodes.h"
 * Abstract: Header file for the nodes class.
 */
#ifndef NETWORK_H
#define NETWORK_H
#include <Rcpp.h>

//#include <iostream>
//#include <ctime> 
//#include <cstdlib>
#include <vector>
#include <Rmath.h>
//#include <fstream>
//#include <sstream>
#include "nodes.hpp"
using namespace std;


void printProgBar(int percent, int epoch, int count, int inter);

class Network{
  
public:
  int count;
  void predict(Rcpp::NumericMatrix predictVals);
  void accuracy(Rcpp::NumericVector &inputs, Rcpp::NumericVector &trueVals, int inter, int epoch);
  void forwardFeed(Rcpp::NumericVector &inputs, Rcpp::NumericVector &trueVals); //for training
  void forwardFeed(Rcpp::NumericVector &predictVals); //for prediction
  void backProp(Rcpp::NumericVector &inputs ,Rcpp::NumericVector &trueVals, int inter, float alpha,int epoch);
  //void predict();
  void train(int nEpoch, float alpha, Rcpp::NumericMatrix trainData, Rcpp::NumericMatrix trueVals);
  void print();
  int size();
  void setCategories(Rcpp::StringVector categories);
  void setPredicted(Rcpp::NumericMatrix &TempMat, int index);
  Rcpp::StringVector getCategories();
  
  Network(vector<int> topology);
  //  ~Network();
  
private:
  vector<layers> net;
  Rcpp::NumericMatrix predicted; // all probabilities of predictions.
  Rcpp::StringVector categories;
  Rcpp::List predictions;
  
};

#endif