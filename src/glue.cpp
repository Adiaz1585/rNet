#include <Rcpp.h>
#include "Network.hpp"
using namespace Rcpp;

RCPP_MODULE(Network){
  class_<Network>("Network")
  .constructor<std::vector<int> >()
  .method("train", &Network::train)
  .method("setCategories", &Network::setCategories)
  .method("getCategories", &Network::getCategories)
  .method("predict", &Network::predict);
}