#Network.R

#' rNet
#'
#' My cool package
#'
#' Imports
#' @useDynLib rNet, .registration = TRUE
#' @export Network
#' @export Model
#' @import Rcpp
"_PACKAGE"

Rcpp::loadModule(module = "Network", TRUE)

Model <- function(vect){
  new(Network, vect)
}