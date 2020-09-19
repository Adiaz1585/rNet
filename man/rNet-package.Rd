\name{rNet-package}
\alias{rNet-package}
\alias{rNet}
\docType{package}
\title{
  A Neural Net R package for solving classification problems.
}
\description{
  This package creates Neural Nets with a custom topology for classification problems. Hidden layers use a ReLu activation function, and the output layer uses a sigmoid activation function. This network uses a classic backpropagation to find gradients and update weights. The loss function used is the mean squared error. All source code is written in C++ and wrapped using the Rcpp package.
}
\details{
  This section should provide a more detailed overview of how to use the
  package, including the most important functions.
}
\author{
Maintainer: Austin Diaz <austin.b.diaz@gmail.com>
}
\references{
  A model is initialize using the $Model function. Training is done by $trian.
}
\keyword{ package }
\examples{
  \dontrun{
    ##net <- Model(c(4,8,8,3))
    ##net$setCategories(categories)
    ##net$getCategories()
    ##net$train(epoch = 10, alpha = .01, trainData, trueValue)
    ##net$predict(testData)  
  }
}
