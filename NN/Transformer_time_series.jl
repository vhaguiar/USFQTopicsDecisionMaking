##Julia Version 1.7.2
##Author: Victor H. Aguiar
##Date: 2023-04-18
##Description: This script is to train a simple tranformer to predict time series data.
using Pkg

# Create and activate a new environment
Pkg.activate("Transformer_env")
Pkg.add("Flux")
Pkg.add("Transformers")
Pkg.add("TimeSeries")
Pkg.add("Statistics")
Pkg.add("MarketData")
Pkg.add("TensorBoardLogger")
Pkg.add("Logging")
Pkg.add("BSON")
Pkg.add("LinearAlgebra")
Pkg.add("Plots")
Pkg.upgrade_manifest()

using BSON: @save
#using CUDA
using Flux
using Flux.Optimise: update!
using Flux: gradient
using Logging
using MarketData
using Statistics
using TensorBoardLogger
using TimeSeries
using Transformers
using Transformers.Basic
