##Julia Version 1.6
##Author: Victor H. Aguiar
##Date: 2021-06-01
##Description: This script is used to run the ABKK experiment.
##It is a pilot study to test if we can predict perfectly out-of-sample choices on the ABBK experiment. 
tempdir1=@__DIR__
rootdir=tempdir1[1:findfirst("USFQTopicsDecisionMaking",tempdir1)[end]]
cd(rootdir)
cd(rootdir*"/NN")
using Pkg
Pkg.activate()
using Distributed
using Statistics
using DataFrames, CSV
using Plots
addprocs(7)

@everywhere begin
  using Random
  using Combinatorics
  using LinearAlgebra
  using JuMP
end

##Machine Learning
using Flux
using Flux: params 

## Set a fixed random seed for reproducibility
using Random
Random.seed!(8942)

X1=CSV.read(rootdir*"/NN/data/ABKK_nnvictor.csv", DataFrame)


##Flux is the Julia package for machine learning. It is a pure-Julia implementation of the popular Python package PyTorch.


using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using Statistics: mean
using Parameters: @with_kw

@with_kw mutable struct Args
    lr::Float64 = 0.5
    repeat::Int = 110
end

function get_processed_data(args)
    labels = string.(X1.choice)
    features = Matrix(X1[:,2:end])'

    # Subract mean, divide by std dev for normed mean of 0 and std dev of 1.
    normed_features = normalise(features, dims=2)

    klasses = sort(unique(labels))
    onehot_labels = onehotbatch(labels, klasses)

    # Split into training and test sets, 2/3 for training, 1/3 for test.
    train_indices = [1:3:12297 ; 2:3:12297]

    X_train = normed_features[:, train_indices]
    y_train = onehot_labels[:, train_indices]

    X_test = normed_features[:, 3:3:12297]
    y_test = onehot_labels[:, 3:3:12297]

    #repeat the data `args.repeat` times
    train_data = Iterators.repeated((X_train, y_train), args.repeat)
    test_data = (X_test,y_test)

    return train_data, test_data
end

# Accuracy Function
accuracy(x, y, model) = mean(onecold(model(x)) .== onecold(y))

# Function to build confusion matrix
function confusion_matrix(X, y, model)
    ŷ = onehotbatch(onecold(model(X)), 1:6)
    y * transpose(ŷ)
end

function train(; kws...)
    # Initialize hyperparameter arguments
    args = Args(; kws...)

    #Loading processed data
    train_data, test_data = get_processed_data(args)

    # Declare model taking 37 features as inputs and outputting 6 probabilities,
    # one for each lottery.
    ##Create a traditional Dense layer with parameters W and b.
    ##y = σ.(W * x .+ b), x is of length 37 and y is of length 6.
    model = Chain(Dense(37, 6))

    # Defining loss function to be used in training
    # For numerical stability, we use here logitcrossentropy
    loss(x, y) = logitcrossentropy(model(x), y)

    # Training
    # Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)

    println("Starting training.")
    Flux.train!(loss, params(model), train_data, optimiser)

    return model, test_data
end

function test(model, test)
    # Testing model performance on test data
    X_test, y_test = test
    accuracy_score = accuracy(X_test, y_test, model)

    println("\nAccuracy: $accuracy_score")

    # Sanity check.
    @assert accuracy_score > 0.8

    # To avoid confusion, here is the definition of a Confusion Matrix: https://en.wikipedia.org/wiki/Confusion_matrix
    println("\nConfusion Matrix:\n")
    display(confusion_matrix(X_test, y_test, model))
    ##Loss function
    println("Loss test data")
    loss(x, y) = logitcrossentropy(model(x), y)
    display(loss(X_test,y_test))
end

cd(@__DIR__)
model, test_data = train()
model, test_data = train(lr=0.1)
test(model, test_data)

normopt=true

labels = string.(X1.choice)
features = Matrix(X1[:,2:end])'
normopt ? normed_features = normalise(features, dims=2) : normed_features=features

klasses = sort(unique(labels))
onehot_labels = onehotbatch(labels, klasses)

# Split into training and test sets, 2/3 for training, 1/3 for test.
train_indices = [1:3:12297 ; 2:3:12297]

X_train = normed_features[:, train_indices]
y_train = onehot_labels[:, train_indices]

X_test = normed_features[:, 3:3:12297]
y_test = onehot_labels[:, 3:3:12297]

#repeat the data `args.repeat` times
train_data = Iterators.repeated((X_train, y_train), 1000)
test_data = (X_test,y_test)



##model = softmax(Dense(37, 6),dims=6)

model2 = Chain(
  Dense(37, 37, relu),
  Dense(37, 6),
  softmax)

loss(x, y) = Flux.mse(model2(x), y)
optimiser = Descent(0.5)

#   train_data =  Iterators.repeated((features,labels), 100)
#   test_data = (features,y_test)
Flux.train!(loss, params(model2), train_data, optimiser)
loss(X_test,y_test)
accuracy(X_test,y_test,model2)
######################
#####################
  ## Third model
model3 = Chain(
  Dense(37,37,relu),
  Dense(37, 37, relu),
  Dense(37, 37, relu),
  Dense(37, 6),
  softmax)

loss(x, y) = Flux.mse(model3(x), y)
#optimiser = Descent(0.5)
optimiser = ADAM(0.001, (0.9, 0.8))

#   train_data =  Iterators.repeated((features,labels), 100)
#   test_data = (features,y_test)
Flux.train!(loss, params(model3), train_data, optimiser)

loss(X_train,y_train)
loss(X_test,y_test)
accuracy(X_test,y_test,model3)
model3(X_test)
X_testc=features[:, 3:3:12297]
X_testc[37,:]=zeros(size(X_testc)[2])
X_testc[1,:].=minimum(X_test[1,:])
X_testc=normalise(X_testc,dims=2)
model3(X_testc)
loss(X_testc,y_test)
accuracy(X_testc,y_test,model3)
#   A confusion matrix is a table that is used to describe the performance of a classification model on a set of data for which the true values are known. Each row of the matrix represents the instances in a predicted class, while each column represents the instances in an actual class (or vice versa). The name stems from the fact that it makes it easy to see if the model is confusing two classes (i.e., mislabeling one as another).
#   In your case, the confusion matrix is a 6x6 matrix, which means that you have a classification problem with 6 classes. The diagonal elements of the matrix represent the number of correct predictions for each class (i.e., true positives), and the off-diagonal elements represent the incorrect predictions (i.e., false positives and false negatives).
confusion_matrix(X_testc,y_test,model3)
######################
#####################
## Fourth model
model4 = Chain(
  Dense(37,37,relu),
  Dense(37, 37, relu),
  Dense(37, 37, relu),
  Dense(37, 6),
  softmax,
  Dense(6,6))

loss(x, y) = Flux.mse(model4(x), y)
#optimiser = Descent(0.5)
optimiser = ADAM(0.001, (0.9, 0.8))
#   train_data =  Iterators.repeated((features,labels), 100)
#   test_data = (features,y_test)
Flux.train!(loss, params(model4), train_data, optimiser)

loss(X_train,y_train)
loss(X_test,y_test)
accuracy(X_testc,y_test,model4)
confusion_matrix(X_testc,y_test,model4)
