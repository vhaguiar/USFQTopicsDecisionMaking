## Author: Victor H. Aguiar

## questions: slack

## Import packages, make sure to install them.
## Julia version: v"1.5.2"
import JuMP
import Ipopt
import Distributions
using CSV, DataFrames
using Random

################################################################################
## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("Microeconomics1",tempdir1)[end]]
diroutput=repdir*"\\finalexam\\programming\\results"
dirdata=repdir*"/finalexam\\programming\\data"


## Definition of countries
I=30

β=zeros(I)

for i in 1:I
    β[i]=1
end

β

σ=1/2

t=zeros(I,I)

#symmetric cost
for i in 1:I
    for j in 1:I
        t[i,j]=1+(i+j)/(I+I)
    end
end

# nominal endowments
y=zeros(I)
for i in 1:I
    y[i]=i^2/I+100
end

# world income
yw=sum(y)

#shares
θ=zeros(I)
for i in 1:I
    θ[i]=y[i]/yw
end

θ

## Simulating demand and prices
gravity=JuMP.Model(Ipopt.Optimizer)
JuMP.@variable(gravity,P[1:I]>=0)

JuMP.@NLobjective(gravity,Min,sum((P[j]^(1-σ)-sum( P[i]^(σ-1)*θ[i]*t[i,j]^(1-σ) for i in 1:I))^2 for j in 1:I))
JuMP.optimize!(gravity)

Psol=JuMP.value.(P)

## Demand data
x=zeros(I,I)

for i in 1:I
    for j in 1:I
        x[i,j]=y[i]*y[j]/yw*(t[i,j]/(Psol[i]*Psol[j]))^(1-σ)
    end
end

x

## Estimation
# Data generation
Random.seed!(123)
lxhat=log.(x)+randn(size(x))/100


CSV.write(dirdata*"/lxhat.csv",DataFrame(lxhat))
CSV.write(dirdata*"/tij.csv",DataFrame(t))
CSV.write(dirdata*"/y.csv",DataFrame(y'))

#####################################################################
sigmaestim=JuMP.Model(Ipopt.Optimizer)

JuMP.@variable(sigmaestim,α>=0)
JuMP.@variable(sigmaestim,Pv[1:I])
z=zeros(I,I)

for i in 1:I
    for j in 1:I
        z[i,j]=lxhat[i,j]-log(y[i])-log(y[j])
    end
end



##initialize


#JuMP.@NLobjective(sigmaestim,Min,sum(sum((z[i,j]-α[1]-α[2]*(log(t[i,j]-log(Pv[i])-α[2]*log(sum( Pv[i]^(-α[2])*y[i]/α[1]*t[i,j]^α[2] for  i in 1:I)))))^2 for i in 1: I) for j in 1:I))
#z[i,j]-α[1]-α[2]*log(t[i,j])-Pv[i]-Pv[j]
JuMP.@objective(sigmaestim,Min,sum(sum((z[i,j]-α*log(t[i,j])-Pv[i]-Pv[j])^2 for i in 1: I) for j in 1:I))


JuMP.optimize!(sigmaestim)

JuMP.value.(α)
JuMP.value.(Pv)
