###Simple illustration of ELVIS
## vhaguiar@gmail.com
import JuMP
import Ipopt
import Distributions
using CSV, DataFrames
using Random
using Tables
using Distributions

## Generate a data Set
##parameters
T=2
K=2
N=100
##utilities
## u(x)=x1^α+x2^(1-α)
Random.seed!(6749)
α=rand(N)/2
## lambdas
λ=randexp(N).+1
## prices
p=randexp(N,K,T)
## true consumption
ct=zeros(N,K,T)
for i in 1:N
    for k in 1:K
            for t in 1:T
                if k==1
                    ct[i,k,t]=(λ[i]/α[i]*p[i,k,t])^(1/(-1+α[i]))
                end
                if k==2 
                    ct[i,k,t]=((1-α[i])/(λ[i]*p[i,k,t]))^(1/α[i])
                end
            end 
    end
end 

##observed consumption
a=.9
b=1.1
ϵ=(b-a).*rand(N,K,T).+a
co=ct.*ϵ

### ELVIS
##Fix a simulation  number
Ns=1000
## We observe co, and p
co
p
## Sample ct such that it satisfies the model 
##prior α, it has to have the correct support
αp=rand(Ns)
##prior lambda
λp=randexp(Ns)
##prices, we have to know who faces each prices
pp=zeros(Ns,K,T)

## prior consumption
## you have to know the model
ctp=zeros(N,Ns,K,T)
for is in 1:Ns 
    for i in 1:N
    for k in 1:K
            for t in 1:T
                if k==1
                    ctp[i,is,k,t]=(λp[is]/αp[is]*p[i,k,t])^(1/(-1+αp[is]))
                end
                if k==2 
                    ctp[i,is,k,t]=((1-αp[is])/(λp[is]*p[i,k,t]))^(1/αp[is])
                end
            end 
    end
    end
end 

ctp

## Moment conditions are known
## E[co-ctp]=0
## co=ctϵ, hence w=ct-co, w=(1-ϵ)ct
## E[w]=E[(1-ϵ)ct|ct]=0 iff E[ϵ|ct]=1, which it is. 
## E[w]=E[E[w|ct]]=0 law of iterated expectations. 

## moments
wsim=zeros(N,K,T)

function jump(c,p)
    αp=rand(N)
    ##prior lambda
    λp=randexp(N)
    for is in 1:Ns 
        for i in 1:N
        for k in 1:K
                for t in 1:T
                    if k==1
                        wsim[i,k,t]=co[i,k,t]-(λp[i]/αp[i]*p[i,k,t])^(1/(-1+αp[i]))
                    end
                    if k==2 
                        wsim[i,k,t]=co[i,k,t]-((1-αp[i])/(λp[i]*p[i,k,t]))^(1/αp[i])
                    end
                    wsim[i,k,t]=co[i,k,t]-ctp[i,is,k,t]
                end 
        end
        end
    end 
    wsim
end

function myfun(gamma=gamma,w=w)
    gvec=ones(N,T*K)
    @simd for j=1:T
        @simd  for k=1:K
             @inbounds gvec[:,1]=w[:,1,1]
             @inbounds gvec[:,2]=w[:,1,2]
             @inbounds gvec[:,3]=w[:,2,1]
             @inbounds gvec[:,4]=w[:,2,2]
        end
    end
    gvec
end


w=jump(co,p,αp,λp)

wc=zeros(N,K,T)

## repetitions n1=burn n2=sample accept
repn=[10,100]
chainM=zeros(N,K*T,repn[2])

function gchain(gamma,co,p,wc=wc,w=w,repn=repn,chainM=chainM)
    r=-repn[1]+1
    while r<=repn[2]
      wc[:,:,:]=jump(co,p);
      logtrydens=(-(sum(wc[:,1,1].^2,dims=1)+sum(wc[:,1,2].^2,dims=1)+sum(wc[:,2,1].^2,dims=1)+sum(wc[:,2,2].^2,dims=1))+ (sum(w[:,1,1].^2,dims=1)+sum(w[:,1,2].^2,dims=1)+sum(w[:,2,1].^2,dims=1)+sum(w[:,2,2].^2,dims=1)))[:,1,1]
      dum=log.(rand(N)).<logtrydens

      @inbounds w[dum,:,:]=wc[dum,:,:]
      if r>0
        print(myfun(gamma,w))
        chainM[:,:,r]=myfun(gamma,w)

      end
      r=r+1
    end
end

gamma=[1 2]
gchain(gamma,co,p,w,wc,repn,chainM)
chainM
