##Julia Version 1.6
##Author: Victor H. Aguiar
##Date: 2023-04-11
##Description: This script is to train a simple GAN to generate data from a normal distribution.

using Pkg

# Create and activate a new environment
Pkg.activate("GAN_env")
# Pkg.add("Flux")
# Pkg.add("Distributions")
# Pkg.add("Plots")
#Pkg.add("Zygote")
#Pkg.add("Plots")


using Random
using Statistics
using Flux
using Flux: params
using Flux.Optimise: update!
using Zygote: gradient
using Plots
## Set a fixed random seed for reproducibility
using Random
Random.seed!(8942)


# Hyperparameters
epochs = 5000
batch_size = 32
latent_size = 5
lr = 0.001

# Data generating process (DGP) to create real samples
function real_data(batch_size)
    return randn(Float32, 1, batch_size) * 5 .+ 10
end

# Function to create random noise (latent vector)
function sample_noise(batch_size, latent_size)
    return randn(Float32, batch_size, latent_size)
end

# Define the Generator and Discriminator networks using Flux
# The Generator takes a noise vector of size `latent_size` and produces a single scalar output.
Generator() = Chain(Dense(latent_size, 64,relu),BatchNorm(64,relu), Dense(64,64,relu), Dense(64, 1))
## batchnorm is a layer that normalizes its inputs, then applies a learned affine transformation with learnable parameters. 
## In GANs it is usually a good practice to include such a layer.
# The Discriminator takes a scalar value from the real data samples or a scalar value generated by the Generator as input.
Discriminator() = Chain(Dense(1, 64, relu), Dense(64, 1, σ))
#The input size is set to 1 because the Generator's output and the real data samples are single-dimensional (scalar) values. 
#The Discriminator takes either a scalar value from the real data samples or a scalar value generated by the Generator as input. 
#Therefore, the input size of the first layer in the Discriminator is 1.
G = Generator()
D = Discriminator()

# Optimizers
opt_G = ADAM(lr)
opt_D = ADAM(lr)

# Training loop
for epoch in 1:epochs
    # Sample real and fake data
    real_samples = real_data(batch_size)
    noise = sample_noise(batch_size, latent_size)
    fake_samples = G(noise')

    # Train the Discriminator
    d_loss() = -mean(log.(D(real_samples))) - mean(log.(1 .- D(fake_samples)))
    grads_D = gradient(() -> d_loss(), params(D))
    update!(opt_D, params(D), grads_D)

    # Train the Generator
    noise = sample_noise(batch_size, latent_size)
    g_loss() = -mean(log.(D(G(noise'))))
    grads_G = gradient(() -> g_loss(), params(G))
    update!(opt_G, params(G), grads_G)

    # Print losses
    if epoch % 500 == 0
        println("Epoch: $epoch | Discriminator Loss: $(d_loss()) | Generator Loss: $(g_loss())")
    end
end

# Generate a sample of 100 data points using the trained Generator
@time noise = sample_noise(10000, latent_size)
@time generated_samples = G(noise')
@time validation_samples=real_data(10000)
# Display the generated data
println("Generated data sample: \n", generated_samples)
mean(generated_samples)
var(generated_samples)
mean(validation_samples)
var(validation_samples)

histogram(vec(generated_samples), bins=30, xlabel="Value", ylabel="Frequency", label="Generated samples", title="Generated Samples Distribution", legend=:topright)

histogram(vec(validation_samples), bins=30, xlabel="Value", ylabel="Frequency", label="Generated samples", title="Validation Samples Distribution", legend=:topright)
