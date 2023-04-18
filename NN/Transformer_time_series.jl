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
#Pkg.upgrade_manifest()

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
using Plots
#using Transformers.Basic

# Load the data
ta = readtimearray("rate.csv", format="mm/dd/yy", delim=',')

# Create the time series
function get_src_trg(
    sequence, 
    enc_seq_len, 
    dec_seq_len, 
    target_seq_len
)
	nseq = size(sequence)[2]
	
	@assert  nseq == enc_seq_len + target_seq_len
	src = sequence[:,1:enc_seq_len,:]
	trg = sequence[:,enc_seq_len:nseq-1,:]
	@assert size(trg)[2] == target_seq_len
	trg_y = sequence[:,nseq-target_seq_len+1:nseq,:]
	@assert size(trg_y)[2] == target_seq_len
	if size(trg_y)[1] == 1
	 	return src, trg, dropdims(trg_y; dims=1)
	else
		return src, trg, trg_y
	end
end

## Model parameters
dim_val = 512 # This can be any value. 512 is used in the original transformer paper.
n_heads = 8 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
n_decoder_layers = 4 # Number of times the decoder layer is stacked in the decoder
n_encoder_layers = 4 # Number of times the encoder layer is stacked in the encoder
input_size = 1 # The number of input variables. 1 if univariate forecasting.
dec_seq_len = 92 # length of input given to decoder. Can have any integer value.
enc_seq_len = 153 # length of input given to encoder. Can have any integer value.
output_sequence_length = 58 # Length of the target sequence, i.e. how many time steps should your forecast cover
in_features_encoder_linear_layer = 2048 # As seen in Figure 1, each encoder layer has a feed forward layer. This variable determines the number of neurons in the linear layer inside the encoder layers
in_features_decoder_linear_layer = 2048 # Same as above but for decoder
max_seq_len = enc_seq_len # What's the longest sequence the model will encounter? Used to make the positional encoder


#define 2 layer of transformer
encode_t1=Transformers.Basic.TransformerEncoderLayer(dim_val, n_heads,64, 2048;future=false,pdrop=0.2)
encode_t1 = Transformer(dim_val, n_heads, 64, 2048;future=false,pdrop=0.2)
encode_t2 = Transformer(dim_val, n_heads, 64, 2048;future=false,pdrop=0.2)
#define 2 layer of transformer decoder
decode_t1 = TransformerDecoder(dim_val, n_heads, 64, 2048,pdrop=0.2) 
decode_t2 = TransformerDecoder(dim_val, n_heads, 64, 2048,pdrop=0.2) 
encoder_input_layer = Dense(input_size,dim_val) 
decoder_input_layer = Dense(input_size,dim_val) 
positional_encoding_layer = PositionEmbedding(dim_val) 
p = 0.2
dropout_pos_enc = Dropout(p) 

#define the layer to get the final output probabilities
#linear = Positionwise(Dense(dim_val, output_sequence_length))
linear = Dense(output_sequence_length*dim_val,output_sequence_length) |> gpu
function encoder_forward(x)
  x = encoder_input_layer(x)
  e = positional_encoding_layer(x)
  t1 = x .+ e
  t1 = dropout_pos_enc(t1)
  t1 = encode_t1(t1)
  t1 = encode_t2(t1)
  return t1