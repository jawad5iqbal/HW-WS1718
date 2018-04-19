using Images
using PyPlot
using Clustering
using MultivariateStats

include("Common.jl")

#---------------------------------------------------------
# Type aliases for arrays of images/features
#---------------------------------------------------------
typealias ImageList Array{Array{Float64,2},1}
typealias FeatureList Array{Array{Float64,2},1}


#---------------------------------------------------------
# Structure for storing datasets
#
# Fields:
#   images      images associated with this dataset
#   labels      corresponding labels
#   n           number of examples
#---------------------------------------------------------
type Dataset
  images::Array{Array{Float64,2},1}
  labels::Array{Float64,1}
  n::Int
end

#---------------------------------------------------------
# Provides Dataset.length() method.
#---------------------------------------------------------
import Base.length
function length(x::Dataset)
  @assert length(x.images) == length(x.labels) == x.n "The length of the dataset is inconsistent."
  return x.n
end


#---------------------------------------------------------
# Structure for storing SIFT parameters.
#
# Fields:
#   fsize         filter size
#   sigma         standard deviation for filtering
#   threshold     SIFT threshold
#   boundary      number of boundary pixels to ignore
#---------------------------------------------------------
type Parameters
  fsize::Int
  sigma::Float64
  threshold::Float64
  boundary::Int
end



#---------------------------------------------------------
# Create input data by separating planes and bikes randomly
# into two equally sized sets.
#
# Note: Use the Dataset type from above.
#
# OUTPUTS:
#   trainingset      Dataset of length 120, contraining bike and plane images
#   testingset       Dataset of length 120, contraining bike and plane images
#
#---------------------------------------------------------
function loadimages()
 nbikes     = 106
 nplanes    = 134
 imgplanes = ImageList(nplanes)
 imgbikes  = ImageList(nbikes)
 for i = 1:nbikes
   imgbikes[i] = PyPlot.imread(@sprintf("bikes/%03i.png",i))
 end
 for i = 1:nplanes
   imgplanes[i] = PyPlot.imread(@sprintf("planes/%03i.png",i))
 end
 bike_rand   = sample(1:nbikes,nbikes,replace = false)
 plane_rand  = sample(1:nplanes,nplanes,replace = false)
 trainbike   = Dataset(imgbikes[bike_rand[1:53]],ones(Float64,53),53)
 testbike    = Dataset(imgbikes[bike_rand[54:106]],ones(Float64,53),53)
 trainplane  = Dataset(imgplanes[plane_rand[1:67]],zeros(Float64,67),67)
 testplane   = Dataset(imgplanes[plane_rand[68:134]],zeros(Float64,67),67)
 trainingset = Dataset([trainbike.images;trainplane.images],[trainbike.labels;trainplane.labels],trainbike.n+trainplane.n)
 testingset  = Dataset([testbike.images;testplane.images],[testbike.labels;testplane.labels],testbike.n+testplane.n)
  @assert length(trainingset) == 120
  @assert length(testingset) == 120
  return trainingset::Dataset, testingset::Dataset
end


#---------------------------------------------------------
# Extract features for all images using im2feat for each
#---------------------------------------------------------
function extractfeatures(images::ImageList,params::Parameters)
  N = length(images)
   features = FeatureList(N)
   for i=1:N
     features[i]= im2feat(images[i],params.fsize,params.sigma,params.threshold,params.boundary)
   end
  @assert length(features) == length(images)
  return features::FeatureList
end


#---------------------------------------------------------
# Extract features for a single image by applying Harris detection
# to find interest points and SIFT to compute the features at these points.
#---------------------------------------------------------
function im2feat(im::Array{Float64,2},fsize::Int,sigma::Float64,threshold::Float64,boundary::Int)
  mask,h = Common.detect_interestpoints(im,sigma,fsize,threshold,boundary)
  mask   = padarray(mask[1+boundary:end-boundary,1+boundary:end-boundary],[boundary,boundary],[boundary,boundary],"value",false)
  y, x   = findn(mask)
  F      = Common.sift([x y],im,sigma)
  @assert size(F,1) == 128
  return F::Array{Float64,2}
end


#---------------------------------------------------------
# Build a concatenated feature matrix from all given features
#---------------------------------------------------------
function concatenatefeatures(features::FeatureList)
  X = cat(2,features...)
  @assert size(X,1) == 128
  return X::Array{Float64,2}
end

#---------------------------------------------------------
# Build a codebook for a given feature matrix by k-means clustering with K clusters
#---------------------------------------------------------
function computecodebook(X::Array{Float64,2},K::Array{Int64,1})
  cluster = kmeans(X,K)
  codebook = cluster.centers
  @assert size(codebook) == (size(X,1),K)
  return codebook::Array{Float64,2}
end


#---------------------------------------------------------
# Compute a histogram over the codebook for all given features
#---------------------------------------------------------
function computehistogram(features::FeatureList,codebook::Array{Float64,2},K::Array{Int64,1})

  @assert size(H) == (K,length(features))
  return H::Array{Float64,2}
end


#---------------------------------------------------------
# Visualize a feature matrix by projection to the first
# two principal components. Points get colored according to class labels y.
#---------------------------------------------------------
function visualizefeatures(X::Array{Float64,2}, y)

  return nothing
end


# Problem 2: Bag of Words Model: Codebook

function problem2()
  # parameters
  params = Parameters(15,1.4,1e-7,10)
  K = collect(1:50)

  # load trainging and testing data
  traininginputs,testinginputs = loadimages()

  # extract features from images
  trainingfeatures = extractfeatures(traininginputs.images,params)
  testingfeatures = extractfeatures(testinginputs.images,params)

  # construct feature matrix from the training features
  X = concatenatefeatures(trainingfeatures)

  # write codebook
  codebook = computecodebook(X,K)

  # compute histogram
  traininghistogram = computehistogram(trainingfeatures,codebook,K)
  testinghistogram = computehistogram(testingfeatures,codebook,K)

  # visualize training features
  visualizefeatures(traininghistogram, traininginputs.labels)

  return nothing
end
