using Images  # Basic image processing functions
using PyPlot
using JLD


# Load the Bayer image from the provided .jld file
function loadbayer()

  return data::Array{Float64,2}
end


# Separate the Bayer image into three images (one for each color channel), filling up all
# unknown values with 0
function separatebayer(data::Array{Float64,2})

  return r::Array{Float64,2}, g::Array{Float64,2}, b::Array{Float64,2}
end


# Combine three color channels into a single image
function makeimage(r::Array{Float64,2}, g::Array{Float64,2}, b::Array{Float64,2})

  return image::Array{Float64,3}
end


# Interpolate missing color values using bilinear interpolation
function debayer(r::Array{Float64,2}, g::Array{Float64,2}, b::Array{Float64,2})

  return image::Array{Float64,3}
end


# display two images in a single figure window
function displayimages(img1::Array{Float64,3}, img2::Array{Float64,3})

  # you may reuse your function from problem1 here
  return nothing
end


#= Problem 2
Bayer Interpolation =#

function problem2()
  # load raw data
  data = loadbayer()

  # separate data
  r,g,b = separatebayer(data)

  # merge raw Bayer
  img1 = makeimage(r,g,b)

  # interpolate Bayer
  img2 = debayer(r,g,b)

  # display images
  displayimages(img1, img2)
  return
end
