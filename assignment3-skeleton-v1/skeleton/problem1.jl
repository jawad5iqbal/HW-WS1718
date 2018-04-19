using Images
using PyPlot

include("Common.jl")

#---------------------------------------------------------
# Loads grayscale and color image given PNG filename.
#
# INPUTS:
#   filename     given PNG image file
#
# OUTPUTS:
#   gray         single precision grayscale image
#   rgb          single precision color image
#
#---------------------------------------------------------
function loadimage(filename)
  rgb = PyPlot.imread("a3p1.png")
  gray = Common.rgb2gray(rgb)
  return gray::Array{Float32,2}, rgb::Array{Float32,3}
end


#---------------------------------------------------------
# Computes structure tensor.
#
# INPUTS:
#   img             grayscale color image
#   sigma           std for presmoothing derivatives
#   sigma_tilde     std for presmoothing coefficients
#   fsize           filter size to use for presmoothing
#
# OUTPUTS:
#   S_xx       first diagonal coefficient of structure tensor
#   S_yy       second diagonal coefficient of structure tensor
#   S_xy       off diagonal coefficient of structure tensor
#
#---------------------------------------------------------
function computetensor(img::Array{Float64,2},sigma::Float64,sigma_tilde::Float64,fsize::Int)
    g = Common.gauss2d(sigma,[fsize,fsize])
    g_tilde = Common.gauss2d(sigma_tilde,[fsize,fsize])
    d = [0.5 0 -0.5] # derivative filter
    smoothed = imfilter(img,g) # replicate borders is default
    dx = imfilter(smoothed,d)
    dy = imfilter(smoothed,d')
    S_xx = imfilter(dx.^2, g_tilde)
    S_yy = imfilter(dy.^2, g_tilde)
    S_xy = imfilter(dx.*dy, g_tilde)

  return S_xx::Array{Float64,2},S_yy::Array{Float64,2},S_xy::Array{Float64,2}
end


#---------------------------------------------------------
# Computes Harris function values.
#
# INPUTS:
#   S_xx       first diagonal coefficient of structure tensor
#   S_yy       second diagonal coefficient of structure tensor
#   S_xy       off diagonal coefficient of structure tensor
#   sigma      std that was used for presmoothing derivatives
#   alpha      weighting factor for trace
#
# OUTPUTS:
#   harris     Harris function score
#
#---------------------------------------------------------
function computeharris(S_xx::Array{Float64,2},S_yy::Array{Float64,2},S_xy::Array{Float64,2}, sigma::Float64, alpha::Float64)
     d = S_xx.*S_yy - S_xy.^2
     t = S_xx+S_yy
     harris = sigma^4 * (d - alpha*t.^2)
  return harris::Array{Float64,2}
end


#---------------------------------------------------------
# Non-maximum suppression of Harris function values.
#   Extracts local maxima within a 5x5 stencils.
#   Allows multiple points with equal values within the same window.
#   Applies thresholding with the given threshold.
#
# INPUTS:
#   harris     Harris function score
#   thresh     param for thresholding Harris function
#
# OUTPUTS:
#   px        x-position of kept Harris interest points
#   py        y-position of kept Harris interest points
#
#---------------------------------------------------------
function nonmaxsupp(harris::Array{Float64,2}, thresh::Float64)

  har = Common.nlfilter(harris, mean ,5,5, "replicate")
  harris = padarray(har[3:end-2,3:end-2],[2,2],[2,2],"value",Inf)
  q = (harris.> thresh)
  #q = newharris .> thresh
  py, px = findn(q)
  return px::Array{Int,1},py::Array{Int,1}
end

#---------------------------------------------------------
# Problem 1: Harris Detector
#---------------------------------------------------------
function problem1()
  # parameters
  sigma = 2.4               # std for presmoothing derivatives
  sigma_tilde = 1.6*sigma   # std for presmoothing coefficients
  fsize = 25                # filter size for presmoothing
  alpha = 0.06              # Harris alpha
  threshold = 1e-7          # Harris function threshold

  # Load both colored and grayscale image from PNG file
  gray,rgb = loadimage("a3p1.png")
  # Convert to double precision
  gray = Float64.(gray)
  rgb = Float64.(rgb)

  # Compute the three coefficients of the structure tensor
  S_xx,S_yy,S_xy = computetensor(gray,sigma,sigma_tilde,fsize)

  # Compute Harris function value
  harris = computeharris(S_xx,S_yy,S_xy,sigma,alpha)

  # Display Harris images
  figure()
  imshow(harris,"jet",interpolation="none")
  axis("off")
  title("Harris function values")

  # Threshold Harris function values
  mask = harris .> threshold
  y,x = findn(mask)
  figure()
  imshow(rgb)
  plot(x,y,"xy")
  axis("off")
  title("Harris interest points without non-maximum suppression")
  gcf()

  # Apply non-maximum suppression
  x,y = nonmaxsupp(harris,threshold)

  # Display interest points on top of color image
  figure()
  imshow(rgb)
  plot(x,y,"xy",linewidth=8)
  axis("off")
  title("Harris interest points after non-maximum suppression")
  return nothing
end
