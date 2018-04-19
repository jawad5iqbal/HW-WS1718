using Images
using PyPlot
include("Common.jl")


#---------------------------------------------------------
# Smoothes a given image with a Gaussian filter.
#
# INPUTS:
#   img             grayscale color image
#   sigma           std for smoothing
#   fsize           filter size to use for smoothing
#
# OUTPUTS:
#   img_smoothed    smoothed image
#
#---------------------------------------------------------
function presmooth(img::Array{Float64,2}, sigma::Float64, fsize::Int)
  img_smoothed = imfilter(img, Common.gauss2d(sigma, [fsize, fsize]))
  return img_smoothed::Array{Float64,2}
end


#---------------------------------------------------------
# Computes first-order derivatives of a given image pair,
#   Applies central differences for spatial derivatives.
#   Applies forward differences for temporal derivative.
#
#
# INPUTS:
#   im1              first image
#   im2              second image
#
# OUTPUTS:
#   dx               derivative in x-dimension of first image
#   dy               derivative in y-dimension of first image
#   dt               temporal derivative
#
#---------------------------------------------------------
function compute_derivatives(im1::Array{Float64,2}, im2::Array{Float64,2})
  dx = Array{Float64, 2}(
  size(im1,1),size(im1,2))
  dy = Array{Float64, 2}(
  size(im1,1),size(im1,2))
  dt = Array{Float64, 2}(
  size(im1,1),size(im1,2))






  for i=2:size(im1,1)-1
    for j=2:size(im1,2)-1
      dy[i,j] = im1[i+1,j]-im1[i-1,j]
      dx[i,j] = im1[i,j+1]-im1[i,j-1]
   end
  end

  dx[1,:] = dx[2,:]
  dx[size(im1,1),:] = dx[size(im1,1)-1,:]
  dx[:,1] = dx[:,2]
  dx[:,size(im1,2)] = dx[:,size(im1,2)-1]

  dy[1,:] = dy[2,:]
  dy[size(im1,1),:] = dy[size(im1,1)-1,:]
  dy[:,1] = dy[:,2]
  dy[:,size(im1,2)] = dy[:,size(im1,2)-1]

  dt[:,:] = im2[:,:]-im1[:,:]
  return dx::Array{Float64,2}, dy::Array{Float64,2}, dt::Array{Float64,2}
end


#---------------------------------------------------------
# Computes coefficients for linear system of equations.
#   Coefficients are presmoothed.
#   Filtering is done with replicate boundaries.
#
# INPUTS:
#   dx              x-derivative of first image
#   dy              y-derivative of first image
#   dt              temporal derivative
#   sigma           std for coefficient smoothing
#   fsize           filter size to use for coefficient smoothing
#
# OUTPUTS:
#                Coefficients of system matrix A
#   dx2             A11 component
#   dy2             A22 component
#   dxdy            A12/A21  component
#                Coefficients of right hand side b
#   dxdt            b1 component
#   dydt            b2 component
#
#---------------------------------------------------------
function compute_coefficients(dx::Array{Float64,2}, dy::Array{Float64,2}, dt::Array{Float64,2}, sigma::Float64, fsize::Int)
  dx2 = dx.^2
  dy2 = dy.^2
  dxdy = dx.*dy
  dxdt = dx.*dt
  dydt = dy.*dt



  dx2 = imfilter(dx2 ,Common.gauss2d(2.0 ,[11,11]))
  dy2 = imfilter(dy2 ,Common.gauss2d(2.0 ,[11,11]))
  dxdy = imfilter(dxdy ,Common.gauss2d(2.0 ,[11,11]))
  dxdt = imfilter(dxdt ,Common.gauss2d(2.0 ,[11,11]))
  dydt = imfilter(dydt ,Common.gauss2d(2.0 ,[11,11]))

  return dx2::Array{Float64,2}, dy2::Array{Float64,2}, dxdy::Array{Float64,2}, dxdt::Array{Float64,2}, dydt::Array{Float64,2}
end

#---------------------------------------------------------
# Computes optical flow at given positions.
#
# INPUTS:
#   px              x-coordinates of interest points in [1, ..., n]
#   py              y-coordinates of interest points in [1, ..., m]
#   dx2             A11 component
#   dy2             A22 component
#   dxdy            A12/A21  component
#   dxdt            b1 component
#   dydt            b2 component
#
# OUTPUTS:
#   u               optical flow (horizontal)
#   v               optical flow (vertical)
#
#---------------------------------------------------------
function compute_flow(px::Array{Int64,1}, py::Array{Int64,1},
                      dx2::Array{Float64,2}, dy2::Array{Float64,2}, dxdy::Array{Float64,2},
                      dxdt::Array{Float64,2}, dydt::Array{Float64,2})
                      A = Array{Float64, 2}(2,2)
                      b = Array{Float64, 2}(2,1)
                      u = Array{Float64, 1}(size(px))
                      v = Array{Float64, 1}(size(px))



                        for i=1:size(px,1)
                          A[1,1] = dx2[py[i],px[i]]
                          A[2,2] = dy2[py[i],px[i]]
                          A[1,2] = dxdy[py[i],px[i]]
                          A[2,1] = dxdy[py[i],px[i]]
                          b[1,1] = dxdt[py[i],px[i]]
                          b[2,1] = dydt[py[i],px[i]]
                          uVector = -inv(A)*b
                          u[i] = uVector[1,1]
                          v[i] = uVector[2,1]
                        end
  return u::Array{Float64,1}, v::Array{Float64,1}
end


#---------------------------------------------------------
# Shows optical flow on top of a given image.
#
# INPUTS:
#   px              x-coordinates of interest points in [1, ..., n]
#   py              y-coordinates of interest points in [1, ..., m]
#   u               optical flow (horizontal)
#   v               optical flow (vertical)
#   im              first image
#
#---------------------------------------------------------
function show_flow(px, py, u, v, im)
  figure()
  PyPlot.imshow(im,"gray")
  PyPlot.quiver(px, py, u, v, color="yellow")
  return nothing
end



# Problem 1: Optical Flow

function problem1()

  # Parameters, smoothing
  smooth_sigma = 2.0  # std
  smooth_fsize = 25   # window size

  # Parameters, coefficients
  coeff_sigma = 2.0  # std
  coeff_fsize = 11   # window size

  # Parameters, Harris keypoint detection
  harris_sigma = 1.0  # std
  harris_fsize = 15   # window size
  harris_threshold = 1e-7

  # Ignore locations to close to image boundaries
  boundary = 11

  # Load the images:
  im1 = Float64.(imread("frame09.png"))
  im2 = Float64.(imread("frame10.png"))

  # Detect interest points using 'detect_interestpoints'
  px, py = Common.detect_interestpoints(im1,harris_sigma,harris_fsize,harris_threshold,boundary)

  # Presmooth images
  im1s = presmooth(im1, smooth_sigma, smooth_fsize)
  im2s = presmooth(im2, smooth_sigma, smooth_fsize)

  # First-order derivatives of smoothed images
  dx, dy, dt = compute_derivatives(im1s, im2s)

  # Coefficients of the linear system of equations
  dx2, dy2, dxdy, dxdt, dydt = compute_coefficients(dx, dy, dt, coeff_sigma, coeff_fsize)

  # Compute optical flow ONLY AT interest points
  u,v = compute_flow(px, py, dx2, dy2, dxdy, dxdt, dydt)

  # Show optical flow on top of first image
  show_flow(px,py,u,v,im1)

  return nothing
end
