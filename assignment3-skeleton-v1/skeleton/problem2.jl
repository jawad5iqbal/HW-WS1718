using Images
using PyPlot
using Grid

include("Common.jl")


#---------------------------------------------------------
# Loads keypoints from JLD container.
#
# INPUTS:
#   filename     JLD container filename
#
# OUTPUTS:
#   keypoints1   [n x 2] keypoint locations (of left image)
#   keypoints2   [n x 2] keypoint locations (of right image)
#
#---------------------------------------------------------
function loadkeypoints(path::String)
  keypoints1 = load(path,"keypoints1")
  keypoints2 = load(path,"keypoints2")

  @assert size(keypoints1,2) == 2
  @assert size(keypoints2,2) == 2
  return keypoints1::Array{Int64,2}, keypoints2::Array{Int64,2}
end


#---------------------------------------------------------
# Compute pairwise Euclidean square distance for all pairs.
#
# INPUTS:
#   features1     [128 x m] descriptors of first image
#   features2     [128 x n] descriptors of second image
#
# OUTPUTS:
#   D             [m x n] distance matrix
#
#---------------------------------------------------------
function euclideansquaredist(features1::Array{Float64,2},features2::Array{Float64,2})
  f1_length = size(features1,2)
  f2_length = size(features2,2)

  D = zeros(f1_length,f2_length)


  for i = 1:f1_length, j = 1:f2_length
    D[i, j] = sum((features1[:, i] - features2[:, j]) .^ 2)
  end
  @assert size(D) == (size(features1,2),size(features2,2))
  return D::Array{Float64,2}
end


#---------------------------------------------------------
# Find pairs of corresponding interest points given the
# distance matrix.
#
# INPUTS:
#   p1      [m x 2] keypoint coordinates in first image.
#   p2      [n x 2] keypoint coordinates in second image.
#   D       [m x n] distance matrix
#
# OUTPUTS:
#   pairs   [min(N,M) x 4] vector s.t. each row holds
#           the coordinates of an interest point in p1 and p2.
#
#---------------------------------------------------------
function findmatches(p1::Array{Int,2},p2::Array{Int,2},D::Array{Float64,2})
  p1_len = size(p1,1)
  p2_len = size(p2,1)

  smaller_set = (p1_len < p2_len) ? p1 : p2
  larger_set =  (p1_len > p2_len) ? p1 : p2

  pairs = zeros(Int, size(smaller_set,1), 4);

  for i=1:size(smaller_set,1)
    minDistance_index = findfirst(D[i,:],minimum(D[i,:]))
    pairs[i,:] = [smaller_set[i,1] smaller_set[i,2] larger_set[minDistance_index,1] larger_set[minDistance_index,2]]
  end


  @assert size(pairs) == (min(size(p1,1),size(p2,1)),4)
  return pairs::Array{Int,2}
end


#---------------------------------------------------------
# Show given matches on top of the images in a single figure.
# Concatenate the images into a single array.
#
# INPUTS:
#   im1     first grayscale image
#   im2     second grayscale image
#   pairs   [n x 4] vector of coordinates containing the
#           matching pairs.
#
#---------------------------------------------------------
function showmatches(im1::Array{Float64,2},im2::Array{Float64,2},pairs::Array{Int,2})
  im1_size = size(im1, 2)
  im2_size = size(im2, 2)

  # while showing the images, making sure the white spaces are not shown
  xy = figure()[:add_subplot](1,1,1)
  xy[:set_xlim]([0, im1_size + im2_size - 1])
  xy[:set_ylim]([size(im1, 1) - 1, 0])

  # concatenating the images into single array
  PyPlot.imshow([im1  im2], "gray");

  # plotting the pairs on both the images
  PyPlot.plot(pairs[:,1], pairs[:,2], "wo");
  PyPlot.plot(pairs[:,3] + im1_size, pairs[:,4], "wo");

  # drawing lines between the match points
  PyPlot.plot([pairs[:, 1], pairs[:, 3] + im1_size], [pairs[:, 2], pairs[:, 4]], "r-");

  return nothing::Void
end


#---------------------------------------------------------
# Computes the required number of iterations for RANSAC.
#
# INPUTS:
#   p    probability that any given correspondence is valid
#   k    number of samples drawn per iteration
#   z    total probability of success after all iterations
#
# OUTPUTS:
#   n   minimum number of required iterations
#
#---------------------------------------------------------
function computeransaciterations(p::Float64,k::Int,z::Float64)

 n = ceil(log(1-z) / log(1-p^k))
 n = convert(Int, n)
  return n::Int
end


#---------------------------------------------------------
# Randomly select k corresponding point pairs.
#
# INPUTS:
#   points1    given points in first image
#   points2    given points in second image
#   k          number of pairs to select
#
# OUTPUTS:
#   sample1    selected [kx2] pair in left image
#   sample2    selected [kx2] pair in right image
#
#---------------------------------------------------------
function picksamples(points1::Array{Int,2},points2::Array{Int,2},k::Int)
  rnd = rand(1:size(points1, 1), k)

  sample1 = points1[rnd, :]
  sample2 = points2[rnd, :]

  @assert size(sample1) == (k,2)
  @assert size(sample2) == (k,2)
  return sample1::Array{Int,2},sample2::Array{Int,2}
end


#---------------------------------------------------------
# Conditioning: Normalization of coordinates for numeric stability.
#
# INPUTS:
#   points    unnormalized coordinates
#
# OUTPUTS:
#   U         normalized (conditioned) coordinates
#   T         [3x3] transformation matrix that is used for
#                   conditioning
#
#---------------------------------------------------------
function condition(points::Array{Float64,2})
  P = points[1:2,:] ./ points[3:3,:]
  t = mean(P,2)
  s = 0.5 * maximum(abs(P))
  T = [[eye(2) -t]./s; 0 0 1];
  U = T*points

  @assert size(U) == size(points)
  @assert size(T) == (3,3)
  return U::Array{Float64,2},T::Array{Float64,2}
end


#---------------------------------------------------------
# Estimates the homography from the given correspondences.
#
# INPUTS:
#   points1    correspondences in left image
#   points2    correspondences in right image
#
# OUTPUTS:
#   H         [3x3] estimated homography
#
#---------------------------------------------------------
function computehomography(points1::Array{Int,2}, points2::Array{Int,2})
  U1, T1 = condition(Common.cart2hom(points1'))
  U2, T2 = condition(Common.cart2hom(points2'))

  A = zeros(size(U1, 2) * 2, 9)

  for i = 1 : size(U1, 2)
    x1 = U1[1, i]
    y1 = U1[2, i]
    x2 = U2[1, i]
    y2 = U2[2, i]
    A[((2 * i) - 1) : (2 * i), :] = [[0 0 0 x1 y1 1 (-x1 * y2) (-y1 * y2) -y2];
                                     [-x1 -y1 -1 0 0 0 (x1 * x2) (y1 * x2) x2]];
  end


  U, S, V = svd(A, thin = false)

  h = V[:, end]
  h = h ./ h[end]

  H_bar = reshape(h, 3, 3)'

  H = inv(T2) * H_bar * T1


  @assert size(H) == (3,3)
  return H::Array{Float64,2}
end


#---------------------------------------------------------
# Computes distances for keypoints after transformation
# with the given homography.
#
# INPUTS:
#   H          [3x3] homography
#   points1    correspondences in left image
#   points2    correspondences in right image
#
# OUTPUTS:
#   d2         distance measure using the given homography
#
#---------------------------------------------------------
function computehomographydistance(H::Array{Float64,2},points1::Array{Int,2},points2::Array{Int,2})
  hp1 = H * Common.cart2hom(points1')
  hp1_cart = hp1[1:end-1,:] ./ hp1[end,:]'

  hp2 = inv(H) * Common.cart2hom(points2')
  hp2_cart = hp2[1:end-1,:] ./ hp2[end,:]'


  d2 = (hp1_cart - points2').^2 + (points1' - hp2_cart ).^2
  d2 = sum(d2,1)




  @assert length(d2) == size(points1,1)
  return d2::Array{Float64,2}
end


#---------------------------------------------------------
# Compute the inliers for a given distances and threshold.
#
# INPUTS:
#   distance   homography distances
#   thresh     threshold to decide whether a distance is an inlier
#
# OUTPUTS:
#  n          number of inliers
#  indices    indices (in distance) of inliers
#
#---------------------------------------------------------
function findinliers(distance::Array{Float64,2},thresh::Float64)
  indices = find(distance.<thresh)
  n = length(indices)

  return n::Int,indices::Array{Int,1}
end


#---------------------------------------------------------
# RANSAC algorithm.
#
# INPUTS:
#   pairs     potential matches between interest points.
#   thresh    threshold to decide whether a homography distance is an inlier
#   n         maximum number of RANSAC iterations
#
# OUTPUTS:
#   bestinliers   [n x 1 ] indices of best inliers observed during RANSAC
#
#   bestpairs     [4x4] set of best pairs observed during RANSAC
#                 i.e. 4 x [x1 y1 x2 y2]
#
#   bestH         [3x3] best homography observed during RANSAC
#
#---------------------------------------------------------
function ransac(pairs::Array{Int,2},thresh::Float64,n::Int)
  bestH = 0;
  bestN = 0;
  bestinliers = 0;
  bestpairs = 0;


  for i = 1 : n
    sample1, sample2 = picksamples(pairs[:, 1:2], pairs[:, 3:4], 4)
    H = computehomography(sample1, sample2);
    d = computehomographydistance(H, pairs[:, 1:2], pairs[:, 3:4]);
    N, indices = findinliers(d, thresh);

    if N > bestN
      bestN = N;
      bestinliers = indices;
      bestpairs = [sample1 sample2];
      bestH = H;
    end
  end


  @assert size(bestinliers,2) == 1
  @assert size(bestpairs) == (4,4)
  @assert size(bestH) == (3,3)
  return bestinliers::Array{Int,1},bestpairs::Array{Int,2},bestH::Array{Float64,2}
end


#---------------------------------------------------------
# Recompute the homography based on all inliers
#
# INPUTS:
#   pairs     pairs of keypoints
#   inliers   inlier indices.
#
# OUTPUTS:
#   H         refitted homography using the inliers
#
#---------------------------------------------------------
function refithomography(pairs::Array{Int64,2}, inliers::Array{Int64,1})
inliers1 = pairs[inliers ,1:2]
inliers2 = pairs[inliers ,3:4]
H = computehomography(inliers1,inliers2)



  @assert size(H) == (3,3)
  return H::Array{Float64,2}
end


#---------------------------------------------------------
# Show panorama stitch of both images using the given homography.
#
# INPUTS:
#   im1     first grayscale image
#   im2     second grayscale image
#   H       [3x3] estimated homography between im1 and im2
#
#---------------------------------------------------------
function showstitch(im1::Array{Float64,2},im2::Array{Float64,2},H::Array{Float64,2})
im2interp = InterpGrid(im2,0,InterpLinear)
height,width = size(im1)
warped = [im1[:,1:300] zeros(height ,400)]
for x = 301:700
  for y = 1:size(im1,1)
    ix,iy = Common.hom2cart(H*[x; y; 1])
    warped[y,x] = im2interp[iy,ix]
  end
end
figure()
imshow(warped,"gray",interpolation="none")
axis("off")
title("Stitched Images")


  return nothing::Void
end


#---------------------------------------------------------
# Problem 2: Image Stitching
#---------------------------------------------------------
function problem2()
  # SIFT Parameters
  sigma = 1.4             # standard deviation for presmoothing derivatives

  # RANSAC Parameters
  ransac_threshold = 50.0 # inlier threshold
  p = 0.5                 # probability that any given correspondence is valid
  k = 4                   # number of samples drawn per iteration
  z = 0.99                # total probability of success after all iterations

  # load images
  im1 = PyPlot.imread("a3p2a.png")
  im2 = PyPlot.imread("a3p2b.png")

  # Convert to double precision
  im1 = Float64.(im1)
  im2 = Float64.(im2)

  # load keypoints
  keypoints1, keypoints2 = loadkeypoints("keypoints.jld")

  # extract SIFT features for the keypoints
  features1 = Common.sift(keypoints1,im1,sigma)
  features2 = Common.sift(keypoints2,im2,sigma)

  # compute chi-square distance  matrix
  D = euclideansquaredist(features1,features2)

  # find matching pairs
  pairs = findmatches(keypoints1,keypoints2,D)

  # show matches
  showmatches(im1,im2,pairs)
  title("Putative Matching Pairs")

  # compute number of iterations for the RANSAC algorithm
  niterations = computeransaciterations(p,k,z)

  # apply RANSAC
  bestinliers,bestpairs,bestH = ransac(pairs,ransac_threshold,niterations)

  # show best matches
  showmatches(im1,im2,bestpairs)
  title("Best 4 Matches")

  # show all inliers
  showmatches(im1,im2,pairs[bestinliers,:])
  title("All Inliers")

  # stitch images and show the result
  showstitch(im1,im2,bestH)

  # recompute homography with all inliers
  H = refithomography(pairs,bestinliers)
  showstitch(im1,im2,H)

  return nothing
end
