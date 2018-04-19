using Images
using PyPlot


# Create 3x3 derivative filters in x and y direction
function createfilters()
  fx = Array{Float64,2}(3,3)
  fy = Array{Float64,2}(3,3)
  dx = [0.5 0 -0.5]
  dy = [0.5 0 -0.5]'
  gy = Kernel.gaussian((1,1),(1,3))
  for  i = 1:3
        for  j = 1:3
            fx[i,j] = gy[i] * dx[j]
            fy[i,j] = dy[i] * gy'[j]
        end
    end
  return fx::Array{Float64,2},fy::Array{Float64,2}
end




# Apply derivate filters to an image and return the derivative images
function filterimage(I::Array{Float32,2},fx::Array{Float64,2},fy::Array{Float64,2})
  Ix = imfilter(I,fx)
  Iy = imfilter(I,fy)
  return Ix::Array{Float64,2},Iy::Array{Float64,2}
end


# Apply thresholding on the gradient magnitudes to detect edges
function detectedges(Ix::Array{Float64,2},Iy::Array{Float64,2}, thr::Float64)
  edges = sqrt(Ix.^2 + Iy.^2)
  edges[(edges .< thr)] = 0
  return edges::Array{Float64,2}
end


# Apply non-maximum-suppression
function nonmaxsupp(edges::Array{Float64,2},Ix::Array{Float64,2},Iy::Array{Float64,2})
    ph = phase(Ix,Iy)
    nonmax = thin_edges(edges,ph)
    edges = nonmax
  return edges::Array{Float64,2}
end


#= Problem 4
Image Filtering and Edge Detection =#

function problem4()
  # load image
  img = PyPlot.imread("a1p4.png")

  # create filters
  fx, fy = createfilters()

  # filter image
  imgx, imgy = filterimage(img, fx, fy)

  # show filter results
  figure()
  subplot(121)
  imshow(imgx, "gray", interpolation="none")
  title("x derivative")
  axis("off")
  subplot(122)
  imshow(imgy, "gray", interpolation="none")
  title("y derivative")
  axis("off")
  gcf()

  # show gradient magnitude
  figure()
  imshow(sqrt(imgx.^2 + imgy.^2),"gray", interpolation="none")
  axis("off")
  title("Derivative magnitude")
  gcf()

  # threshold derivative
  threshold = 0.1
  edges = detectedges(imgx,imgy,threshold)
  figure()
  imshow(edges.>0, "gray", interpolation="none")
  axis("off")
  title("Binary edges")
  gcf()

  # non maximum suppression
  edges2 = nonmaxsupp(edges,imgx,imgy)
  figure()
  imshow(edges2,"gray", interpolation="none")
  axis("off")
  title("Non-maximum suppression")
  gcf()
  return
end
#Threshold
#Threshold which was given to us 42 is too big. We cannot see anything at this
#value. So, we set it at 0.1. If threshold is too small, we detect more edges.
#And if it is too large, edges gets ignored. There should be a middle value
#decided for the particular picture.
