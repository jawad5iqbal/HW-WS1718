#Matrikel-Nr. 2009889,2840174
#Name: Luqman Saqib, Ahsan Saeed


using Images
using PyPlot

# Create 3x3 derivative filters in x and y direction

function createfilters()

    fx = Array{Float64,2}(3,3)  # x-derivative filter
    fy = Array{Float64,2}(3,3)  # y-derivative filter

    x_gaussian    = 0.04*[7 11 7]  # 3x1 Matrix, "row vector"
    y_derivative  =  0.5*[1 0 -1]' # 1x3 Matrix, "column vector"

    y_gaussian    = 0.04*[7 11 7]' # 3x1 Matrix, "column vector"
    x_derivative  =  0.5*[1 0 -1]  # 1x3 Matrix, "row vector"

    for  i = 1:3
        for  j = 1:3
            fx[i,j] = y_gaussian[i] * x_derivative[j] # calculates the x-derivative filter
            fy[i,j] = y_derivative[i] * x_gaussian[j] # calculates the y-derivative filter
        end
    end

  return fx::Array{Float64,2},fy::Array{Float64,2}
end

# Apply derivate filters to an image and return the derivative images
function filterimage(I::Array{Float32,2},fx::Array{Float64,2},fy::Array{Float64,2})

    Ix = imfilter(I,fx,"replicate") # filters the image
    Iy = imfilter(I,fy,"replicate") # filters the image

    return Ix::Array{Float64,2},Iy::Array{Float64,2}
end


# Apply thresholding on the gradient magnitudes to detect edges
function detectedges(Ix::Array{Float64,2}, Iy::Array{Float64,2}, thr::Float64)

    edges::Array{Float64,2}

    magnitudes = magnitude(Ix,Iy)

    for i = 1:320
        for j = 1:480
            if magnitudes[j,i] < thr # if the value is lower then threshold, then color it black
                magnitudes[j,i] = 0
            else
                magnitudes[j,i] = 1 # else color the edge white
            end
        end
    end

    edges = magnitudes

  return edges::Array{Float64,2}
end

# Apply non-maximum-suppression
function nonmaxsupp(edges::Array{Float64,2},Ix::Array{Float64,2},Iy::Array{Float64,2})


    magnitudes = magnitude(Ix,Iy) # calculate the magnitudes sqrt(Ix^2+Iy^2)
    p = phase(Ix,Iy) # gives a matrix of angles

    nonmax = thin_edges(edges,p, "replicate")

    edges = nonmax

  return edges::Array{Float64,2}
end


#= Problem 3
Image Filtering and Edge Detection =#

function problem3()
  # load image
  img = PyPlot.imread("a1p4.png")
  # create filters
  fx, fy = createfilters()

  # filter image
  Ix, Iy = filterimage(img, fx, fy)

  # show filter results
  figure()
  subplot(121)
  imshow(Ix, "gray", interpolation="none")
  title("x derivative")
  axis("off")
  subplot(122)
  imshow(Iy, "gray", interpolation="none")
  title("y derivative")
  axis("off")
  gcf()

  # show gradient magnitude
  figure()
  imshow(sqrt(Ix.^2 + Iy.^2),"gray", interpolation="none")
  axis("off")
  title("Derivative magnitude")
  gcf()

  # threshold derivative
  threshold = 0.1
  edges = detectedges(Ix,Iy,threshold)
  figure()
  imshow(edges.>0, "gray", interpolation="none")
  axis("off")
  title("Binary edges")
  gcf()

  # non maximum suppression
  edges2 = nonmaxsupp(edges,Ix,Iy)
  figure()
  imshow(edges2.>0,"gray", interpolation="none")
  axis("off")
  title("Non-maximum suppression")
  gcf()

  return
end
