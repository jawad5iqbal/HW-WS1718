using Images
using PyPlot
using JLD


# load and return the given image
function loadimage()
    img = PyPlot.imread(string("01.pgm"))
    return img::Array{Float32,3}
end


# save the image as a .jld file
function savefile(img::Array{Float32,3})
    @save "img.jld" img#savefile(img1)
end


# load and return the .jld file
function loadfile()
        img = load("img.jld")["img"]
        return img::Array{Float32,3}
end


# create and return a vertically mirrored image
function mirrorvertical(img::Array{Float32,3})
    mirrored = flipdim(img,1)
    return mirrored::Array{Float32,3}
end


# display the normal and the mirrored image in one plot
function showimages(img1::Array{Float32,3}, img2::Array{Float32,3})
    figure()

    subplot(1,2,1)

    PyPlot.imshow(img1)

    subplot(1,2,2)

    PyPlot.imshow(img2)

end


#= Problem 1
Load and Display =#

function problem1()

  img1 = loadimage()

  savefile(img1)

  img2 = loadfile()

  img2 = mirrorvertical(img2)

  showimages(img1, img2)

end
