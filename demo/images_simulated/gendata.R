noise_level = 0.2

n_train = 1000
n_target = 100
n_alt = 100

img_width = 128
img_height = 128
img_size = img_width * img_height

gentargetimg <- function() {
   img = matrix(runif(img_size, 0, noise_level), img_height, img_width)

   coords = expand.grid(seq_len(img_height), seq_len(img_width))
   cx = round(runif(1, 40, 80))
   cy = round(runif(1, 40, 80))
   r = round(runif(1, 40, 120))
   ind1 = ((coords[, 1] - cx)^2 + (coords[, 2] - cy)^2) < r

   img[ind1] = img[ind1] + 3 * noise_level

   return(img)
}

genalt1img <- function() {
   # here we generate images with circles of the same size but then add
   # an additional smaller circle

   # initial image filled with random numbers between 0 and "noise_level"
   img = matrix(runif(img_size, 0, noise_level), img_height, img_width)

   # get (x, y) coordinates of each pixel on the image
   coords = expand.grid(seq_len(img_height), seq_len(img_width))

   # generate coordinates and radius of a main circle
   cx = round(runif(1, 40, 80))
   cy = round(runif(1, 40, 80))
   r = round(runif(1, 40, 120))

   # find pixels inside the circle and increase their intensity
   ind1 = ((coords[, 1] - cx)^2 + (coords[, 2] - cy)^2) < r
   img[ind1] = img[ind1] + 3 * noise_level


   # generate coordinates and radius of a smaller circle
   r = round(runif(1, 5, 10))
   cx = cx + (ifelse(runif(1) > 0.5, 1, -1)) * (r + round(runif(1, r, r + 5)))
   cy = cy + (ifelse(runif(1) > 0.5, 1, -1)) * (r + round(runif(1, r, r + 5)))


   # find pixels inside the circle (without overlapping with the main one) and increase their intensity
   ind2 = ((coords[, 1] - cx)^2 + (coords[, 2] - cy)^2) < r
   ind2 = !ind1 & ind2
   img[ind2] = img[ind2] + 3 * noise_level

   return(img)
}


genalt2img <- function() {
   # here we generate images wich are similar to target images but have
   # difference intensities

   # initial image filled with random numbers between 0 and "noise_level"
   img = matrix(runif(img_size, 0, noise_level), img_height, img_width)

   coords = expand.grid(seq_len(img_height), seq_len(img_width))
   cx = round(runif(1, 40, 80))
   cy = round(runif(1, 40, 80))
   r = round(runif(1, 40, 120))
   ind1 = ((coords[, 1] - cx)^2 + (coords[, 2] - cy)^2) < r

   img[ind1] = img[ind1] + 4 * noise_level

   return(img)
}

# training set
for (i in 1:n_train) {
   img = gentargetimg()
   squash::savemat(img, sprintf("train/target/t%d.png", i))
}

# test set - target
for (i in 1:n_target) {
   img = gentargetimg()
   squash::savemat(img, sprintf("target/target/t%d.png", i))
}

# test set - alt1
for (i in 1:n_alt) {
   img = genalt1img()
   squash::savemat(img, sprintf("alt1/alt1/a%d.png", i))
}

# test set - alt2
for (i in 1:n_alt) {
   img = genalt2img()
   squash::savemat(img, sprintf("alt2/alt2/a%d.png", i))
}

