#variaitional autoencorders, deep compression algorithms
library(keras)
library(dplyr)
library(stringr)
library(tidyr)
library(ggplot2)
library(ggpubr)
library(tensorflow)
library(umap)
library(tfdatasets)
library(glue)

length <- 100 #length of the image
width <- 100 #width of the image
image.blank <- matrix(rep(0, (width*length)), nrow = length, ncol = width) #inialize a matrix
matrix_storage <- matrix(0, nrow = 100, ncol = 100) #used to store a list of matrices for testing
matrix_storage2 <- matrix(0, nrow = 100, ncol = 100) #used to store a list of matrices for training

image.generator <- function(shape){ #generates a shape given a string with name of desired shape
  # the output will be a matrix with the desired shape as 1s
  length <- 100
  width <- 100
  image.blank <- matrix(rep(0, (width*length)), nrow = length, ncol = width) #inialize the matrix
  if (shape == "square"){ #if the input was square
    return(square(image.blank)) #use square function to generate a square
  }
  if (shape == "random"){ #if the input was random
    return(randomstring(length, width)) #use square function to generate a random matrix of 1 and 0
    #return(as.vector(randomstring(length, width)))
  }
  if (shape == "lines"){ #if the input was lines
    return(lines(image.blank)) #use square function to generate random orientation of lines
  }
}
square <- function(imag){ #generates a centered square with max side length 50
  random.int <- sample(1:25, 1) #generate random number for half the length of a square
  for (i in 0:random.int){ #main for loop
    for (x in 0:random.int){ #secondary for loop
      imag[((width/2)-x),((length/2)-i)] <- 1 #changes one quadrant to 1s
      imag[((width/2)-x),((length/2)+i)] <- 1 #^
      imag[((width/2)+x),((length/2)+i)] <- 1 #^
      imag[((width/2)+x),((length/2)-i)] <- 1 #^ 
    }
  }
  return(imag) #return a matrix with the values
}

randomstring <- function(length, width){ #takes a matrix and fills it with 1 and 0 randomly
  #initialize the variables
  i<-0 
  string.1 <- c()
  while (i<(length*width)){ #main loop
    i <- i + 1 #counter
    string.1 <- c(string.1, sample(0:1, 1, replace = T)) # make a string with random values
  }
  string.1 <- matrix(string.1, nrow = length, ncol = width) #create matrix from the string
  return(string.1) #return matrix
}

lines <- function(imag){ #generate a matrix that has random rows and columns of 1s 
  i<-0
  random.int <- sample(1:25, 1) #generate random number
  while (i < random.int){ #main while loop
    i <- i + 1 #counter
    coin <- runif(1,0,1) #change the row to all 1s
    if (coin >= 0.5){
      a <- sample(1:length, 1, replace = F)
      imag[a, ] <- 1
    }
    else{ #change the column to all 1s
      a <- sample(1:width, 1, replace = F)
      imag[ ,a] <- 1
    }
  }
  #image(rotate(imag))
  return(imag) #returns a matrix
}
rotate <- function(x) t(apply(x, 2, rev))

#Generates training data
# this makes a list of 2000 of each image category in a row
num.for <- 999
for (i in 1:num.for){ #main for loop
  if (i<= (num.for/3)){ #generates 2000 samples of squares
    matrix_storage2 <- cbind(matrix_storage2, image.generator("square"))}
  if ((num.for/3) < i && i<= (num.for*(2/3))){
    matrix_storage2 <- cbind(matrix_storage2, image.generator("random"))
  }
  if ((num.for*(2/3)) < i&& i <= num.for){
    matrix_storage2 <- cbind(matrix_storage2, image.generator("lines"))
  }
}
#generates the category name data of 2000 square, random, and line images
category_names2 <- array(c(0,rep(0, 333), rep(1, 333), rep(2, 333)), 1000)
#0 corresponds to square
#1 corresponds to random
#2 corresponds to lines

#Generates test data
# this makes a list of 700 of each image category in a row
num.for2 <- 333
for (i in 1:num.for2){ #main for loop
  if (i<= (num.for2/3)){ #generates 2000 samples of squares
    matrix_storage <- cbind(matrix_storage, image.generator("square"))}
  if ((num.for2/3) < i && i<= (num.for2*(2/3))){
    matrix_storage <- cbind(matrix_storage, image.generator("random"))
  }
  if ((num.for2*(2/3)) < i&& i <= num.for2){
    matrix_storage <- cbind(matrix_storage, image.generator("lines"))
  }
}

#generates the category name data of 700 square, random, and line images
category_names <- array(c(rep(0, 112), rep(1, 111), rep(2, 111)), 334) #generates a list of category names
#0 corresponds to square
#1 corresponds to random
#2 corresponds to lines

#save my generated data.Change this to your directory if you want...
setwd("/Users/Aaron/Documents/COGS/Project Stuff")
save(matrix_storage, file="matrix_storage.RData")
save(matrix_storage2, file= "matrix_storage2.Rdata")

#rearrange Data in to Arrays to make them fit the input
arr_matrix_storage <- array(as.vector(matrix_storage), dim=c(334,100, 100))
arr_matrix_storage2 <- array(as.vector(matrix_storage2), dim=c(1000, 100, 100))
###################################################################################################

#I chose to rename my data to match examples
x_train <- arr_matrix_storage2
y_train <- to_categorical(category_names2, 3)
x_test <-arr_matrix_storage
y_test <- to_categorical(category_names, 3)
class_names = c('square',
                'random',
                'lines')
loss<- c()
accuracy <-c()
## Building My Basic Model
my_model <- keras_model_sequential()
my_model %>%
  layer_flatten(input_shape = c(100,100)) %>%
  layer_dense(units =100, activation = 'relu') %>% #I count this as layer 1
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 5, activation = 'relu') %>%  #Icount this as layer 2
  layer_dense(units = 100, activation = 'relu') %>% #I count this as layer 3
  layer_dense(units =3, activation = 'softmax')

my_model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)
# saving model and running it on different data 
history5<- as.data.frame(my_model %>% fit (x_train, y_train, epochs = 10)) #I manually changed this line
#with the number corresponding to the number of nodes
score <- my_model %>% evaluate(x_test, y_test)

cat('Test loss', score$loss, "\n")
cat('Test accuracy', score$acc, "\n")

loss <- c(loss, score$loss)
accuracy <- c(accuracy, score$acc)

#this is data that I manually recorded (don't be mad I thought it was more time efficient)
Nodes1<- (1:5)
tl1 <- c( 1.98 ,2.24, 2.21, 1.98, 3.20)
ta1 <- c(0.33, 0.35, 0.27, 0.2485, 0.3323)
Test_loss <- data_frame(
  "tl" <- tl1,
  "Nodes" <- Nodes1
)
Test_accuracies <- data_frame(
  "ta" <- ta1,
  "Nodes" <- Nodes1
)

#filter through dataframe selecting for loss for nodes 1 to 5
f.lossdata1 <- filter(history1, metric == "loss")
f.lossdata2 <- filter(history2, metric == "loss")
f.lossdata3 <- filter(history3, metric == "loss")
f.lossdata4 <- filter(history4, metric == "loss")
f.lossdata5 <- filter(history5, metric == "loss")
#filter through dataframe selecting for accuracy for nodes 1 to 5
f.accdata1 <- filter(history1, metric == "accuracy")
f.accdata2 <- filter(history2, metric == "accuracy")
f.accdata3 <- filter(history3, metric == "accuracy")
f.accdata4 <- filter(history4, metric == "accuracy")
f.accdata5 <- filter(history5, metric == "accuracy")

#plotting loss across epochs
#this generate a graph of final loss on test data for the node values
test.loss <- ggplot() + geom_line(data=Test_loss, aes(x = Nodes1, y= tl1))+
  xlab('Nodes') +
  ylab('Test Loss')
#this generates a graph of final accuracy on test data for the node values
test.accuracies <- ggplot() + geom_line(data=Test_accuracies, aes(x = Nodes1, y= ta1))+
  xlab('Nodes') +
  ylab('Test Accuracy')
#this generates graphs of loss during training for each node
graph.loss <-  ggplot() + 
  geom_line(data = f.lossdata1, aes(x = epoch, y = value, color = "1 node")) +
  geom_line(data = f.lossdata2, aes(x = epoch, y = value, color = "2 nodes")) +
  geom_line(data = f.lossdata3, aes(x = epoch, y = value, color = "3 nodes")) +
  geom_line(data = f.lossdata4, aes(x = epoch, y = value, color = "4 nodes")) +
  geom_line(data = f.lossdata5, aes(x = epoch, y = value, color = "5 nodes")) +
  xlab('Epoch') +
  ylab('Loss') + theme(legend.position = c(1,1),
                       legend.justification = c("right", "top"))
#this generates graphs of accuracy during training for each node
graph.acc <-  ggplot() + 
  geom_line(data = f.accdata1, aes(x = epoch, y = value, colour = "1 node"))+
  geom_line(data = f.accdata2, aes(x = epoch, y = value, colour = "2 nodes")) +
  geom_line(data = f.accdata3, aes(x = epoch, y = value, color = "3 nodes")) +
  geom_line(data = f.accdata4, aes(x = epoch, y = value, color = "4 nodes")) +
  geom_line(data = f.accdata5, aes(x = epoch, y = value, color = "5 nodes")) +
  xlab('Epoch') +
  ylab('Accuracy') + 
  theme(legend.position = c(1,1),
        legend.justification = c("right", "top"))
#the next line stitches the above graphs together
ggarrange(graph.acc, graph.loss, test.loss, test.accuracies + rremove("x.text"), 
          labels = c("A", "B", "C", "D"),
          ncol = 2, nrow = 2)
############################## Create second DNN with 4 layers ##############################
############################## Create second DNN with 4 layers ##############################
############################## Create second DNN with 4 layers ##############################
############################## Create second DNN with 4 layers ##############################

#I chose to rename my data to match examples
x_train <- arr_matrix_storage2
y_train <- to_categorical(category_names2, 3)
x_test <-arr_matrix_storage
y_test <- to_categorical(category_names, 3)
class_names = c('square',
                'random',
                'lines')
loss<- c()
accuracy <-c()
## Building My Basic Model
my_model2 <- keras_model_sequential()
my_model2 %>%
  layer_flatten(input_shape = c(100,100)) %>%
  layer_dense(units =100, activation = 'relu') %>% #I count this as layer 1
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 5, activation = 'relu') %>% #I count this as layer 2
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 5, activation = 'relu') %>% #I count this as layer 3
  layer_dense(units = 100, activation = 'relu') %>% # I count this as layer 4
  layer_dense(units =3, activation = 'softmax')

my_model2 %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

tensorboard("logs/run_a") #hey look I'm wising up and using tensorboard :)

history <- my_model2 %>% fit(
  x_train, y_train,
  epochs = 5,
  callbacks = callback_tensorboard("logs/run_a")
)


#with the number corresponding to the number of nodes
score <- my_model %>% evaluate(x_test, y_test)

cat('Test loss', score$loss, "\n")
cat('Test accuracy', score$acc, "\n")

loss <- c(loss, score$loss)
accuracy <- c(accuracy, score$acc)



############################### Autoencoder #######################################
############################### Autoencoder #######################################
############################### Autoencoder #######################################
############################### Autoencoder #######################################
#https://github.com/rstudio/keras/blob/master/vignettes/examples/variational_autoencoder.R
#L26

K <- keras::backend()

# Parameters --------------------------------------------------------------
#I again rename my variables to better fit the examples
x_train <- arr_matrix_storage2
x_test <-arr_matrix_storage
#I reshape my arrays to match the input layer of the autoencoder
# but this proved difficult as the visualizations were a composite of all three category types
# which while interesting was definitely wrong.
x_train <- array_reshape(arr_matrix_storage2, c(nrow(arr_matrix_storage2), 10000), order = "C")
x_test <- array_reshape(arr_matrix_storage, c(nrow(arr_matrix_storage), 10000), order = "C")
x_train <- matrix(arr_matrix_storage2, 1000, 100*100)
x_test <- matrix(arr_matrix_storage, 334, 100*100)
arr_matrix_storage <- array(as.vector(matrix_storage), dim=c(334,100, 100))
arr_matrix_storage2 <- array(as.vector(matrix_storage2), dim=c(1000, 100, 100))
#Building the model -----------------------------------------------------

#initialize the variables and define the bottlenecks
batch_size <- 100L
original_dim <- 10000L
latent_dim <- 100L #this is the bottleneck size
intermediate_dim <- 625L #this is the layer before and after the bottleneck
epochs <- 10L
epsilon_std <- 1.0

#builds a composite network by having layers reference each other
x <- layer_input(shape = c(original_dim))
h <- layer_dense(x, intermediate_dim, activation = "relu") 
z_mean <- layer_dense(h, latent_dim)
z_log_var <- layer_dense(h, latent_dim)

sampling <- function(arg){ #this function adds noise the to the compression to prevent
  z_mean <- arg[, 1:(latent_dim)] #overtraining.
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]), 
    mean=0.,
    stddev=epsilon_std
  )
  
  z_mean + k_exp(z_log_var/2)*epsilon #this outputs a tensor with the random variation mixed in
}

# note that "output_shape" isn't necessary with the TensorFlow backend
z <- layer_concatenate(list(z_mean, z_log_var)) %>% #combines the split layers
  layer_lambda(sampling) #adds variation to the combined layers

# we instantiate these layers separately so as to reuse them later
decoder_h <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
h_decoded <- decoder_h(z)
x_decoded_mean <- decoder_mean(h_decoded)

# end-to-end autoencoder
vae <- keras_model(x, x_decoded_mean) # gives the full model

# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean) #gives model up to bottleneck

# generator, from latent space to reconstructed inputs
decoder_input <- layer_input(shape = latent_dim)
h_decoded_2 <- decoder_h(decoder_input)
x_decoded_mean_2 <- decoder_mean(h_decoded_2)
generator <- keras_model(decoder_input, x_decoded_mean_2) #gives just the decoding section as a model


vae %>% compile(optimizer = "adam", loss ='mean_squared_error', metric= 'mse')


# Model training ----------------------------------------------------------

vae %>% fit(
  x_train, x_train, 
  shuffle = TRUE, 
  epochs = 3, 
  batch_size = batch_size, 
  validation_data = list(x_test, x_test)
)

# Visualizations ----------------------------------------------------------

#creates a matrix of predicted values based on x_test
x_test_encoded <-encoder %>% predict(x_test)
x_test_decoded <- vae %>% predict(x_test)

input_data_image <- image(matrix(x_test, nrow=100, ncol=100))
#I'm not sure if the following images were extracted correctly
#they seem to give a lot of noise

#These lines generate a single image for prediction

z_test <- predict(vae, x_test, batch_size = batch_size)
z_test_map1 <- matrix(z_test[1,], nrow=100, ncol=100)
z_test_map2 <- matrix(z_test[2,], nrow=100, ncol=100)
z_test_map10 <- matrix(z_test[10,], nrow=100, ncol=100)
image(z_test_map)
#Generate a data frame of x_test_encoded 
#and plots the two columns of the dataframe by the class (0,1, or 2)
x_test_encoded %>%
  as_data_frame() %>% 
  mutate(class = as.factor(category_names)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()
################################################################################
#using UMAP to reduce data dimensions (like reducing the columns of the dataframe)
#and visualize in plots
library(umap)

original.data <- umap(as.data.frame(x_test, category_names))
head(original.data$layout, 3)
plot.iris(original.data, category_names)

#this gives the umaping of encoded data
encoded.data <- umap(as.data.frame(x_test_encoded, category_names)) #orgnizes the data into two dimensions
head(encoded.data$layout, 3) #provides a list of first and last parts of the vae.data$layout 
plot.iris(encoded.data, category_names) #THIS REQUIRES RUNNING THE plot.iris FUNCTION BELOW

#this gives the umaping of predictions from the entire model
vae.data2 <- umap(as.data.frame(x_test_decoded, category_names)) #orgnizes the data into two dimensions
head(vae.data2$layout, 3) #provides a list of first and last parts of the vae.data$layout 
plot.iris(vae.data2, category_names) #THIS REQUIRES RUNNING THE plot.iris FUNCTION BELOW

###################################
n <- 10  # figure with 4x4 images of predictions of the generator model
digit_size <- 28 # size of the images

# we will sample n points within [-4, 4] standard deviations
grid_x <- seq(-4, 4, length.out = n)
grid_y <- seq(-4, 4, length.out = n)

rows <- NULL #initialize the variables
for(i in 1:length(grid_x)){ #create main for loop
  column <- NULL #initialize the variables
  for(j in 1:length(grid_y)){ #create secondary for loop
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 100) #create a simple matrix
    column <- rbind(column, predict(generator, z_sample) %>% matrix(ncol = 100) ) #stitch it into a column matrix
  }
  rows <- cbind(rows, column) #stitch into overall matrix
}
rows %>% as.raster() %>% plot() #plot it
########################################################PLOT.IRIS FUNCTION#############################
#https://cran.r-project.org/web/packages/umap/vignettes/umap.html
plot.iris <- function(x, labels, #variables and labels
                      main="A UMAP visualization of x_test_encoded", #title of plot
                      colors=c("#ff7f00", "#e377c2", "#17becf"), #colors of dots
                      pad=0.1, cex=0.65, pch=19, add=FALSE, legend.suffix="", #legend specifications
                      cex.main=1, cex.legend=1) {
  
  layout = x
  if (is(x, "umap")) {
    layout = x$layout
  } 
  
  xylim = range(layout)
  xylim = xylim + ((xylim[2]-xylim[1])*pad)*c(-0.5, 0.5)
  if (!add) {
    par(mar=c(0.2,0.7,1.2,0.7), ps=10)
    plot(xylim, xylim, type="n", axes=F, frame=F)
    rect(xylim[1], xylim[1], xylim[2], xylim[2], border="#aaaaaa", lwd=0.25)  
  }
  points(layout[,1], layout[,2], col=colors[as.integer(labels+1)],
         cex=cex, pch=pch)
  mtext(side=3, main, cex=cex.main)
  
  labels.u = unique(labels)
  legend.pos = "topright"
  legend.text = as.character(labels.u)
  if (add) {
    legend.pos = "bottomright"
    legend.text = paste(as.character(labels.u), legend.suffix)
  }
  legend(legend.pos, legend=legend.text,
         col=colors[as.integer(labels.u)+1],
         bty="n", pch=pch, cex=cex.legend)
}



#################### miscillaneous code ####################
#################### miscillaneous code ####################
#################### miscillaneous code ####################
#################### miscillaneous code ####################
#################### miscillaneous code ####################
#image(rotate(mat))
#f.image <- sapply(image.blank, lines)
#line.m <- apply(image.blank, MARGIN = c(1,2), function(x) x+2)

#for (i in 1:10){
#random.int <- sample(1:50, 1)
#image <- matrix(c(rep(random.int, random.int), rep(0,random.int), rep(random.int, random.int)), nrow = random.int, byrow=T)
#}


#image.1 <- matrix(c(rep(1, random.int), rep(0,random.int), rep(1, random.int)), nrow = random.int, byrow=T)
#image.2 <- matrix(c(sample(0:1),random.int, replace = T))



#https://keras.rstudio.com/articles/tutorial_basic_classification.html
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y


#Pre-Processing Data
fashion_mnist <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test
class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')
library(tidyr)
library(ggplot2)

image_1 <- as.data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")

train_images <- train_images / 255
test_images <- test_images / 255

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}

model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

model %>% fit(train_images, train_labels, epochs = 5)

score <- model %>% evaluate(test_images, test_labels)

cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")

predictions <- model %>% predict(test_images)