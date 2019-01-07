#######################################
#
# weighted betti numbers of images
#
#######################################

library(TDA)
library(imager)
library(ggplot2)

# lifetime weighted betti number
wbetti <- function(X, progress=F, sub=FALSE, loc=FALSE){
  d <- dim(X)
  Xlim <- c(1, d[1]);  Ylim <- c(1, d[2]);  by <- 1
  Diag <- gridDiag(FUNvalues = X, lim = cbind(Xlim, Ylim),
                   by = by, sublevel = sub, library = "Dionysus",
                   printProgress = progress, location =loc)
  D <- Diag$diagram
  return(  c(sum(D[D[,1]==0,3]-D[D[,1]==0,2]),sum(D[D[,1]==1,3]-D[D[,1]==1,2])) )
}


############## End of function definition ####################

path <- "betti/"
imgfiles <- list.files(path, pattern=".jpg$")
nfile <- 999

## load images "dt_00???.jpg" 
## and compute (lifetime weighted) betti numbers
out <- file("betti.txt", "w")
for(i in 0:nfile){
  f <- formatC(i,width=5,flag="0")
  print(f)
  img <- load.image(paste0(path,f,".png"))[,,1,1]
#  img <- load.image(paste0(path,"dt_",f,".jpg"))[,,1,1]  # distance transform image
  b1 <- wbetti(img)
  writeLines(f, out, sep=",")
  writeLines(paste0(b1[1],",",b1[2]), out, sep="\n")
}
close(out) 

