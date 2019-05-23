#######################################
#
# weighted betti numbers of images
#
#######################################

library(TDA)
library(imager)
library(ggplot2)

# persistent homology
PH <- function(X, progress=F, sub=FALSE, loc=TRUE){
  d <- dim(X)
  Xlim <- c(1, d[1]);  Ylim <- c(1, d[2]);  by <- 1
  Diag <- gridDiag(FUNvalues = X, lim = cbind(Xlim, Ylim),
                   by = by, sublevel = sub, library = "Dionysus",
                   printProgress = progress, location =loc)
  return( Diag )
}

## make H_0 image
PHimg <- function(D,DL,img,min_life=0.03,max_life=1){
  cycleidx <- which(D[,1]==0 & abs(D[,3]-D[,2])>min_life)
  label_im <- array(0,dim(img)[1:2])
  for (i in seq(along = cycleidx)) {
    life <- D[cycleidx[i],3]-D[cycleidx[i],2]
    x <- DL[[cycleidx[i],1]]
    y <- DL[[cycleidx[i],2]]
    label_im[x,y] <- max(label_im[x,y],life/max_life)
  }
  return(as.cimg(label_im))
}

# plot persistence diagram
plotPD <- function(D,min_life=0){
  cycleidx <- which(D[,3]-D[,2]>min_life)
  c <- as.factor(D[cycleidx,1])
  xylim <- c(min(D[,2:3]),max(D[,2:3]))
  q <- qplot(x=(D[cycleidx,2]),y=(D[cycleidx,3]), col=c,
             xlim=xylim, ylim=xylim)
  print(q)
}

## highlight cycle generators
highlight_cycles <- function(Diag,img,min_life=0.03){
  D <- Diag[["diagram"]]
  DL <- Diag[["cycleLocation"]]
  cycleidx <- which( (D[,1]==1 | D[,1]==0) & D[,3]-D[,2]>min_life)
  plot(img, axes=F)#,rescale=F)
  palette(heat.colors(100))
  for (i in seq(along = cycleidx)) {
    life <- D[cycleidx[i],3]-D[cycleidx[i],2]
    life <- sqrt(life)*400
    for (j in seq_len(dim(DL[[cycleidx[i]]])[1])) {
      if(D[cycleidx[i],1]==0){
        xy <- DL[[cycleidx[i]]][j, , ]
        points(xy[1],xy[2],pch=1,col=life)
      }else{
        lines(DL[[cycleidx[i]]][j, , ], pch = 19, cex = 1, col = life)
      }
    }
  }
}

clip <- function(x, a, b) pmax(a, pmin(x, b) )


############## End of function definition ####################

#### set the paths to the directory containing image files
#args = commandArgs(trailingOnly=TRUE)
#path = args[1]     ## dir containing images

path = "betti"
setwd(path)

imgfiles <- list.files(".", pattern=paste0("[0-9].png$"))
#imgfiles <- imgfiles[1:2]

## batch compute PH
out <- file("betti.txt", "w")
for(f in imgfiles){
#  f <- formatC(i,width=5,flag="0")
  fname <- substr(basename(f), 1, nchar(basename(f)) - 4)
  # ordinary betti number
  img <- load.image(f)
  Diag <- PH(img[,,1,1],sub=F)
  D <- Diag$diagram
  b0 = sum(D[,1]==0)
  b1 = sum(D[,1]==1)

  # persistent betti number
  dt_img <- distance_transform(img,0)-distance_transform(img,1)
  s <- dim(dt_img)
  dt_img <- as.cimg(clip(2*dt_img/s[1] + 0.6,0,1),dim=s)
  save.image(dt_img,paste0(fname,"_dt.png"))
  #  plot(dt_img)
  Diag <- PH(dt_img[,,1,1],sub=F)
  D <- Diag$diagram
  #plotPD(D,0.03)
  #highlight_cycles(Diag,dt_img)
  pb0 = sum(D[D[,1]==0,3]-D[D[,1]==0,2])
  pb1 = sum(D[D[,1]==1,3]-D[D[,1]==1,2])
  writeLines(paste(f,b0,b1,pb0,pb1,sep=","), out, sep="\n")
  print(paste(f,b0,b1,pb0,pb1,min(dt_img),max(dt_img),sep=","))
#  print(paste(sum(D[,1]==0),sum(D[,1]==1)))
    # PH image
  label_im <- PHimg(D,Diag[["birthLocation"]],dt_img[,,1,1])
  save.image(as.cimg(label_im),paste0(fname,"_PH0_life.png"))

  Diag <- PH(dt_img[,,1,1],sub=T)
  D <- Diag$diagram
  #plotPD(D,0.03)
  #highlight_cycles(Diag,dt_img)
  label_im <- PHimg(D,Diag[["birthLocation"]],dt_img[,,1,1])
  save.image(as.cimg(label_im),paste0(fname,"_PH1_life.png"))
  pb0 = sum(D[D[,1]==0,3]-D[D[,1]==0,2])
  pb1 = sum(D[D[,1]==1,3]-D[D[,1]==1,2])
  print(paste(f,b0,b1,pb0,pb1,sep=","))
}
close(out) 

#library(plotly)
#plot_ly(z = dt_img[,,1,1]) %>% add_surface()

