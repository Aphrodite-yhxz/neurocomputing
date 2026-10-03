# Unmodified author recursion with four explicitly configured scorecasters.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1]); job <- normalizePath(args[2])
.libPaths(c(file.path(root,'.tools/R-library'),.libPaths()))
suppressPackageStartupMessages(library(forecast))
source(file.path(root,'docs/references/v4/acmcp/R/pid.R'))
pf <- ts(as.matrix(read.csv(file.path(job,'mean.csv'),header=FALSE)))
errors <- ts(as.matrix(read.csv(file.path(job,'errors.csv'),header=FALSE)))
cfg <- read.csv(file.path(job,'config.csv'))
obj <- list(MEAN=pf,ERROR=errors,forward=TRUE,level=90,x=NULL,series='V8_check')
arfun <- function(x,h,...) {
  xx <- head(as.numeric(x),-1); yy <- tail(as.numeric(x),-1)
  den <- sum((xx-mean(xx))^2)
  phi <- if(den<=1e-20) 0 else sum((xx-mean(xx))*(yy-mean(yy)))/den
  phi <- min(.995,max(-.995,phi)); c <- mean(yy)-phi*mean(xx)
  z <- tail(x,1); out <- numeric(h)
  for(i in seq_len(h)) {z <- c+phi*z;out[i]<-z}
  list(mean=out)
}
for(name in c('none','mean','naive','ar1')) {
  fn <- switch(name,none=NULL,mean=forecast::meanf,naive=forecast::naive,ar1=arfun)
  ans <- pid(obj,alpha=.1,symmetric=FALSE,ncal=cfg$ncal,rolling=TRUE,
             integrate=TRUE,scorecast=(name!='none'),scorecastfun=fn,
             lr=cfg$lr,KI=cfg$KI,Csat=.5)
  write.table(as.matrix(ans$LOWER[[1]]),file.path(job,paste0(name,'_lower.csv')),sep=',',row.names=FALSE,col.names=FALSE)
  write.table(as.matrix(ans$UPPER[[1]]),file.path(job,paste0(name,'_upper.csv')),sep=',',row.names=FALSE,col.names=FALSE)
}
cat('All source PID configurations completed\n')
