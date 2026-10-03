# Independently refit sampled scorecasts from the full new-source AcMCP-CSS runs.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1]);.libPaths(c(file.path(root,".tools/R-library"),.libPaths()))
suppressPackageStartupMessages(library(forecast))
base <- file.path(root,"results/focused_v10/acmcp_n450")
jobs <- list.dirs(base,recursive=FALSE)
results <- list()
for(job in jobs){
  if(!file.exists(file.path(job,"scorecaster.csv"))) next
  errors <- ts(as.matrix(read.csv(file.path(job,"errors.csv"),header=FALSE)))
  scorecasts <- ts(as.matrix(read.csv(file.path(job,"scorecaster.csv"),header=FALSE)))
  cfg <- read.csv(file.path(job,"config.csv"));ncal <- cfg$ncal
  method <- if("ma_method" %in% names(cfg)) cfg$ma_method else "CSS-ML"
  for(h in intersect(c(1L,3L,6L,12L,24L),seq_len(ncol(errors)))){
    available <- which(is.finite(scorecasts[,h]))-h
    tt <- unique(available[c(1L,ceiling(length(available)/2),length(available))])
    for(t in tt){
      es <- subset(errors[,h],start=max(t-ncal+1L,h),end=t)
      warnings_seen <- character()
      withCallingHandlers({
      if(h==1L){
        computed <- as.numeric(forecast::meanf(es,h=h)$mean)
      } else {
        si <- sapply(seq_len(h-1L),function(j) scorecasts[t-h+1L+j,j])
        model_ma <- forecast::forecast(forecast::Arima(es,order=c(0,0,h-1L),method=method),h=h)
        frame <- setNames(as.data.frame(sapply(seq_len(h),function(j)
          subset(errors[,j],start=t-ncal+1L-h+j,end=t-h+j))),paste0("V",seq_len(h)))
        model_lr <- forecast::forecast(lm(as.formula(paste0("V",h," ~ .")),data=frame),
          newdata=setNames(data.frame(matrix(si,nrow=1L)),paste0("V",seq_len(h-1L))))
        computed <- (as.numeric(model_ma$mean[h])+as.numeric(model_lr$mean))/2
      }
      }, warning=function(w){warnings_seen <<- c(warnings_seen,conditionMessage(w));invokeRestart("muffleWarning")})
      expected <- as.numeric(scorecasts[t+h,h]);difference <- abs(computed-expected)
      stopifnot(is.finite(difference),difference<1e-8)
      results[[length(results)+1L]] <- data.frame(job=basename(job),horizon=h,update_time=t,
        method=method,computed=computed,saved=expected,absolute_difference=difference,
        fit_warning=paste(unique(warnings_seen),collapse="; "))
    }
  }
}
out <- file.path(root,"results/focused_v10/acmcp_n450/scorecast_refits.csv")
write.csv(do.call(rbind,results),out,row.names=FALSE)
cat("Saved scorecast fits reproduced:",length(results),"\n")

