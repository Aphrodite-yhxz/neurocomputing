# Execute the unmodified pinned author's function on target-aligned frozen forecasts.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1]); job <- normalizePath(args[2])
.libPaths(c(file.path(root,".tools","R-library"),.libPaths()))
suppressPackageStartupMessages(library(forecast))
source(file.path(root,"docs/references/v4/acmcp/R/pid.R"))
source(file.path(root,"docs/references/v4/acmcp/R/acmcp.R"))
pf <- ts(as.matrix(read.csv(file.path(job,"mean.csv"),header=FALSE)))
errors <- ts(as.matrix(read.csv(file.path(job,"errors.csv"),header=FALSE)))
cfg <- read.csv(file.path(job,"config.csv"))
object <- list(MEAN=pf,ERROR=errors,forward=TRUE,level=90,x=NULL,series="frozen_forecast_archive")
start <- proc.time()[3]
result <- acmcp(object,alpha=0.1,ncal=cfg$ncal,rolling=TRUE,lr=cfg$lr,
    KI=cfg$KI,Csat=cfg$Csat,scorecast=as.logical(cfg$scorecast),
    integrate=TRUE,ma_method=if("ma_method" %in% names(cfg)) cfg$ma_method else "CSS-ML")
saveRDS(result,file.path(job,"author_result.rds"))
write.table(as.matrix(result$LOWER[[1]]),file.path(job,"lower.csv"),sep=",",row.names=FALSE,col.names=FALSE)
write.table(as.matrix(result$UPPER[[1]]),file.path(job,"upper.csv"),sep=",",row.names=FALSE,col.names=FALSE)
write.table(result$model$scorecaster$upper,file.path(job,"scorecaster.csv"),sep=",",row.names=FALSE,col.names=FALSE)
write.csv(data.frame(horizon=seq_len(ncol(pf)),cp_times=result$cp_times,
    scorecast_times=if(is.null(result$scorecast_times)) 0 else result$scorecast_times),
    file.path(job,"counts.csv"),row.names=FALSE)
writeLines(c(paste("elapsed_seconds",proc.time()[3]-start),capture.output(sessionInfo())),file.path(job,"runtime.txt"))
cat("Author AcMCP complete:",job,"\n")
