suppressPackageStartupMessages(library(rstan))

suppressPackageStartupMessages(library(jsonlite))

suppressPackageStartupMessages(library(posterior))

options(mc.cores=1)
Sys.setenv(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

args <- commandArgs(trailingOnly=TRUE)

if(length(args)!=1) stop('Usage: Rscript sample.R path/to/frozen/inference-inputs')
root <- normalizePath(args[1],winslash='/',mustWork=TRUE)

setwd(root)

hash <- function(p) digest::digest(file=p,algo='sha256')

save_json <- function(x,p) write_json(x,p,pretty=TRUE,auto_unbox=TRUE,digits=16)

frozen <- fromJSON('design_freeze.json')

stopifnot(hash('specification.json')==frozen$specification_sha256,hash('designs.json')==frozen$designs_sha256,hash('mixture.stan')==frozen$stan_sha256)

all <- fromJSON('designs.json',simplifyVector=FALSE)$jobs

jobs <- names(all); jobs <- c(jobs[!grepl('_omit_',jobs)],jobs[grepl('_omit_',jobs)])

dir.create('results',showWarnings=FALSE)

batch_start <- proc.time()[3]

model <- stan_model(model_code=paste(readLines('mixture.stan'),collapse='\n'),model_name='lineage_locality_mixture',auto_write=FALSE)

for(requested in jobs) {

 stopifnot(proc.time()[3]-batch_start<8*60*60)

 receipt <- file.path('results',paste0(requested,'.json'))

 if(file.exists(receipt)) {
  previous <- fromJSON(receipt)
  stopifnot(previous$design_sha256==hash('designs.json'), previous$model_sha256==hash('mixture.stan'))
  stopifnot(previous$draws_sha256==hash(file.path('results',paste0(requested,'_draws.csv.gz'))))
  stopifnot(previous$contrasts_sha256==hash(file.path('results',paste0(requested,'_contrasts.csv'))))
  next
 }

 d <- all[[requested]]

 xx <- do.call(rbind,lapply(d$x,unlist))

 dat <- list(N=nrow(xx),P=ncol(xx),L=length(d$y),K=length(d$categories),X=xx,y=as.integer(unlist(d$y))+1L,counts=as.integer(unlist(d$counts)),starts=as.integer(unlist(d$starts))+1L,scales=unlist(d$scales))

 singles <- which(dat$counts==1L)
 dat$Ns <- length(singles);dat$Xs <- xx[dat$starts[singles],,drop=FALSE];dat$ys <- dat$y[singles]
 dat$mi <- which(dat$counts>1L);dat$Lm <- length(dat$mi)
 preflight <- sampling(model,data=dat,chains=1,iter=1,warmup=0,algorithm='Fixed_param',init=0,refresh=0)
 off <- numeric(); ge0 <- 0
 for(q in d$density_checks) {
  off <- c(off,log_prob(preflight,unlist(q$v))-q$log_density)
  ge0 <- max(ge0,max(abs(grad_log_prob(preflight,unlist(q$v))-unlist(q$gradient))))
 }
 stopifnot(diff(range(off))<1e-8,ge0<1e-8)
 rm(preflight);gc(verbose=FALSE)

 cr <- lapply(names(d$contrasts),function(n) list(target=n,comparator='frozen',scope=d$scope,vector=d$contrasts[[n]]))

 contrast_names <- names(d$contrasts)

 started <- proc.time()[3]; attempts <- list()

 for(attempt in 1:2) {

  depth <- if(attempt==1) 12 else 14; target <- if(attempt==1) .95 else .99

  cat('START',requested,'attempt',attempt,'\n');flush.console()

  fit <- sampling(model,data=dat,chains=4,cores=4,iter=2000*attempt,warmup=1000*attempt,seed=d$seed,init_r=1,control=list(adapt_delta=target,max_treedepth=depth),refresh=100)

  ar <- as.array(fit,pars='beta')

  st <- summarise_draws(as_draws_array(ar),'rhat','ess_bulk','ess_tail')

  sp <- do.call(rbind,get_sampler_params(fit,inc_warmup=FALSE))

  wm <- t(vapply(cr,function(q) as.numeric(unlist(q$vector)),numeric(dim(ar)[3])))

  ct <- array(matrix(ar,nrow=dim(ar)[1]*dim(ar)[2],ncol=dim(ar)[3]) %*% t(wm),dim=c(dim(ar)[1:2],length(cr)))

  dimnames(ct) <- list(NULL,NULL,contrast_names)

  cd <- summarise_draws(as_draws_array(ct),'rhat','ess_bulk','ess_tail','mcse_mean')

  cd <- cd[match(contrast_names,cd$variable),]

  dg <- list(max_rhat=max(st$rhat),min_bulk_ess=min(st$ess_bulk),min_tail_ess=min(st$ess_tail),contrast_max_rhat=max(cd$rhat),contrast_min_bulk_ess=min(cd$ess_bulk),contrast_min_tail_ess=min(cd$ess_tail),divergences=sum(sp[,'divergent__']),treedepth_hits=sum(sp[,'treedepth__']>=depth))

  dg$passed <- dg$max_rhat<=1.01 && dg$min_bulk_ess>=400 && dg$min_tail_ess>=400 && dg$contrast_max_rhat<=1.01 && dg$contrast_min_bulk_ess>=400 && dg$contrast_min_tail_ess>=400 && dg$divergences==0 && dg$treedepth_hits==0

  attempts[[attempt]] <- c(list(attempt=attempt,adapt_delta=target,warmup=1000*attempt,retained_per_chain=1000*attempt),dg)

  if(isTRUE(dg$passed)) break

 }

 offsets <- numeric();ge <- 0

 for(q in d$density_checks) {

  v <- unlist(q$v);offsets <- c(offsets,log_prob(fit,v)-q$log_density)

  ge <- max(ge,max(abs(grad_log_prob(fit,v)-unlist(q$gradient))))

 }

 dg$density_error <- diff(range(offsets));dg$gradient_error <- ge

 dg$passed <- isTRUE(dg$passed) && dg$density_error<1e-8 && dg$gradient_error<1e-8

 keep <- seq_len(dat$P*(dat$K-1))

 small <- data.frame(chain=rep(seq_len(dim(ar)[2]),each=dim(ar)[1]),iteration=rep(seq_len(dim(ar)[1]),dim(ar)[2]))

 for(i in keep) small[[dimnames(ar)[[3]][i]]] <- as.vector(ar[,,i])

 # Contrasts are reconstructed from coefficient draws when read.

 draw_path <- file.path('results',paste0(requested,'_draws.csv.gz'))

 con <- gzfile(draw_path,'wt');write.csv(small,con,row.names=FALSE);close(con)

 result <- do.call(rbind,lapply(cr,function(q) data.frame(target=q$target,comparator=q$comparator,scope=q$scope)))

 for(i in seq_along(cr)) {

  v <- as.vector(ct[,,i]); interval <- quantile(v,c(.025,.975))

  result$mean[i] <- mean(v);result$median[i] <- median(v);result$sd[i] <- sd(v)

  result$lower[i] <- interval[1];result$upper[i] <- interval[2]

  result$probability_positive[i] <- mean(v>0)

  result$rhat[i] <- cd$rhat[i];result$ess_bulk[i] <- cd$ess_bulk[i];result$ess_tail[i] <- cd$ess_tail[i]

  result$mcse_mean[i] <- cd$mcse_mean[i]

  qp <- mcse_quantile(ct[,,i],probs=c(.025,.975))

  result$mcse_lower[i] <- qp[1];result$mcse_upper[i] <- qp[2]

  result$all_draws_same_sign[i] <- all(v>0) || all(v<=0)

  result$mcse_probability_positive[i] <- if(result$all_draws_same_sign[i]) NA_real_ else mcse_mean(ct[,,i]>0)

 }

 summary_path <- file.path('results',paste0(requested,'_contrasts.csv'))

 write.csv(result,summary_path,row.names=FALSE)

 write.csv(as.data.frame(st),file.path('results',paste0(requested,'_parameter_diagnostics.csv')),row.names=FALSE)

 save_json(list(job_id=requested,labels=dat$L,localities=dat$N,draws=nrow(small),diagnostics=dg,attempts=attempts,elapsed_seconds=proc.time()[3]-started,design_sha256=hash('designs.json'),model_sha256=hash('mixture.stan'),draws_sha256=hash(draw_path),contrasts_sha256=hash(summary_path),status=if(dg$passed) 'diagnosed' else 'diagnostic failure; withhold inference'),receipt)

 cat('DONE',requested,'passed',dg$passed,'seconds',round(proc.time()[3]-started,2),'\n');flush.console()

 rm(fit,ar,ct,small);gc(verbose=FALSE)

}

