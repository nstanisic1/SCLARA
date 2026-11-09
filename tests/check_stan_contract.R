suppressPackageStartupMessages(library(rstan))
suppressPackageStartupMessages(library(jsonlite))
options(mc.cores=1)
root <- normalizePath(".",winslash="/")
inputs <- fromJSON(file.path(root,"tests/stan_contract_inputs.json"),simplifyVector=FALSE)
model <- stan_model(file=file.path(root,"analysis/mixture.stan"),model_name="sclara_contract",auto_write=FALSE)
results <- list()
for (name in names(inputs)) {
 d <- inputs[[name]]; x <- do.call(rbind,lapply(d$x,unlist))
 dat <- list(N=nrow(x),P=ncol(x),L=length(d$y),K=3L,X=x,y=as.integer(unlist(d$y))+1L,
             counts=as.integer(unlist(d$counts)),starts=as.integer(unlist(d$starts))+1L,scales=unlist(d$scales))
 single <- which(dat$counts==1L)
 dat$Ns <- length(single);dat$Xs <- x[dat$starts[single],,drop=FALSE];dat$ys <- dat$y[single]
 dat$mi <- which(dat$counts>1L);dat$Lm <- length(dat$mi)
 fit <- sampling(model,data=dat,chains=1,iter=1,warmup=0,algorithm="Fixed_param",init=0,refresh=0)
 offsets <- numeric(); gradient_error <- 0
 for (q in d$probes) {
   v <- unlist(q$v)
   offsets <- c(offsets,log_prob(fit,v)-q$log_density)
   gradient_error <- max(gradient_error,max(abs(grad_log_prob(fit,v)-unlist(q$gradient))))
 }
 stopifnot(diff(range(offsets))<1e-10,gradient_error<1e-10)
 results[[name]] <- list(singleton_units=dat$Ns,multiple_units=dat$Lm,
                         density_offset_spread=diff(range(offsets)),maximum_gradient_error=gradient_error)
 cat("PASS",name,"singleton",dat$Ns,"multiple",dat$Lm,"\n")
}
dir.create("build",showWarnings=FALSE)
write_json(list(model_sha256=digest::digest(file="analysis/mixture.stan",algo="sha256"),cases=results),
 "build/stan_contract_check.json",pretty=TRUE,auto_unbox=TRUE,digits=16)
