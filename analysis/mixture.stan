data {

  int<lower=1> N;

  int<lower=1> P;

  int<lower=1> L;

  int<lower=2> K;

  matrix[N,P] X;
  int<lower=0> Ns;
  matrix[Ns,P] Xs;
  array[Ns] int<lower=1,upper=K> ys;
  int<lower=0> Lm;
  array[Lm] int<lower=1,upper=L> mi;

  array[L] int<lower=1,upper=K> y;

  array[L] int<lower=1> counts;

  array[L] int<lower=1,upper=N> starts;

  vector<lower=0>[P] scales;

}

transformed data {

  matrix[K-1,K-1] Sigma = rep_matrix(0.5,K-1,K-1);

  matrix[K-1,K-1] prior_chol;

  for (k in 1:(K-1)) Sigma[k,k] = 1;

  prior_chol = cholesky_decompose(Sigma);

}

parameters {

  matrix[P,K-1] beta;

}

model {

  matrix[N,K-1] eta = X * beta;

  for (p in 1:P)

    beta[p]' ~ multi_normal_cholesky(rep_vector(0,K-1), scales[p]*prior_chol);

  if (Ns > 0)
    target += categorical_logit_glm_lpmf(ys | Xs, rep_vector(0,K), append_col(rep_vector(0,P), beta));
  for (u in 1:Lm) {
    int l = mi[u];

    vector[counts[l]] ll;

    for (r in 1:counts[l]) {

      vector[K] logits = append_row(0,eta[starts[l]+r-1]');

      ll[r] = categorical_logit_lpmf(y[l] | logits);

    }

    target += log_sum_exp(ll) - log(counts[l]);

  }

}

generated quantities {

  vector[L] log_lik;

  {

    matrix[N,K-1] eta = X * beta;

    for (l in 1:L) {

      vector[counts[l]] ll;

      for (r in 1:counts[l]) {

        vector[K] logits = append_row(0,eta[starts[l]+r-1]');

        ll[r] = categorical_logit_lpmf(y[l] | logits);

      }

      log_lik[l] = log_sum_exp(ll) - log(counts[l]);

    }

  }

}

