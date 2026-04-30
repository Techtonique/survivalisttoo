// cox_boost_core.cpp
#include <Rcpp.h>
#include <numeric>    // std::iota
#include <algorithm>  // std::sort
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector cox_gradient_at_F(NumericVector time,
                                IntegerVector event,
                                NumericVector F_scores) {
  int n = time.size();
  
  std::vector<int> ord(n);
  std::iota(ord.begin(), ord.end(), 0);
  std::sort(ord.begin(), ord.end(), [&](int a, int b) {
    return time[a] < time[b];
  });
  
  NumericVector time_s(n), F_s(n), eF_s(n);
  IntegerVector event_s(n);
  
  for (int i = 0; i < n; i++) {
    time_s[i]  = time[ord[i]];
    event_s[i] = event[ord[i]];
    F_s[i]     = F_scores[ord[i]];
    eF_s[i]    = std::exp(F_s[i]);
  }
  
  // Reverse cumulative sum of exp(F): risk-set denominator at each rank
  NumericVector rev_cumsum_eF(n);
  double running = 0.0;
  for (int i = n - 1; i >= 0; i--) {
    running += eF_s[i];
    rev_cumsum_eF[i] = running;
  }
  
  // Breslow increments -> cumulative hazard at each sorted time
  NumericVector H_s(n);
  double H_cum = 0.0;
  for (int i = 0; i < n; i++) {
    if (event_s[i] == 1) H_cum += 1.0 / rev_cumsum_eF[i];
    H_s[i] = H_cum;
  }
  
  // Martingale residuals, mapped back to original order
  NumericVector resid(n);
  for (int i = 0; i < n; i++) {
    resid[ord[i]] = event_s[i] - eF_s[i] * H_s[i];
  }
  return resid;
}

// [[Rcpp::export]]
List breslow_F(NumericVector time,
               IntegerVector event,
               NumericVector risk_scores) {
  int n = time.size();
  
  std::vector<int> ord(n);
  std::iota(ord.begin(), ord.end(), 0);
  std::sort(ord.begin(), ord.end(), [&](int a, int b) {
    return time[a] < time[b];
  });
  
  NumericVector time_s(n), escore_s(n);
  IntegerVector event_s(n);
  
  for (int i = 0; i < n; i++) {
    time_s[i]   = time[ord[i]];
    event_s[i]  = event[ord[i]];
    escore_s[i] = std::exp(risk_scores[ord[i]]);
  }
  
  // Unique event times (data is sorted, so just track changes)
  std::vector<double> evt_times;
  for (int i = 0; i < n; i++)
    if (event_s[i] == 1) {
      double t = time_s[i];
      if (evt_times.empty() || evt_times.back() != t)
        evt_times.push_back(t);
    }
    
    int K = evt_times.size();
    NumericVector H0(K);
    
    // Two-pointer O(n + K) Breslow accumulation
    double total = 0.0;
    for (int i = 0; i < n; i++) total += escore_s[i];
    
    int obs_ptr = 0;
    double excluded = 0.0;
    
    for (int k = 0; k < K; k++) {
      double t_k = evt_times[k];
      
      // Remove obs with time < t_k from the risk set
      while (obs_ptr < n && time_s[obs_ptr] < t_k) {
        excluded += escore_s[obs_ptr];
        obs_ptr++;
      }
      double risk_sum = total - excluded;
      
      // Count events at t_k
      int ev_count = 0;
      for (int i = obs_ptr; i < n && time_s[i] == t_k; i++)
        if (event_s[i] == 1) ev_count++;
        
        H0[k] = (double)ev_count / risk_sum;
    }
    
    // Cumulative sum in-place
    for (int k = 1; k < K; k++) H0[k] += H0[k - 1];
    
    return List::create(
      Named("times") = NumericVector(evt_times.begin(), evt_times.end()),
      Named("H0")    = H0
    );
}