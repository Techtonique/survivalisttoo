# =============================================================================
# CoxGradientBoost: Model-agnostic Cox-gradient boosting survival model (S3)
# Extends SurvivalCustom to multiple gradient-boosting iterations.
#
# Algorithm (one-shot Cox gradient boosting):
#   1.  Initialise F(x) = 0  (log-hazard = 0 everywhere)
#   2.  For m = 1 ... M:
#         a. Compute martingale residuals r_i = delta_i - H(t_i | F)
#            (negative gradient of partial log-likelihood w.r.t. F)
#         b. Fit base learner h_m(x) ~ r
#         c. Update F(x) <- F(x) + nu * h_m(x)
#   3.  Breslow baseline hazard at final F(x)
#
# S3 methods:
#   predict.CoxGradientBoost()  -- type = "lp"       -> linear predictor
#                       -- type = "survival" -> S(t|x) matrix
#                       -- type = "cumhaz"   -> H(t|x) matrix
# =============================================================================

# ---- helpers (shared with SurvivalCustom; source that file first if needed) -

# Generic prediction extractor (ranger, lm, xgboost, e1071::svm, ...)
.extract_pred <- function(pred) {
  if (is.numeric(pred))           return(as.numeric(pred))
  if (!is.null(pred$predictions)) return(as.numeric(pred$predictions))
  if (!is.null(pred$pred))        return(as.numeric(pred$pred))
  as.numeric(pred)
}

# =============================================================================
# Progress bar (pure base R, no dependencies)
# =============================================================================
.pb_init <- function(total, width = 50L) {
  list(total = total, width = width, start = proc.time()[["elapsed"]])
}

.pb_update <- function(pb, i) {
  frac  <- i / pb$total
  filled <- round(frac * pb$width)
  bar   <- paste0(
    "[",
    paste(rep("=", filled),    collapse = ""),
    paste(rep(" ", pb$width - filled), collapse = ""),
    "]"
  )
  elapsed <- proc.time()[["elapsed"]] - pb$start
  eta     <- if (frac > 0) elapsed / frac * (1 - frac) else NA_real_
  eta_str <- if (!is.na(eta)) sprintf(" ETA %ds", round(eta)) else ""
  cat(sprintf("\rIter %*d/%d %s %3.0f%%%s",
              nchar(pb$total), i, pb$total, bar, frac * 100, eta_str))
  if (i == pb$total) cat("\n")
  invisible(pb)
}

# =============================================================================
# Constructor
# =============================================================================

#' Cox Gradient Boosting Model
#' 
#' @description Fit a model-agnostic Cox-gradient boosting survival model
#'
#' @param X            numeric matrix of covariates (n x p)
#' @param time         numeric vector of observed times
#' @param event        integer/logical vector (1 = event, 0 = censored)
#' @param regr_fun     function(X, y, ...) -> model with predict() method
#' @param M            number of boosting iterations (default 100)
#' @param nu           learning rate / shrinkage (default 0.1)
#' @param show_progress logical; print a progress bar? (default TRUE)
#' @param ...          extra arguments forwarded to regr_fun at every iteration
#'
#' @return object of class "CoxGradientBoost"
#'
#' @examples
#' 
#' require(glmnet)
#' require(survival)
#' 
#' data(ovarian)
#' 
#' set.seed(42)
#' idx_train <- sample(nrow(ovarian), floor(0.75 * nrow(ovarian)))
#' df <- ovarian
#' train <- df[idx_train, ]; test <- df[-idx_train, ]
#'
#' regr_lm <- function(X, y, ...) lm(y ~ ., data = data.frame(X, y = y))
#' 
#' fit_boost_lm <- survivalisttoo::cox_gradient_boost(train, train$futime, 
#' train$fustat, regr_lm)
#' 
#' y_test   <- Surv(test$futime, test$fustat)
#' (ci_blm <- glmnet::Cindex(predict(fit_boost_lm, test), y_test)) # C-index
#'
#'
#' fit_boost_lm <- survivalisttoo::cox_gradient_boost(train, train$futime, 
#' train$fustat, regr_lm, M=1, nu=1)
#' 
#' y_test   <- Surv(test$futime, test$fustat)
#' (ci_blm <- glmnet::Cindex(predict(fit_boost_lm, test), y_test)) # C-index
#' 
#' @export 
cox_gradient_boost <- function(X, time, event,
                      regr_fun,
                      M            = 100L,
                      nu           = 0.1,
                      show_progress = TRUE,
                      ...) {
  X     <- as.matrix(X)
  event <- as.integer(event)
  n     <- nrow(X)
  Xdf   <- data.frame(X)
  
  if (nu <= 0 || nu > 1) stop("`nu` must be in (0, 1]")
  if (M  < 1)            stop("`M` must be a positive integer")
  
  # Initialise ensemble: F(x) = 0 for all x
  F_train <- rep(0.0, n)
  models  <- vector("list", M)
  
  pb <- if (show_progress) .pb_init(M) else NULL
  
  for (m in seq_len(M)) {
    # Negative gradient (martingale residuals at current F)
    resid <- cox_gradient_at_F(time, event, F_train)
    
    # Fit base learner to pseudo-residuals
    h_m <- regr_fun(X, resid, ...)
    models[[m]] <- h_m
    
    # Update ensemble
    h_pred  <- .extract_pred(predict(h_m, Xdf))
    F_train <- F_train + nu * h_pred
    
    if (show_progress) .pb_update(pb, m)
  }
  
  # Breslow baseline at final ensemble scores
  baseline <- breslow_F(time, event, F_train)
  
  structure(
    list(
      models       = models,
      baseline     = baseline,
      F_train      = F_train,
      nu           = nu,
      M            = M,
      time         = time,
      event        = event,
      X            = X
    ),
    class = "CoxGradientBoost"
  )
}

# =============================================================================
# print method
# =============================================================================

#' Print a CoxGradientBoost Object
#'
#' Displays a concise summary of a fitted \code{CoxGradientBoost} model,
#' including the base learner, number of boosting iterations, learning rate,
#' and basic information about the training data.
#'
#' @param x An object of class \code{"CoxGradientBoost"}.
#' @param ... Further arguments passed to or from other methods (currently unused).
#'
#' @details
#' This method prints key characteristics of the fitted model:
#' \itemize{
#'   \item The base learner used in boosting
#'   \item The number of boosting iterations (\eqn{M})
#'   \item The learning rate (\eqn{\nu})
#'   \item The number of observations
#'   \item The number of observed events
#'   \item The number of unique event times
#' }
#'
#' @return The input object \code{x}, invisibly.
#'
#' @seealso \code{\link{cox_gradient_boost}}, \code{\link{predict.CoxGradientBoost}}
#'
#' @method print CoxGradientBoost
#' @export
print.CoxGradientBoost <- function(x, ...) {
  cat("CoxGradientBoost\n")
  cat("  Base learner  :", if (length(x$models)) class(x$models[[1L]])[1L] else "NA", "\n")
  cat("  Iterations (M):", x$M,  "\n")
  cat("  Learning rate :", x$nu, "\n")
  cat("  n             :", length(x$time), "\n")
  cat("  Events        :", sum(x$event), "\n")
  cat("  Unique times  :", length(x$baseline$times), "\n")
  invisible(x)
}

# =============================================================================
# predict method
# =============================================================================

#' Predict from a CoxGradientBoost model
#'
#' @param object     a CoxGradientBoost object
#' @param newdata    numeric matrix or data.frame of new covariates
#' @param type       "lp" (default), "survival", or "cumhaz"
#' @param times      time grid for type != "lp"; NULL -> training event times
#' @param M          use only the first M base learners (early stopping probe);
#'                   NULL -> all M (default)
#' @param ...        unused
#'
#' @export 
#'
#' @return
#'   type = "lp"       -> numeric vector (length n_new)
#'   type = "survival" -> matrix (n_new x length(times))
#'   type = "cumhaz"   -> matrix (n_new x length(times))
predict.CoxGradientBoost <- function(object,
                             newdata,
                             type  = "lp",
                             times = NULL,
                             M     = NULL,
                             ...) {
  newdata <- data.frame(as.matrix(newdata))
  M_use   <- if (is.null(M)) object$M else min(as.integer(M), object$M)
  
  # Accumulate ensemble prediction F(x) = nu * sum_m h_m(x)
  lp <- rep(0.0, nrow(newdata))
  for (m in seq_len(M_use)) {
    h_pred <- try(.extract_pred(predict(object$models[[m]], newdata)), 
                  silent = TRUE)
    if (inherits(h_pred, "try-error"))
    {
      h_pred <- try(.extract_pred(predict(object$models[[m]], as.matrix(newdata))), 
                                  silent = TRUE)
      if (inherits(h_pred, "try-error"))
      {
        stop("Prediction failed.")
      }
    }
      
    lp     <- lp + object$nu * h_pred
  }
  
  if (type == "lp") return(lp)
  
  if (is.null(times)) times <- object$baseline$times
  times <- sort(times)
  
  H0_step <- stats::stepfun(object$baseline$times, c(0, object$baseline$H0))
  H0_at_t <- H0_step(times)
  
  exp_lp  <- exp(lp)
  H_mat   <- outer(exp_lp, H0_at_t)
  colnames(H_mat) <- as.character(times)
  
  if (type == "cumhaz")  return(H_mat)
  if (type == "survival") return(exp(-H_mat))
  
  stop("`type` must be one of 'lp', 'survival', 'cumhaz'")
}