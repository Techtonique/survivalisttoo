# survivalisttoo

**Model-Agnostic Survival Analysis in R**

`survivalisttoo` provides a flexible framework for performing survival analysis 
using *any* machine learning algorithm. 

[![Documentation](https://img.shields.io/badge/documentation-is_here-green)](http://docs.techtonique.net/survivalisttoo/index.html)

---

## ✨ Features

* Model-agnostic survival modeling
* Plug-and-play with arbitrary regression/ML learners
* Simple `predict()` interface

---

## 📦 Installation

### From GitHub

```r
install.packages("remotes")
remotes::install_github("Techtonique/survivalisttoo")
```

---

## 🚀 Quick Example

```r
library(survivalisttoo)
library(glmnet)
library(survival)

data(pbc)

pbc2       <- pbc[!is.na(pbc$trt), ]
pbc2$event <- as.integer(pbc$status[!is.na(pbc$trt)] == 2)
pbc2$sex_n <- as.integer(pbc2$sex == "f")

feat_cols <- c("trt","age","sex_n","ascites","hepato","spiders","edema",
               "bili","chol","albumin","copper","alk.phos","ast",
               "trig","platelet","protime","stage")

df <- pbc2[, c("time", "event", feat_cols)]

# Simple imputation
for (col in feat_cols)
  if (any(is.na(df[[col]])))
    df[[col]][is.na(df[[col]])] <- median(df[[col]], na.rm = TRUE)

set.seed(42)
idx_train <- sample(nrow(df), floor(0.75 * nrow(df)))
train <- df[idx_train, ]
test  <- df[-idx_train, ]

X_tr <- as.matrix(train[, feat_cols])
X_te <- as.matrix(test[,  feat_cols])

# Custom regression model
regr_lm <- function(X, y, ...) {
  lm(y ~ ., data = data.frame(X, y = y))
}

# Fit model-agnostic survival boosting
fit_boost_lm <- cox_gradient_boost(
  X=X_tr, time=train$time, event=train$event, regr_fun=regr_lm
)

# Classical Cox model (baseline)
fit_cox <- coxph(
  Surv(time, event) ~ .,
  data = train[, c("time","event",feat_cols)],
  x = TRUE
)

# Evaluate
y_te   <- Surv(test$time, test$event)
ci_blm <- glmnet::Cindex(predict(fit_boost_lm, X_te), y_te)
ci_cox <- glmnet::Cindex(predict(fit_cox, newdata = test), y_te)

cat("\n=== Test-set C-index ===\n")
cat(sprintf("  CoxBoost (LM): %.4f\n", ci_blm))
cat(sprintf("  Classical Cox : %.4f\n", ci_cox))
```

---

## 🔌 Using Different ML Models

You can plug in different learners via wrappers (e.g. from `mlS3`):

```r
library(mlS3)
require(glmnet)
require(survival)

X_tr <- train[, feat_cols]
X_te <- test[,  feat_cols]
y_te <- Surv(test$time, test$event)

# Elastic net (glmnet)
fit_glmnet <- cox_gradient_boost(
  X_tr, train$time, train$event,
  mlS3::wrap_glmnet,
  alpha = 1,
  show_progress = FALSE
)

glmnet::Cindex(predict(fit_glmnet, X_te), y_te)

# SVM (radial)
fit_svm <- cox_gradient_boost(
  X_tr, train$time, train$event,
  mlS3::wrap_svm,
  kernel = "radial",
  show_progress = FALSE
)

glmnet::Cindex(predict(fit_svm, X_te), y_te)

# caret model
fit_caret <- cox_gradient_boost(
  X_tr, train$time, train$event,
  mlS3::wrap_caret,
  method = "enet",
  show_progress = FALSE
)

glmnet::Cindex(predict(fit_caret, X_te), y_te)
```

---

## 📚 How it works

For now, the package implements a **gradient boosting strategy for survival analysis** 
in `cox_gradient_boost`:

1. Iteratively fits a base learner on pseudo-residuals
2. Uses any regression/ML model as the base learner
3. Produces risk scores compatible with survival metrics (e.g. C-index)

This enables combining:

* statistical survival modeling
* modern machine learning flexibility

---

## 📈 Output

Models return objects compatible with:

```r
predict(fit, newdata)
```

which can be directly used with:

```r
glmnet::Cindex()
```

---

## 🧩 Dependencies

* `Rcpp`
* Suggested: `survival`, `glmnet`, `caret`, `mlS3`

---

## 👤 Author

**T. Moudiki**
📧 [thierry.moudiki@gmail.com](mailto:thierry.moudiki@gmail.com)

---

## 📄 License

MIT License

---

## 🚧 Status

* Actively developed
