# R script to run DynTxRegime methods.

suppressMessages(library(DynTxRegime))
suppressMessages(library(rpart))

args = commandArgs(trailingOnly=TRUE)
sim_num <- paste("sim", args[[2]], sep="_")
iter <- paste("iter", args[[3]], sep="_")
data_filepath <- paste(paste(args[[1]], sim_num, sep="/"), iter, sep="_")
method <- args[[4]]

set.seed(strtoi(args[[3]]))

binary_threshold <- 25
df <- read.csv(paste(data_filepath, "df_obs.csv", sep="_"))
df <- df[ , colSums(is.na(df)) == 0]

keep_cols <- c("X", "Burden", "Dose", "Conc", "binary_outcome")
df <- df[ , grepl(paste(keep_cols, collapse='|'), names( df )) ]

get_int <- function(x){ strtoi(substr(x, nchar(x), nchar(x))) }
timesteps <- max(get_int(names(df)[grepl("Dose", names(df))])) + 1

if (grepl('linear', method, fixed=TRUE)) {
  solver_method <- 'lm'
} else {
  solver_method <- 'rpart'
}

if (grepl('bowl-', method, fixed = TRUE)) {
  if (grepl('r1', method, fixed = TRUE)) {
    for (i in timesteps:1) {
      df[paste("Reward", i, sep="")] <- (df[paste("Burden", i-1, sep="")] - df[paste("Burden", i, sep="")]) + 0.25*(50 - df[paste("Dose", i-1, sep="")])    
      }
  } else if (grepl('r2', method, fixed = TRUE)) {
    for (i in timesteps:1) {
      df[paste("Reward", i, sep="")] <- -((df[paste("Burden", i, sep="")] * exp(df['X0'])) + (df[paste("Dose", i-1, sep="")] * exp(df['X2'])))
    }
  } else if (grepl('r3', method, fixed = TRUE)) {
    for (i in timesteps:1) {
      df[paste("Reward", i, sep="")] <- -(((exp(df[paste("Burden", i, sep="")] / 50) - 1) * exp(rowMeans(df[c('X0', 'X1')]))) + ((exp(df[paste("Conc", i, sep="")] / 50) - 1) * exp(rowMeans(df[c('X2', 'X3')]))))
    }    
  }
  final_response <- df[[paste("Reward", timesteps, sep="")]]
}

for (d in names(df)[grepl("Dose", names(df))]) {
  df[d] <- as.integer(df[d] > binary_threshold)
}

if (grepl('q-', method, fixed=TRUE) | grepl('optclass-', method, fixed=TRUE)) {
  final_response <- as.integer(!as.logical(df$binary_outcome))
} else if (method == 'causaltree') {
  df["Y"] <- as.integer(!as.logical(df$binary_outcome))
}

scale_cols <- c("X", "Burden")
df[ , grepl(paste(scale_cols, collapse='|'), names( df ))] <- scale(df[ , grepl(paste(scale_cols, collapse='|'), names( df ))])


covs <- c(names(df)[grepl("X", names( df ))])
covs <- append(covs, "Burden0")
if (timesteps > 1) {
    for (i in 0:max((timesteps-2), 0)) {
      covs <- c(covs, c(paste("Dose", i, sep=""), paste("Burden", i+1, sep="")))
    }
}

gen_model <- function(method, covs, df, t, response = NULL, prev_m = NULL, solver_method = 'lm') {
  if (grepl('bowl-', method, fixed = TRUE)) {
    if (grepl('bowl-linear', method, fixed = TRUE)) {
      kernel <- 'linear'
      kparam <- NULL
    } else {
      kernel <- 'poly'
      kparam <- 2
    }
    moPropen<-buildModelObj(model=as.formula(paste( " ~ 1 + ", paste(covs, collapse="+"))), solver.method='glm', solver.args=list('family'='binomial'), predict.method='predict.glm', predict.args=list(type='response'))
    model <- tryCatch({
      return(bowl(moPropen=moPropen, data=df, reward=response, txName=paste("Dose", t, sep=""), regime=reformulate(covs), BOWLObj=prev_m, lambdas = c(0.5, 1.0, 2.0), cvFolds = 5L, kernel = kernel, kparam = kparam, verbose=0L))
    }, error = function(err) {
      tryCatch({
        moPropen<-buildModelObj(model=as.formula(paste( " ~ 1 + ", paste(covs, collapse="+"))), solver.method='glm', solver.args=list('family'='binomial'), predict.method='predict.glm', predict.args=list(type='response'))
        return(bowl(moPropen=moPropen, data=df, reward=response, txName=paste("Dose", t, sep=""), regime=reformulate(covs), BOWLObj=prev_m, lambdas = 2.0, kernel = kernel, kparam = kparam, verbose=0L))
      }, error = function(err) {
        tryCatch({
          moPropen<-buildModelObj(model=as.formula(paste( " ~ 1 + ", paste(covs, collapse="+"))), solver.method='glm', solver.args=list('family'='binomial'), predict.method='predict.glm', predict.args=list(type='response'))
          return(bowl(moPropen=moPropen, data=df, reward=response, txName=paste("Dose", t, sep=""), regime=reformulate(covs), BOWLObj=prev_m, lambdas = 0.5, kernel = kernel, kparam = kparam, verbose=0L))          
        }, error = function(err) {
          print(err)
        })
      })
    })
  } else if (grepl('q-', method, fixed=TRUE)) {
    moMain <- buildModelObj(model = reformulate(covs), solver.method = 'lm')
    moCont <- buildModelObj(model = reformulate(covs), solver.method = solver_method)
    model <- qLearn(moMain = moMain, moCont = moCont, data = df, response = response,  txName = paste("Dose", t, sep=""), verbose=FALSE)
  } else if (grepl('optclass-', method, fixed=TRUE)) {
    moPropen <- buildModelObj(model=as.formula(paste( " ~ 1 + ", paste(covs, collapse="+"))), solver.method='glm', solver.args=list('family'='binomial'), predict.method='predict.glm', predict.args=list(type='response'))
    moMain <- buildModelObj(model=reformulate(covs), solver.method='lm')
    moCont <- buildModelObj(model=reformulate(covs), solver.method=solver_method)
    moClass <- buildModelObj(model=reformulate(covs), solver.method='rpart', solver.args=list(method="class"), predict.args=list(type='class'))
    model <- optimalClass(moPropen=moPropen, moMain=moMain, moCont=moCont, moClass=moClass, data=df, response=response, txName=paste("Dose", t, sep=""), verbose=FALSE)
  } else if (method == 'causaltree') {
    moPropen <- glm(as.formula(paste(paste("Dose", t, sep=""), paste(covs, collapse="+"), sep=" ~ ")), data = df, family=binomial(link = "logit"))
    scores <- predict(moPropen, type = "response")
    scores <- pmin(pmax(scores, 0.05), 0.95)
    model <- causalTree(as.formula(paste("Y", paste(covs, collapse="+"), sep=" ~ ")), data = df, treatment = df[[paste("Dose", t, sep="")]], weights = 1 / scores, split.Rule = "CT", cv.option = "CT", split.Honest = T, cv.Honest = T, split.Bucket = T, xval = 5, cp = 0, minsize = 30, bucketMax = 20)
  }
  return(model)
}

modelFS <- gen_model(method = method, covs = covs, df = df, response = final_response, t = timesteps-1, solver_method = solver_method)

if (method == 'causaltree') {
  tau <- predict(modelFS, type = "vector")
  df["Y"] <- df["Y"] + ((as.numeric(tau > 0) - df[[paste("Dose", timesteps-1, sep="")]])*tau)
  policy <- data.frame(matrix(ncol = 0, nrow = 1000))
  policy[[paste("T", timesteps-1, sep="")]] <- as.numeric(tau > 0)
} else {
  policy <- data.frame(matrix(ncol = 0, nrow = 1000))
  policy[[paste("T", timesteps-1, sep="")]] <- optTx(modelFS)$optimalTx
}

if (timesteps > 1) {
  for (i in (timesteps-1):1) {
    if (grepl('bowl-', method, fixed = TRUE)) {
      this_response <- df[[paste("Reward", i, sep="")]]
    }

    covs <- covs[c(-length(covs))]
    covs <- covs[c(-length(covs))]
    if (i == (timesteps - 1)) {
      if (grepl('bowl-', method, fixed = TRUE)) {
        prev_m <- modelFS
      } else if (grepl('q-', method, fixed = TRUE) | grepl('optclass-', method, fixed = TRUE)) {
        this_response <- modelFS
        prev_m <- NULL
      } else {
        this_response <- NULL
        prev_m <- NULL
      }
    } else {
      if (grepl('bowl-', method, fixed = TRUE)) {
        prev_m <- modelPS
      } else if (grepl('q-', method, fixed = TRUE) | grepl('optclass-', method, fixed = TRUE)) {
        this_response <- modelPS
        prev_m <- NULL
      } else {
        this_response <- NULL
        prev_m <- NULL
      }
    }

    modelPS <- gen_model(method = method, covs = covs, df = df, response = this_response, t = i-1, prev_m = prev_m, solver_method = solver_method)
    if (method == 'causaltree') {
      tau <- predict(modelPS, type = "vector")
      df["Y"] <- df["Y"] + ((as.numeric(tau > 0) - df[[paste("Dose", i-1, sep="")]])*tau)
      policy[[paste("T", i-1, sep="")]] <- as.numeric(predict(modelPS, type = "vector") > 0)
    } else {
      policy[[paste("T", i-1, sep="")]] <- optTx(modelPS, df)$optimalTx
    }
  }
}


sim_step <- function(dose, conc, gamma, alpha, beta, ed50) {
  dose <- 50*dose
  conc <- (exp(-gamma)*conc) + dose
  burden <- max(min(100, beta * (1 - (conc ** alpha) / ((conc ** alpha) + (ed50 ** alpha)))), 0)
  return(list(conc, burden))
}


df_true <- read.csv(paste(data_filepath, "df_true.csv", sep="_"))
df_x <- df[ , grepl("X", names( df )) ]
max_timesteps <- max(df_true$timesteps)+1
all_burdens <- matrix(ncol = max_timesteps, nrow = 0)
all_concentrations <- matrix(ncol = max_timesteps, nrow = 0)

for (i in 1:nrow(df_true)) {
  this_row <- df_true[i,]
  this_x <- data.frame(df_x)[i,]
  this_policy <- policy[i,]
  beta <- this_row$beta
  alpha <- this_row$alpha
  ed50 <- this_row$ed50
  gamma <- this_row$gamma
  conc <- 0
  burdens <- c(this_row$init_burden)
  concentrations <- c(conc)
  binary_doses <- c()
  for (t in 0:(timesteps-1)) {
    if (timesteps == 1) {
      binary_doses <- c(binary_doses, this_policy)
      this_step <- sim_step(this_policy, conc, gamma, alpha, beta, ed50)
    } else {
      binary_doses <- c(binary_doses, this_policy[[paste("T", t, sep="")]])
      this_step <- sim_step(this_policy[[paste("T", t, sep="")]], conc, gamma, alpha, beta, ed50)
    }
    conc <- this_step[[1]]
    concentrations <- c(concentrations, conc)
    burdens <- c(burdens, this_step[[2]])
  }
  if (timesteps < this_row$timesteps) {
    for (t in (timesteps):(this_row$timesteps-1)) {
      this_df <- data.frame(this_x)
      this_df[["Burden0"]] <- burdens[length(burdens)-(timesteps-1)]
      k <- 0
      for (j in (timesteps-2):0) {
        this_df[[paste("Dose", k, sep="")]] <- binary_doses[length(binary_doses)-j]
        k <- k + 1
        this_df[[paste("Burden", k, sep="")]] <- burdens[length(burdens)-j]
      }
      if (method == 'causaltree') {
        this_dose <- as.numeric(predict(modelFS, this_df, type = "vector") > 0)
      } else {
        this_dose <- optTx(modelFS, this_df)$optimalTx[[1]]
      }

      binary_doses <- c(binary_doses, this_dose)
      this_step <- sim_step(this_dose, conc, gamma, alpha, beta, ed50)
      conc <- this_step[[1]]
      concentrations <- c(concentrations, conc)
      burdens <- c(burdens, this_step[[2]])
    }
  }
  length(burdens) <-  max_timesteps
  length(concentrations) <-  max_timesteps
  all_burdens <- rbind(all_burdens, c(burdens))
  all_concentrations <- rbind(all_concentrations, c(concentrations))
}

write.csv(all_burdens, paste(paste(data_filepath, method, sep="_"), "burdens.csv", sep="_"))
write.csv(all_concentrations, paste(paste(data_filepath, method, sep="_"), "concentrations.csv", sep="_"))
