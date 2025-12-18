# Usage: Rscript ts2net_case.R <case.json> <out_dir>
suppressMessages({
  library(jsonlite)
  library(ts2net)
  library(igraph)
})

args <- commandArgs(trailingOnly = TRUE)
case <- fromJSON(readLines(args[1]))
out_dir <- args[2]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

read_series <- function(p) {
  as.numeric(read.csv(p, header = FALSE)[,1])
}

write_graphml <- function(g, path) {
  igraph::write_graph(g, path, format = "graphml")
}

# Case fields: kind, path(s), params
if (case$kind == "HVG") {
  x <- read_series(case$series)
  g <- ts2net::HVG(x)
  write_graphml(g, file.path(out_dir, "graph.graphml"))
} else if (case$kind == "NVG") {
  x <- read_series(case$series)
  g <- ts2net::NVG(x)
  write_graphml(g, file.path(out_dir, "graph.graphml"))
} else if (case$kind == "RN") {
  x <- read_series(case$series)
  m <- case$params$m
  tau <- case$params$tau
  eps <- case$params$epsilon
  k <- case$params$k
  rule <- case$params$rule
  theiler <- ifelse(is.null(case$params$theiler), 0, case$params$theiler)
  metric <- ifelse(is.null(case$params$metric), "euclidean", case$params$metric)
  if (rule == "epsilon") {
    g <- ts2net::RN(x, m = m, tau = tau, eps = eps, theiler = theiler, metric = metric)
  } else {
    g <- ts2net::RN_kNN(x, m = m, tau = tau, k = k, theiler = theiler, metric = metric)
  }
  write_graphml(g, file.path(out_dir, "graph.graphml"))
} else if (case$kind == "TN") {
  x <- read_series(case$series)
  sym <- case$params$symbolizer
  if (sym == "ordinal") {
    ord <- case$params$order
    delay <- case$params$delay
    ties <- case$params$tie_rule
    g <- ts2net::TN_ordinal(x, order = ord, delay = delay, ties = ties)
  } else if (sym == "equal_width") {
    g <- ts2net::TN_bins(x, bins = case$params$bins, method = "equal_width")
  } else if (sym == "equal_freq") {
    g <- ts2net::TN_bins(x, bins = case$params$bins, method = "equal_freq")
  } else {
    stop("unknown symbolizer")
  }
  write_graphml(g, file.path(out_dir, "graph.graphml"))
} else if (case$kind == "DTW") {
  df <- read.csv(case$panel)
  nm <- if ("name" %in% names(df)) df$name else names(df)
  X <- if ("name" %in% names(df)) as.matrix(df[ , !(names(df) %in% "name")]) else t(as.matrix(df))
  # Save pairwise DTW as CSV
  D <- ts2net::cdist_dtw(X, band = case$params$band)
  write.csv(D, file.path(out_dir, "dtw.csv"), row.names = FALSE)
} else {
  stop("unknown case kind")
}

writeLines(toJSON(list(ok = TRUE), auto_unbox = TRUE), file.path(out_dir, "done.json"))
