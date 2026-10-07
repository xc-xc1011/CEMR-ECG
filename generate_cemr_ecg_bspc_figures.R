options(stringsAsFactors = FALSE)

local_lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library", "4.6")
if (dir.exists(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

required_packages <- c(
  "ggplot2", "patchwork", "svglite", "ragg", "dplyr", "tidyr",
  "readr", "scales", "ggrepel", "cowplot", "stringr", "purrr",
  "viridisLite", "jsonlite"
)
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop("Missing R packages: ", paste(missing_packages, collapse = ", "))
}

library(ggplot2)
library(patchwork)
library(dplyr)
library(tidyr)
library(readr)
library(scales)
library(ggrepel)
library(cowplot)
library(stringr)
library(purrr)
library(jsonlite)
library(grid)

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) {
    return(y)
  }
  x
}

ROOT <- getwd()
RESULTS <- file.path(ROOT, "results")
PRIMARY_DIR <- file.path(ROOT, "els-cas-templates", "figures")
BACKUP_DIR <- file.path(RESULTS, "cemr_ecg_bspc_figures")
SOURCE_DIR <- file.path(BACKUP_DIR, "source_data")
GENERATE_SCHEMATICS <- identical(tolower(Sys.getenv("CEMR_GENERATE_R_SCHEMATICS", "false")), "true")
dir.create(PRIMARY_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(BACKUP_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(SOURCE_DIR, showWarnings = FALSE, recursive = TRUE)

SEEDS <- c(303, 1303, 2303, 3303, 4303)
CLASSES <- c("N", "S", "V", "F", "Q")
CLASS4 <- c("N", "S", "V", "F")

palette_contract <- c(
  neutral_dark = "#272727",
  neutral_mid = "#767676",
  neutral_light = "#D8D8D8",
  baseline_dark = "#484878",
  baseline_mid = "#7884B4",
  baseline_soft = "#B4C0E4",
  evidence = "#33B5A5",
  cemr = "#0F4D92",
  accent_red = "#D24B40",
  accent_orange = "#E28E2C",
  delta_up = "#2E9E44",
  delta_down = "#B64342"
)

pc <- function(name) unname(palette_contract[[name]])

family_labels <- c(
  raw_machine_learning = "Machine learning",
  raw_deep_or_time_series = "Deep/time-series"
)

stage_labels <- c(
  raw = "Raw",
  evidence_argmax = "Encoder",
  `CEMR-BioAdaptive` = "Full"
)

theme_bspc <- function(base_size = 7, base_family = "Arial") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 0.5, colour = "black"),
      legend.title = element_text(size = base_size - 0.3),
      legend.text = element_text(size = base_size - 0.7),
      legend.key.size = unit(3.2, "mm"),
      legend.box.spacing = unit(1, "mm"),
      strip.background = element_blank(),
      strip.text = element_text(size = base_size - 0.2, face = "bold"),
      plot.title = element_text(size = base_size + 0.5, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = base_size - 0.4, colour = pc("neutral_mid")),
      plot.caption = element_text(size = base_size - 1.1, colour = pc("neutral_mid"), hjust = 0),
      panel.grid = element_blank()
    )
}
theme_set(theme_bspc())

save_pub_r <- function(plot, stem, width_mm = 183, height_mm = 120, dpi = 600) {
  base <- file.path(PRIMARY_DIR, stem)
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  unlink(file.path(PRIMARY_DIR, paste0(stem, c(".svg", ".pdf", ".tiff", ".png"))), force = TRUE)
  unlink(file.path(BACKUP_DIR, paste0(stem, c(".svg", ".pdf", ".tiff", ".png"))), force = TRUE)

  message("Writing ", stem, " SVG")
  svglite::svglite(paste0(base, ".svg"), width = w, height = h)
  print(plot)
  dev.off()
  gc()

  message("Writing ", stem, " PDF")
  grDevices::cairo_pdf(paste0(base, ".pdf"), width = w, height = h, family = "Arial")
  print(plot)
  dev.off()
  gc()

  message("Writing ", stem, " TIFF")
  ragg::agg_tiff(paste0(base, ".tiff"), width = w, height = h, units = "in", res = dpi)
  print(plot)
  dev.off()
  gc()

  message("Writing ", stem, " PNG")
  ragg::agg_png(paste0(base, ".png"), width = w, height = h, units = "in", res = 300)
  print(plot)
  dev.off()
  gc()

  for (ext in c(".svg", ".pdf", ".tiff", ".png")) {
    file.copy(paste0(base, ext), file.path(BACKUP_DIR, paste0(stem, ext)), overwrite = TRUE)
  }
}

write_source <- function(df, stem) {
  readr::write_csv(df, file.path(SOURCE_DIR, paste0(stem, "_source_data.csv")))
}

read_main_csv <- function(path) {
  readr::read_csv(file.path(RESULTS, path), show_col_types = FALSE)
}

safe_json <- function(x) {
  if (is.null(x) || length(x) == 0 || is.na(x) || !nzchar(as.character(x))) {
    return(NULL)
  }
  jsonlite::fromJSON(as.character(x), simplifyVector = FALSE)
}

count_json_to_long <- function(json_text, dataset, seed, split) {
  parsed <- safe_json(json_text)
  if (is.null(parsed)) {
    return(tibble())
  }
  tibble(
    dataset = dataset,
    seed = as.integer(seed),
    split = split,
    class = names(parsed),
    count = as.integer(unlist(parsed, use.names = FALSE))
  )
}

matrix_from_conf <- function(conf_list, key) {
  x <- conf_list[[key]]
  if (is.null(x)) {
    return(NULL)
  }
  matrix(as.integer(unlist(x)), nrow = 5, byrow = TRUE, dimnames = list(CLASSES, CLASSES))
}

sum_confusions <- function(conf_list, keys) {
  mats <- purrr::map(keys, ~ matrix_from_conf(conf_list, .x))
  mats <- mats[!vapply(mats, is.null, logical(1))]
  if (length(mats) == 0) {
    return(matrix(0L, nrow = 5, ncol = 5, dimnames = list(CLASSES, CLASSES)))
  }
  Reduce(`+`, mats)
}

conf_to_long <- function(mat, dataset, stage) {
  as.data.frame(as.table(mat), stringsAsFactors = FALSE) |>
    as_tibble() |>
    rename(true = Var1, pred = Var2, count = Freq) |>
    mutate(
      dataset = dataset,
      stage = stage,
      row_total = ave(count, true, FUN = sum),
      row_pct = if_else(row_total > 0, count / row_total, 0)
    )
}

metric_cols_to_long <- function(df, prefix = "") {
  metrics <- c("F1", "Se", "Pr")
  classes <- CLASS4
  purrr::map_dfr(metrics, function(metric) {
    purrr::map_dfr(classes, function(cls) {
      col <- paste0(prefix, metric, "_", cls, "_mean")
      if (!col %in% names(df)) {
        return(tibble())
      }
      tibble(metric = metric, class = cls, value = df[[col]])
    })
  })
}

multi_summary <- read_main_csv("datasetwise_multimethod_permethod_bioadaptive_summary.csv")
multi_detail <- read_main_csv("datasetwise_multimethod_permethod_bioadaptive_detail.csv")
outlier_summary <- read_main_csv("datasetwise_multimethod_permethod_bioadaptive_summary_no_outliers.csv")
removed_outliers <- read_main_csv("datasetwise_multimethod_permethod_bioadaptive_removed_outliers.csv")
bio_detail <- read_main_csv("cemr_bio_adaptive_decoder_detail.csv") |>
  filter(seed %in% SEEDS)
bio_summary <- read_main_csv("cemr_bio_adaptive_decoder_summary.csv")

if (nrow(multi_detail) != 300) {
  stop("datasetwise_multimethod_permethod_bioadaptive_detail.csv must have 300 rows; found ", nrow(multi_detail))
}
if (nrow(multi_summary) != 60) {
  stop("datasetwise_multimethod_permethod_bioadaptive_summary.csv must have 60 rows; found ", nrow(multi_summary))
}
if (dplyr::n_distinct(multi_summary$method) != 20) {
  stop("Summary must contain 20 methods; found ", dplyr::n_distinct(multi_summary$method))
}
seed_check <- multi_detail |>
  distinct(dataset, family, method, seed) |>
  count(dataset, family, method, name = "n_seeds") |>
  filter(n_seeds != 5)
if (nrow(seed_check) > 0) {
  stop("Not every dataset-method has five seeds.")
}

bio_conf <- jsonlite::fromJSON(file.path(RESULTS, "cemr_bio_adaptive_decoder_confusion.json"), simplifyVector = FALSE)
multi_conf <- jsonlite::fromJSON(file.path(RESULTS, "datasetwise_multimethod_permethod_bioadaptive_confusion.json"), simplifyVector = FALSE)

# Fig. 1 -----------------------------------------------------------------------
stage_band <- tibble(
  stage = c("Input", "Evidence", "Adapter", "Decoder"),
  xmin = c(0.20, 1.80, 4.80, 6.40),
  xmax = c(1.60, 4.60, 6.20, 8.40),
  fill = c("#EEF3FA", "#EAF6EC", "#FFF3E2", "#E8EEF8"),
  border = c("#7A9CC6", "#5DAB6F", "#E2A862", "#4D6BAB")
)

pipeline_nodes <- tibble(
  x = c(0.90, 2.40, 3.80, 5.50, 6.90, 8.10),
  y = 1,
  label = c(
    "ECG beat\nwindow",
    "Named ECG\nevidence",
    "Evidence\nencoder",
    "Backbone-evidence\nadapter (alpha)",
    "BioAdaptive\ndecoder",
    "Class-wise\noutput"
  )
)

pipeline_edges <- tibble(
  x = pipeline_nodes$x[-nrow(pipeline_nodes)] + 0.46,
  xend = pipeline_nodes$x[-1] - 0.46,
  y = 1,
  yend = 1
)

stage_caption <- tibble(
  x = c(0.90, 3.10, 5.50, 7.50),
  y = 0.40,
  label = c(
    "Two-lead, 234-sample window",
    "Morphology, RR, lead, shape, prototype",
    "alpha in {0.20, 0.35, 0.50, 0.65, 0.80}",
    "Bounded prevalence-aware multipliers"
  )
)

if (GENERATE_SCHEMATICS) {
  write_source(pipeline_nodes, "fig1_framework_nodes")
}

p1a <- ggplot() +
  geom_rect(
    data = stage_band,
    aes(xmin = xmin, xmax = xmax, ymin = 0.55, ymax = 1.45, fill = fill, colour = border),
    linewidth = 0.38,
    alpha = 0.65
  ) +
  scale_fill_identity() +
  scale_colour_identity() +
  geom_text(
    data = stage_band,
    aes(x = (xmin + xmax) / 2, y = 1.39, label = stage),
    fontface = "bold",
    size = 2.85,
    colour = pc("neutral_dark")
  ) +
  geom_segment(
    data = pipeline_edges,
    aes(x = x, xend = xend, y = y, yend = yend),
    arrow = arrow(length = unit(2.1, "mm"), type = "closed"),
    linewidth = 0.44,
    colour = pc("neutral_dark")
  ) +
  geom_label(
    data = pipeline_nodes,
    aes(x = x, y = y, label = label),
    fill = "white",
    colour = pc("neutral_dark"),
    label.r = unit(1.2, "mm"),
    label.padding = unit(1.6, "mm"),
    label.size = 0.32,
    size = 2.35,
    lineheight = 0.92
  ) +
  geom_text(
    data = stage_caption,
    aes(x = x, y = y, label = label),
    size = 2.05,
    colour = pc("neutral_mid"),
    lineheight = 0.94
  ) +
  coord_cartesian(xlim = c(0, 8.6), ylim = c(0.20, 1.55), clip = "off") +
  labs(title = "Reusable CEMR-ECG evidence flow") +
  theme_void(base_family = "Arial") +
  theme(plot.title = element_text(size = 9, face = "bold", hjust = 0.02))

module_df <- tibble(
  module = c("Morphology", "RR rhythm", "Lead-aware", "Shape deriv.", "Class prototype", "Training prior"),
  role = c(
    "beat shape & local slopes",
    "pre/post RR timing context",
    "between-lead consistency",
    "resampled waveform & slopes",
    "distance to class anchors",
    "prevalence-aware multipliers"
  ),
  group = c("Evidence", "Evidence", "Evidence", "Evidence", "Evidence", "Decoder"),
  x = c(0.65, 1.85, 3.05, 4.25, 5.45, 7.05),
  y = 1
)
if (GENERATE_SCHEMATICS) {
  write_source(module_df, "fig1_framework_modules")
}

p1b <- ggplot() +
  annotate("rect", xmin = 0.10, xmax = 6.10, ymin = 0.55, ymax = 1.55,
           fill = "#F0F7F1", colour = pc("delta_up"), linewidth = 0.38, alpha = 0.55) +
  annotate("rect", xmin = 6.40, xmax = 7.65, ymin = 0.55, ymax = 1.55,
           fill = "#EAF0F8", colour = pc("cemr"), linewidth = 0.38, alpha = 0.55) +
  annotate("text", x = 3.10, y = 1.49, label = "Five named evidence streams",
           fontface = "bold", size = 2.55, colour = pc("delta_up")) +
  annotate("text", x = 7.025, y = 1.49, label = "Decoder prior",
           fontface = "bold", size = 2.55, colour = pc("cemr")) +
  geom_label(
    data = module_df,
    aes(x = x, y = y, label = paste0(module, "\n", role)),
    fill = "white",
    colour = pc("neutral_dark"),
    label.r = unit(1.0, "mm"),
    label.padding = unit(1.4, "mm"),
    label.size = 0.30,
    size = 2.05,
    lineheight = 0.94
  ) +
  coord_cartesian(xlim = c(0, 7.85), ylim = c(0.40, 1.60), clip = "off") +
  labs(title = "Named, auditable evidence modules") +
  theme_void(base_family = "Arial") +
  theme(plot.title = element_text(size = 9, face = "bold", hjust = 0.02))

fig1 <- (p1a / p1b) +
  plot_layout(heights = c(1.05, 1.00)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold"))
if (GENERATE_SCHEMATICS) {
  save_pub_r(fig1, "fig1_cemr_ecg_framework", width_mm = 183, height_mm = 108)
} else {
  message("Skipping R export for fig1_cemr_ecg_framework; Python schematic is the manuscript source.")
}

# Graphical Abstract -----------------------------------------------------------
make_wedge <- function(cx, cy, r_outer, r_inner, theta_start, theta_end, n = 80) {
  theta_o <- seq(theta_start, theta_end, length.out = n)
  theta_i <- seq(theta_end, theta_start, length.out = n)
  tibble(
    x = c(cx + r_outer * cos(theta_o), cx + r_inner * cos(theta_i)),
    y = c(cy + r_outer * sin(theta_o), cy + r_inner * sin(theta_i))
  )
}

ring_cx <- 5.70
ring_cy <- 2.60
r_out <- 1.15
r_in <- 0.58
chip_radius <- 1.65

deg2rad <- function(d) d * pi / 180

wedge_meta <- tibble(
  name = c("Morphology", "RR rhythm", "Lead-aware", "Shape deriv.", "Class proto."),
  fill = c("#2E9E44", "#0F4D92", "#33B5A5", "#E28E2C", "#B64342"),
  start_deg = c(90, 162, 234, 306, 378),
  end_deg   = c(162, 234, 306, 378, 450),
  mid_deg   = c(126, 198, 270, 342, 54)
) |>
  mutate(
    start = deg2rad(start_deg),
    end = deg2rad(end_deg),
    mid = deg2rad(mid_deg),
    cx = ring_cx + chip_radius * cos(mid),
    cy = ring_cy + chip_radius * sin(mid)
  )

wedge_poly <- wedge_meta |>
  mutate(geom = pmap(list(start, end), function(s, e) make_wedge(ring_cx, ring_cy, r_out, r_in, s, e))) |>
  select(name, fill, geom) |>
  unnest(geom)

ecg_t <- seq(0, 3, length.out = 360)
ecg_amp <- function(t) {
  peaks <- c(0.55, 1.55, 2.55)
  val <- numeric(length(t))
  for (p in peaks) {
    val <- val + 1.45 * exp(-((t - p) ^ 2) / 0.0012)
    val <- val - 0.32 * exp(-((t - (p - 0.10)) ^ 2) / 0.0030)
    val <- val - 0.20 * exp(-((t - (p + 0.09)) ^ 2) / 0.0040)
    val <- val + 0.24 * exp(-((t - (p - 0.28)) ^ 2) / 0.013)
    val <- val + 0.30 * exp(-((t - (p + 0.30)) ^ 2) / 0.020)
  }
  val
}
ecg_raw_v <- ecg_amp(ecg_t)
ecg_v_norm <- (ecg_raw_v - mean(ecg_raw_v)) / max(abs(ecg_raw_v - mean(ecg_raw_v)))
ecg_df <- tibble(
  x = 0.45 + ecg_t * (2.20 / 3.0),
  y = ring_cy + 0.55 * ecg_v_norm
)

out_props <- c(N = 0.46, S = 0.20, V = 0.20, F = 0.14)
bar_xmin <- 8.95
bar_xmax <- 11.55
bar_ymin <- 2.20
bar_ymax <- 3.00
out_df <- tibble(
  class = factor(names(out_props), levels = c("N", "S", "V", "F")),
  prop = unname(out_props),
  fill = c("#C9C9D8", "#7A9CC6", "#33B5A5", "#E28E2C")
) |>
  mutate(
    xstart = bar_xmin + (bar_xmax - bar_xmin) * c(0, cumsum(prop)[-length(prop)]),
    xend = bar_xmin + (bar_xmax - bar_xmin) * cumsum(prop)
  )

ga <- ggplot() +
  annotate("text", x = 6.0, y = 4.62, label = "CEMR-ECG morphology–rhythm evidence",
           size = 3.4, fontface = "bold", colour = pc("neutral_dark")) +
  annotate("text", x = 6.0, y = 4.20,
           label = "Named, auditable evidence streams turn raw beats into AAMI class probabilities",
           size = 2.05, colour = pc("neutral_mid")) +
  annotate("rect", xmin = 0.30, xmax = 2.85, ymin = 1.50, ymax = 3.55,
           fill = "#F4F6FA", colour = pc("neutral_mid"), linewidth = 0.30) +
  annotate("text", x = 1.575, y = 3.36, label = "ECG beat window",
           size = 2.20, fontface = "bold", colour = pc("neutral_dark")) +
  geom_path(data = ecg_df, aes(x, y), linewidth = 0.55, colour = pc("cemr")) +
  annotate("text", x = 1.575, y = 1.70, label = "two-lead, 234 samples",
           size = 1.85, colour = pc("neutral_mid")) +
  geom_polygon(data = wedge_poly,
               aes(x = x, y = y, group = name, fill = fill),
               colour = "white", linewidth = 0.32, alpha = 0.92) +
  scale_fill_identity() +
  annotate("text", x = ring_cx, y = ring_cy + 0.12,
           label = "CEMR-ECG",
           size = 2.45, fontface = "bold", colour = pc("neutral_dark")) +
  annotate("text", x = ring_cx, y = ring_cy - 0.18,
           label = "evidence ring",
           size = 1.95, colour = pc("neutral_mid")) +
  geom_label(data = wedge_meta,
             aes(x = cx, y = cy, label = name),
             fill = "white", colour = pc("neutral_dark"),
             label.r = unit(1.0, "mm"),
             label.padding = unit(1.05, "mm"),
             label.size = 0.28,
             size = 2.00) +
  geom_rect(data = out_df,
            aes(xmin = xstart, xmax = xend, ymin = bar_ymin, ymax = bar_ymax, fill = fill),
            colour = "white", linewidth = 0.32) +
  geom_text(data = out_df,
            aes(x = (xstart + xend) / 2, y = (bar_ymin + bar_ymax) / 2, label = class),
            size = 2.10, fontface = "bold", colour = pc("neutral_dark")) +
  annotate("text", x = (bar_xmin + bar_xmax) / 2, y = bar_ymax + 0.30,
           label = "BioAdaptive class output", size = 2.20,
           fontface = "bold", colour = pc("neutral_dark")) +
  annotate("text", x = (bar_xmin + bar_xmax) / 2, y = bar_ymin - 0.25,
           label = "N / S / V / F probabilities",
           size = 1.85, colour = pc("neutral_mid")) +
  annotate("segment", x = 2.95, xend = ring_cx - r_out - 0.05,
           y = ring_cy, yend = ring_cy,
           arrow = arrow(length = unit(2.2, "mm"), type = "closed"),
           linewidth = 0.55, colour = pc("neutral_dark")) +
  annotate("segment", x = ring_cx + r_out + 0.05, xend = bar_xmin - 0.10,
           y = ring_cy, yend = (bar_ymin + bar_ymax) / 2,
           arrow = arrow(length = unit(2.2, "mm"), type = "closed"),
           linewidth = 0.55, colour = pc("neutral_dark")) +
  coord_fixed(xlim = c(0, 12.0), ylim = c(0.30, 4.95), clip = "off") +
  theme_void(base_family = "Arial") +
  theme(plot.margin = margin(2, 2, 2, 2))

if (GENERATE_SCHEMATICS) {
  write_source(wedge_meta |> select(name, start_deg, end_deg, mid_deg, fill, cx, cy),
               "fig_graphical_abstract_wedges")
  write_source(out_df |> select(class, prop, fill, xstart, xend),
               "fig_graphical_abstract_outputs")
}
if (GENERATE_SCHEMATICS) {
  save_pub_r(ga, "fig_graphical_abstract", width_mm = 180, height_mm = 75)
} else {
  message("Skipping R export for fig_graphical_abstract; Python schematic is the manuscript source.")
}

# Fig. 2 -----------------------------------------------------------------------
counts_df <- multi_detail |>
  filter(seed %in% SEEDS, family == "raw_machine_learning", method == "RbfSVM_raw") |>
  select(dataset, seed, train_counts, test_counts) |>
  distinct() |>
  pmap_dfr(function(dataset, seed, train_counts, test_counts) {
    bind_rows(
      count_json_to_long(train_counts, dataset, seed, "Train"),
      count_json_to_long(test_counts, dataset, seed, "Test")
    )
  }) |>
  mutate(class = factor(class, levels = CLASSES))
counts_mean <- counts_df |>
  group_by(dataset, split, class) |>
  summarise(count = mean(count), .groups = "drop") |>
  group_by(dataset, split) |>
  mutate(prop = count / sum(count)) |>
  ungroup()
minority_burden <- counts_mean |>
  filter(split == "Test") |>
  mutate(group = if_else(class == "N", "N", "Minority")) |>
  group_by(dataset, group) |>
  summarise(count = sum(count), prop = sum(prop), .groups = "drop")
write_source(counts_mean, "fig2_dataset_counts")

p2a <- ggplot(counts_mean |> filter(split == "Test"), aes(dataset, prop, fill = class)) +
  geom_col(width = 0.72, colour = "white", linewidth = 0.15) +
  scale_y_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.04))) +
  scale_fill_manual(values = c(N = "#C9C9D8", S = "#7A9CC6", V = "#33B5A5", F = "#E28E2C", Q = "#D24B40")) +
  labs(x = NULL, y = "Test-set class share", fill = "Class", title = "Test distribution") +
  theme(legend.position = "bottom")

p2b <- ggplot(counts_mean |> filter(split == "Test"), aes(class, count + 1, fill = class)) +
  geom_col(width = 0.7, colour = "white", linewidth = 0.15) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_y_log10(labels = label_number(accuracy = 1), expand = expansion(mult = c(0.02, 0.08))) +
  scale_fill_manual(values = c(N = "#C9C9D8", S = "#7A9CC6", V = "#33B5A5", F = "#E28E2C", Q = "#D24B40"), guide = "none") +
  labs(x = "AAMI class", y = "Beat count, log10 scale", title = "Rare-class sample support")

p2c <- ggplot(minority_burden, aes(dataset, prop, fill = group)) +
  geom_col(width = 0.72, colour = "white", linewidth = 0.15) +
  scale_y_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.04))) +
  scale_fill_manual(values = c(N = "#D8D8D8", Minority = pc("cemr"))) +
  labs(x = NULL, y = "Test-set share", fill = NULL, title = "Majority versus minority burden") +
  theme(legend.position = "bottom")

fig2 <- (p2a | p2b) / p2c +
  plot_layout(heights = c(1, 0.82), widths = c(0.85, 1.35), guides = "collect") +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8, face = "bold"))
save_pub_r(fig2, "fig2_dataset_support_context", width_mm = 183, height_mm = 128)

# Fig. 3 -----------------------------------------------------------------------
gain_df <- multi_summary |>
  mutate(
    family_label = recode(family, !!!family_labels),
    method_short = str_replace_all(method, "_raw|raw_", ""),
    raw_mf1 = raw_macro_f1_4_mean * 100,
    cemr_mf1 = macro_f1_4_mean * 100,
    delta_mf1 = delta_macro_f1_4_mean * 100,
    delta_acc = delta_accuracy_mean * 100,
    delta_s = delta_F1_S_mean * 100,
    delta_f = delta_F1_F_mean * 100
  )
write_source(gain_df, "fig3_20method_framework_gains")

gain_long <- gain_df |>
  select(dataset, family_label, method_short, raw_mf1, cemr_mf1) |>
  pivot_longer(c(raw_mf1, cemr_mf1), names_to = "stage", values_to = "mf1") |>
  mutate(stage = recode(stage, raw_mf1 = "Raw", cemr_mf1 = "+ CEMR-ECG"),
         stage = factor(stage, levels = c("Raw", "+ CEMR-ECG")))

p3a <- ggplot(gain_long, aes(stage, mf1, group = method_short, colour = family_label)) +
  geom_line(alpha = 0.46, linewidth = 0.35) +
  geom_point(size = 1.25, alpha = 0.85) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_colour_manual(values = c("Machine learning" = pc("baseline_dark"), "Deep/time-series" = pc("baseline_mid"))) +
  labs(x = NULL, y = "M-F1(4), %", colour = "Backbone family", title = "Paired gain across 20 backbones") +
  theme(legend.position = "bottom")

p3b <- ggplot(gain_df, aes(dataset, delta_mf1, fill = family_label)) +
  geom_hline(yintercept = 0, linewidth = 0.25, colour = pc("neutral_mid")) +
  geom_boxplot(width = 0.55, outlier.shape = NA, alpha = 0.82, linewidth = 0.28, position = position_dodge(width = 0.65)) +
  geom_point(position = position_jitterdodge(jitter.width = 0.08, dodge.width = 0.65), size = 0.85, alpha = 0.65) +
  scale_fill_manual(values = c("Machine learning" = pc("baseline_dark"), "Deep/time-series" = pc("baseline_mid"))) +
  labs(x = NULL, y = "Delta M-F1(4), pp", fill = "Backbone family", title = "Distribution of gains") +
  theme(legend.position = "bottom")

class_gain <- gain_df |>
  select(dataset, family_label, method_short, delta_s, delta_f) |>
  pivot_longer(c(delta_s, delta_f), names_to = "class", values_to = "delta") |>
  mutate(class = recode(class, delta_s = "S", delta_f = "F"))
p3c <- ggplot(class_gain, aes(class, delta, fill = class)) +
  geom_hline(yintercept = 0, linewidth = 0.25, colour = pc("neutral_mid")) +
  geom_boxplot(width = 0.62, alpha = 0.88, outlier.shape = NA, linewidth = 0.28) +
  geom_point(position = position_jitter(width = 0.08), size = 0.75, alpha = 0.55) +
  facet_grid(family_label ~ dataset) +
  scale_fill_manual(values = c(S = "#7A9CC6", F = "#E28E2C"), guide = "none") +
  labs(x = "Minority class", y = "Delta class F1, pp", title = "Minority-class contribution")

best_labels <- gain_df |>
  group_by(dataset) |>
  slice_max(cemr_mf1, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(label = paste0(method_short, "\n", sprintf("%.1f%%", cemr_mf1)))
p3d <- ggplot(gain_df, aes(delta_acc, delta_mf1, colour = family_label)) +
  geom_hline(yintercept = 0, linewidth = 0.24, colour = pc("neutral_mid")) +
  geom_vline(xintercept = 0, linewidth = 0.24, colour = pc("neutral_mid")) +
  geom_point(size = 1.35, alpha = 0.72) +
  geom_text_repel(data = best_labels, aes(label = label), size = 1.9, min.segment.length = 0, segment.size = 0.18, seed = 4) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_colour_manual(values = c("Machine learning" = pc("baseline_dark"), "Deep/time-series" = pc("baseline_mid"))) +
  labs(x = "Delta accuracy, pp", y = "Delta M-F1(4), pp", colour = "Backbone family", title = "Macro-F1 gain versus accuracy gain") +
  theme(legend.position = "bottom")

fig3 <- (p3a / (p3b | p3c) / p3d) +
  plot_layout(heights = c(0.9, 1.10, 0.95), guides = "collect") +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8, face = "bold"), legend.position = "bottom")
save_pub_r(fig3, "fig3_20method_framework_gains", width_mm = 183, height_mm = 185)

# Fig. 4 -----------------------------------------------------------------------
raw_class_cols <- names(multi_summary)[grepl("^raw_(F1|Se|Pr)_[NSVFQ]_mean$", names(multi_summary))]
cemr_class_cols <- names(multi_summary)[grepl("^(F1|Se|Pr)_[NSVFQ]_mean$", names(multi_summary))]
raw_class <- multi_summary |>
  group_by(dataset) |>
  summarise(across(all_of(raw_class_cols), \(x) mean(x, na.rm = TRUE)), .groups = "drop")
cemr_class <- multi_summary |>
  group_by(dataset) |>
  summarise(across(all_of(cemr_class_cols), \(x) mean(x, na.rm = TRUE)), .groups = "drop")

class_mech <- purrr::map_dfr(c("F1", "Se", "Pr"), function(metric) {
  purrr::map_dfr(CLASS4, function(cls) {
    met <- metric
    cl <- cls
    tibble(
      dataset = raw_class$dataset,
      class = cl,
      metric = met,
      raw = raw_class[[paste0("raw_", met, "_", cl, "_mean")]] * 100,
      cemr = cemr_class[[paste0(met, "_", cl, "_mean")]] * 100
    )
  })
}) |>
  mutate(delta = cemr - raw)
write_source(class_mech, "fig4_classwise_pattern")

p4a <- ggplot(class_mech |> filter(metric == "F1"), aes(class, raw, xend = class, yend = cemr)) +
  geom_segment(linewidth = 0.55, colour = pc("neutral_mid")) +
  geom_point(aes(y = raw), size = 1.7, colour = pc("baseline_mid")) +
  geom_point(aes(y = cemr), size = 1.9, colour = pc("cemr")) +
  facet_wrap(~ dataset, nrow = 1) +
  labs(x = "Class", y = "Class F1, %", title = "Class-wise F1 shifts")

p4b <- ggplot(class_mech |> filter(metric %in% c("Se", "Pr"), class %in% c("S", "F")), aes(metric, delta, fill = class)) +
  geom_hline(yintercept = 0, linewidth = 0.25, colour = pc("neutral_mid")) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6, colour = "white", linewidth = 0.15) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_fill_manual(values = c(S = "#7A9CC6", F = "#E28E2C")) +
  labs(x = NULL, y = "Delta, pp", fill = "Class", title = "Sensitivity-precision mechanism") +
  theme(legend.position = "bottom")

p4c <- ggplot(class_mech |> filter(metric == "F1"), aes(raw, cemr, colour = class, label = class)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.28, colour = pc("neutral_mid")) +
  geom_point(size = 2.1) +
  geom_text_repel(size = 2.1, min.segment.length = 0, segment.size = 0.2, seed = 2) +
  facet_wrap(~ dataset, nrow = 1) +
  coord_equal(xlim = c(0, 100), ylim = c(0, 100)) +
  scale_colour_manual(values = c(N = "#777777", S = "#7A9CC6", V = "#33B5A5", F = "#E28E2C")) +
  labs(x = "Raw class F1, %", y = "+ CEMR-ECG class F1, %", colour = "Class", title = "Minority gains are not majority accuracy only") +
  theme(legend.position = "bottom")

fig4 <- (p4a / p4b / p4c) +
  plot_layout(heights = c(0.95, 0.85, 1.0), guides = "collect") +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8, face = "bold"))
save_pub_r(fig4, "fig4_classwise_pattern", width_mm = 183, height_mm = 165)

# Fig. 5 -----------------------------------------------------------------------
decoder_label_map <- c(
  evidence_argmax = "Evidence",
  clinical_default = "Clinical",
  stable_prior = "Stable prior",
  mild_prior = "Mild prior",
  s_recovery = "S recovery",
  f_recovery = "F recovery"
)

decoder_rows <- multi_detail |>
  filter(seed %in% SEEDS) |>
  mutate(
    decoder_label = recode(selected_decoder, !!!decoder_label_map, .default = selected_decoder),
    multiplier = map(selected_multiplier, safe_json),
    cands = map(validation_table, safe_json)
  )

decoder_mult <- decoder_rows |>
  select(dataset, family, method, seed, selected_decoder, decoder_label, selected_alpha, multiplier) |>
  unnest_longer(multiplier, indices_to = "class_index") |>
  mutate(class = CLASSES[class_index], multiplier = as.numeric(multiplier))

decoder_train <- decoder_rows |>
  select(dataset, family, method, seed, train_counts) |>
  distinct() |>
  mutate(items = map(train_counts, safe_json)) |>
  mutate(items = map(items, ~ tibble(class = names(.x), count = as.integer(unlist(.x))))) |>
  select(-train_counts) |>
  unnest(items) |>
  group_by(dataset, family, method, seed) |>
  mutate(prior = count / sum(count)) |>
  ungroup()

candidate_rows <- decoder_rows |>
  select(dataset, family, method, seed, cands) |>
  mutate(cands = map(cands, function(x) {
    if (is.null(x)) return(tibble())
    map_dfr(x, function(z) {
      tibble(
        alpha = as.numeric(z$alpha %||% NA_real_),
        decoder = z$decoder %||% NA_character_,
        val_macro_f1_4 = as.numeric(z$val_macro_f1_4 %||% NA_real_),
        val_accuracy = as.numeric(z$val_accuracy %||% NA_real_),
        val_F1_S = as.numeric(z$val_F1_S %||% NA_real_),
        val_F1_F = as.numeric(z$val_F1_F %||% NA_real_)
      )
    })
  })) |>
  unnest(cands) |>
  mutate(decoder_label = recode(decoder, !!!decoder_label_map, .default = decoder))

prior_summary <- decoder_train |>
  group_by(dataset, class) |>
  summarise(prior = mean(prior, na.rm = TRUE), .groups = "drop")
mult_summary <- decoder_mult |>
  filter(class %in% CLASS4) |>
  group_by(dataset, decoder_label, class) |>
  summarise(multiplier = mean(multiplier, na.rm = TRUE), .groups = "drop")
candidate_summary <- candidate_rows |>
  group_by(dataset, decoder_label) |>
  summarise(
    val_macro_f1_4_mean = mean(val_macro_f1_4, na.rm = TRUE) * 100,
    val_macro_f1_4_sd = sd(val_macro_f1_4, na.rm = TRUE) * 100,
    .groups = "drop"
  ) |>
  filter(is.finite(val_macro_f1_4_mean))
decoder_choice <- decoder_rows |>
  count(dataset, decoder_label, name = "n")
alpha_choice <- decoder_rows |>
  count(dataset, selected_alpha, name = "n")

write_source(decoder_mult, "fig5_decoder_multipliers")
write_source(prior_summary, "fig5_decoder_train_priors")
write_source(candidate_rows, "fig5_decoder_validation_candidates")
write_source(decoder_choice, "fig5_decoder_choice_counts")
write_source(alpha_choice, "fig5_alpha_choice_counts")

p5a <- ggplot(prior_summary, aes(class, prior, fill = class)) +
  geom_col(width = 0.66, colour = "white", linewidth = 0.15) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_y_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.05))) +
  scale_fill_manual(values = c(N = "#C9C9D8", S = "#7A9CC6", V = "#33B5A5", F = "#E28E2C", Q = "#D24B40"), guide = "none") +
  labs(x = "Training class", y = "Training prior", title = "Training priors across method-specific runs")

p5b <- ggplot(mult_summary, aes(class, multiplier, fill = class)) +
  geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.28, colour = pc("neutral_mid")) +
  geom_boxplot(width = 0.58, outlier.shape = NA, linewidth = 0.25, alpha = 0.82) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_fill_manual(values = c(N = "#C9C9D8", S = "#7A9CC6", V = "#33B5A5", F = "#E28E2C"), guide = "none") +
  labs(x = "Decision class", y = "Decoder multiplier", title = "Bounded class-wise decoder weights")

p5c <- ggplot(candidate_summary, aes(reorder(decoder_label, val_macro_f1_4_mean), val_macro_f1_4_mean, fill = decoder_label)) +
  geom_col(width = 0.7, colour = "white", linewidth = 0.12) +
  geom_errorbar(aes(ymin = val_macro_f1_4_mean - val_macro_f1_4_sd, ymax = val_macro_f1_4_mean + val_macro_f1_4_sd),
                width = 0.18, linewidth = 0.22, colour = pc("neutral_dark")) +
  coord_flip() +
  facet_wrap(~ dataset, nrow = 1) +
  scale_y_continuous(limits = c(0, 100), expand = expansion(mult = c(0, 0.03))) +
  scale_fill_manual(values = rep(c("#B4C0E4", "#7884B4", "#33B5A5", "#E4CCD8", "#E28E2C", "#D24B40"), 3), guide = "none") +
  labs(x = "Candidate decoder", y = "Validation M-F1(4), %", title = "Validation-ranked candidate family")

p5d <- ggplot(decoder_choice, aes(dataset, n, fill = decoder_label)) +
  geom_col(width = 0.65, colour = "white", linewidth = 0.15) +
  labs(x = NULL, y = "Selections across 20 methods and 5 seeds", fill = "Decoder", title = "Selected BioAdaptive policy") +
  theme(legend.position = "bottom")

p5e <- ggplot(alpha_choice, aes(factor(selected_alpha), n, fill = factor(selected_alpha))) +
  geom_col(width = 0.62, colour = "white", linewidth = 0.15) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_fill_manual(values = c("0.2" = "#D8D8D8", "0.35" = "#B4C0E4", "0.5" = "#7884B4", "0.65" = "#33B5A5", "0.8" = "#0F4D92"), guide = "none") +
  labs(x = "Selected alpha", y = "Selections", title = "Adapter keeps backbone contribution")

fig5 <- (p5a | p5b) / (p5c | (p5d / p5e)) +
  plot_layout(heights = c(0.95, 1.20), widths = c(1.08, 0.92), guides = "collect") +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8, face = "bold"), legend.position = "bottom")
save_pub_r(fig5, "fig5_bioadaptive_decoder_behavior", width_mm = 183, height_mm = 175)

# Fig. 6 -----------------------------------------------------------------------
stage_df <- bio_detail |>
  filter(stage %in% names(stage_labels), seed %in% SEEDS) |>
  mutate(
    stage_label = recode(stage, !!!stage_labels),
    stage_label = factor(stage_label, levels = unname(stage_labels)),
    macro_f1_4_pct = macro_f1_4 * 100
  )
write_source(stage_df, "fig6_stage_ablation")

stage_conf <- purrr::map_dfr(unique(stage_df$dataset), function(ds) {
  purrr::map_dfr(names(stage_labels), function(st) {
    keys <- paste(ds, SEEDS, st, sep = "|")
    mat <- sum_confusions(bio_conf, keys)
    conf_to_long(mat, ds, stage_labels[[st]])
  })
})
error_focus <- stage_conf |>
  filter(true %in% c("S", "F"), pred == "N") |>
  group_by(dataset, stage, true) |>
  summarise(error_rate = sum(count) / unique(row_total), .groups = "drop")
write_source(error_focus, "fig6_stage_error_focus")

tradeoff_stage <- stage_df |>
  select(dataset, seed, stage_label, Se_S, Pr_S, Se_F, Pr_F) |>
  pivot_longer(c(Se_S, Pr_S, Se_F, Pr_F), names_to = "name", values_to = "value") |>
  separate(name, into = c("metric", "class"), sep = "_") |>
  group_by(dataset, stage_label, metric, class) |>
  summarise(value = mean(value) * 100, .groups = "drop") |>
  pivot_wider(names_from = metric, values_from = value)
write_source(tradeoff_stage, "fig6_stage_tradeoff")

p6a <- ggplot(stage_df, aes(stage_label, macro_f1_4_pct, fill = stage_label)) +
  geom_boxplot(width = 0.56, outlier.shape = NA, alpha = 0.88, linewidth = 0.28) +
  geom_point(position = position_jitter(width = 0.07), size = 0.8, alpha = 0.65) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_fill_manual(values = c("Raw" = "#B4C0E4", "Encoder" = "#33B5A5", "Full" = "#0F4D92"), guide = "none") +
  labs(x = NULL, y = "M-F1(4), %", title = "Stage and component ablation") +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))

p6b <- ggplot(tradeoff_stage |> filter(class %in% c("S", "F")), aes(Pr, Se, colour = stage_label, shape = class)) +
  geom_path(aes(group = interaction(dataset, class)), linewidth = 0.25, colour = pc("neutral_mid"), alpha = 0.55) +
  geom_point(size = 2.0, alpha = 0.95) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_colour_manual(values = c("Raw" = "#B4C0E4", "Encoder" = "#33B5A5", "Full" = "#0F4D92")) +
  labs(x = "Precision, %", y = "Sensitivity, %", colour = "Stage", shape = "Class", title = "S/F recall-precision tradeoff") +
  theme(legend.position = "none")

p6c <- ggplot(error_focus, aes(stage, error_rate * 100, fill = true)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.62, colour = "white", linewidth = 0.12) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_fill_manual(values = c(S = "#7A9CC6", F = "#E28E2C")) +
  labs(x = NULL, y = "To N error, %", fill = "True class", title = "Dominant minority-to-normal errors") +
  theme(axis.text.x = element_text(angle = 15, hjust = 1), legend.position = "bottom")

p6d_data <- stage_df |>
  group_by(dataset, stage_label) |>
  summarise(
    F1_S = mean(F1_S) * 100,
    F1_F = mean(F1_F) * 100,
    F1_V = mean(F1_V) * 100,
    .groups = "drop"
  ) |>
  pivot_longer(starts_with("F1_"), names_to = "class", values_to = "F1") |>
  mutate(class = str_remove(class, "F1_"))
p6d <- ggplot(p6d_data, aes(stage_label, F1, fill = class)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.62, colour = "white", linewidth = 0.12) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_fill_manual(values = c(S = "#7A9CC6", V = "#33B5A5", F = "#E28E2C")) +
  labs(x = NULL, y = "Class F1, %", fill = "Class", title = "Which class each stage helps") +
  theme(axis.text.x = element_text(angle = 15, hjust = 1), legend.position = "bottom")

fig6 <- (p6a / p6b) / (p6c / p6d) +
  plot_layout(heights = c(1.0, 1.0, 1.05), guides = "collect") +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8, face = "bold"), legend.position = "bottom")
save_pub_r(fig6, "fig6_stage_component_ablation", width_mm = 183, height_mm = 210)

# Fig. 7 -----------------------------------------------------------------------
best_raw <- multi_summary |>
  group_by(dataset) |>
  slice_max(raw_macro_f1_4_mean, n = 1, with_ties = FALSE) |>
  ungroup()

conf_cemr <- purrr::map_dfr(unique(best_raw$dataset), function(ds) {
  method_row <- best_raw |> filter(dataset == ds) |> slice(1)
  fam <- method_row$family
  method <- method_row$method
  raw_keys <- paste(ds, fam, method, SEEDS, "raw", sep = "|")
  cemr_keys <- paste(ds, fam, method, SEEDS, "CEMR-BioAdaptive", sep = "|")
  bind_rows(
    conf_to_long(sum_confusions(multi_conf, raw_keys), ds, "Strongest raw baseline"),
    conf_to_long(sum_confusions(multi_conf, cemr_keys), ds, "CEMR-ECG")
  )
})
write_source(conf_cemr, "fig7_confusion_source")

key_errors <- conf_cemr |>
  filter(true %in% c("S", "F"), pred == "N") |>
  group_by(dataset, stage, true) |>
  summarise(error_rate = sum(count) / unique(row_total), .groups = "drop")
write_source(key_errors, "fig7_key_error_rates")

p7a <- ggplot(conf_cemr |> filter(stage == "CEMR-ECG"), aes(pred, true, fill = row_pct)) +
  geom_tile(colour = "white", linewidth = 0.18) +
  geom_text(aes(label = if_else(row_pct >= 0.005, sprintf("%.0f", row_pct * 100), "")), size = 1.9) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_fill_gradient(low = "#F3F6FA", high = pc("cemr"), labels = percent_format(accuracy = 1)) +
  coord_fixed() +
  labs(x = "Predicted class", y = "True class", fill = "Row %", title = "CEMR-ECG row-normalized confusion matrices")

p7b <- ggplot(key_errors, aes(stage, error_rate * 100, fill = stage)) +
  geom_col(width = 0.6, colour = "white", linewidth = 0.12) +
  facet_grid(true ~ dataset) +
  scale_fill_manual(values = c("Strongest raw baseline" = pc("baseline_mid"), "CEMR-ECG" = pc("cemr")), guide = "none") +
  labs(x = NULL, y = "S/F to N error rate, %", title = "Minority-to-normal error reduction") +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

fig7 <- (p7a / p7b) +
  plot_layout(heights = c(1.2, 0.9)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8, face = "bold"))
save_pub_r(fig7, "fig7_confusion_error_anatomy", width_mm = 183, height_mm = 138)

# Fig. 8 -----------------------------------------------------------------------
stable_df <- multi_detail |>
  mutate(
    family_label = recode(family, !!!family_labels),
    macro_f1_4_pct = macro_f1_4 * 100,
    delta_mf1_pct = delta_macro_f1_4 * 100,
    train_s = train_time,
    predict_s = predict_time,
    evidence_train_s = evidence_train_time,
    backbone_train_s = backbone_train_time
  )
write_source(stable_df, "fig8_stability_runtime")

p8a <- ggplot(stable_df, aes(dataset, macro_f1_4_pct, fill = family_label)) +
  geom_boxplot(width = 0.56, outlier.shape = NA, linewidth = 0.28, alpha = 0.82, position = position_dodge(width = 0.72)) +
  scale_fill_manual(values = c("Machine learning" = pc("baseline_dark"), "Deep/time-series" = pc("baseline_mid"))) +
  labs(x = NULL, y = "CEMR-ECG M-F1(4), %", fill = "Backbone family", title = "Five-seed stability across all methods") +
  theme(legend.position = "bottom")

runtime_df <- stable_df |>
  select(dataset, family_label, method, seed, backbone_train_s, evidence_train_s, predict_s) |>
  pivot_longer(c(backbone_train_s, evidence_train_s, predict_s), names_to = "phase", values_to = "seconds") |>
  mutate(phase = recode(phase, backbone_train_s = "Backbone training", evidence_train_s = "Evidence training", predict_s = "Prediction"))
p8b <- ggplot(runtime_df, aes(phase, seconds, fill = phase)) +
  geom_boxplot(width = 0.56, outlier.shape = NA, linewidth = 0.28, alpha = 0.86) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_y_log10(labels = label_number(accuracy = 0.1)) +
  scale_fill_manual(values = c("Backbone training" = pc("baseline_mid"), "Evidence training" = pc("evidence"), Prediction = pc("cemr")), guide = "none") +
  labs(x = NULL, y = "Time, seconds, log10", title = "Training and prediction cost") +
  theme(axis.text.x = element_text(angle = 18, hjust = 1))

p8c <- ggplot(stable_df, aes(dataset, delta_mf1_pct, fill = family_label)) +
  geom_hline(yintercept = 0, linewidth = 0.25, colour = pc("neutral_mid")) +
  geom_boxplot(width = 0.58, outlier.shape = NA, linewidth = 0.28, alpha = 0.86, position = position_dodge(width = 0.72)) +
  scale_fill_manual(values = c("Machine learning" = pc("baseline_dark"), "Deep/time-series" = pc("baseline_mid"))) +
  labs(x = NULL, y = "Delta M-F1(4), pp", fill = "Backbone family", title = "Seed-level gains across all methods") +
  theme(legend.position = "bottom")

std_df <- multi_summary |>
  mutate(family_label = recode(family, !!!family_labels), mf1_std = macro_f1_4_std * 100) |>
  group_by(dataset, family_label) |>
  summarise(median_std = median(mf1_std, na.rm = TRUE), .groups = "drop")
p8d <- ggplot(std_df, aes(dataset, median_std, fill = family_label)) +
  geom_col(position = position_dodge(width = 0.68), width = 0.58, colour = "white", linewidth = 0.12) +
  scale_fill_manual(values = c("Machine learning" = pc("baseline_dark"), "Deep/time-series" = pc("baseline_mid"))) +
  labs(x = NULL, y = "Median seed SD, pp", fill = "Backbone family", title = "Variability audit") +
  theme(legend.position = "bottom")

fig8 <- (p8a | p8b) / (p8c | p8d) +
  plot_layout(guides = "collect") +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8, face = "bold"), legend.position = "bottom")
save_pub_r(fig8, "fig8_stability_practicality", width_mm = 183, height_mm = 138)

# Fig. 9 -----------------------------------------------------------------------
robust_df <- multi_summary |>
  select(dataset, family, method, full_mf1 = macro_f1_4_mean, full_acc = accuracy_mean, full_f1s = F1_S_mean, full_f1f = F1_F_mean) |>
  left_join(
    outlier_summary |>
      select(dataset, family, method, n_seeds_before, n_seeds_after, removed_seeds,
             no_outlier_mf1 = macro_f1_4_mean, no_outlier_acc = accuracy_mean,
             no_outlier_f1s = F1_S_mean, no_outlier_f1f = F1_F_mean),
    by = c("dataset", "family", "method")
  ) |>
  mutate(
    family_label = recode(family, !!!family_labels),
    delta_no_outlier_mf1 = (no_outlier_mf1 - full_mf1) * 100,
    delta_no_outlier_acc = (no_outlier_acc - full_acc) * 100,
    delta_no_outlier_f1s = (no_outlier_f1s - full_f1s) * 100,
    delta_no_outlier_f1f = (no_outlier_f1f - full_f1f) * 100,
    full_mf1_pct = full_mf1 * 100,
    no_outlier_mf1_pct = no_outlier_mf1 * 100,
    removed_count = n_seeds_before - n_seeds_after
  )
write_source(robust_df, "fig9_outlier_robustness")

p9a <- ggplot(robust_df, aes(full_mf1_pct, no_outlier_mf1_pct, colour = family_label)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.28, colour = pc("neutral_mid")) +
  geom_point(size = 1.4, alpha = 0.72) +
  facet_wrap(~ dataset, nrow = 1) +
  coord_equal(xlim = c(35, 75), ylim = c(35, 75)) +
  scale_colour_manual(values = c("Machine learning" = pc("baseline_dark"), "Deep/time-series" = pc("baseline_mid"))) +
  labs(x = "Full five-seed M-F1(4), %", y = "IQR-filtered M-F1(4), %", colour = "Backbone family", title = "Full-result conclusion is stable to IQR filtering") +
  theme(legend.position = "bottom")

p9b <- ggplot(robust_df, aes(dataset, delta_no_outlier_mf1, fill = family_label)) +
  geom_hline(yintercept = 0, linewidth = 0.25, colour = pc("neutral_mid")) +
  geom_boxplot(width = 0.56, outlier.shape = NA, linewidth = 0.28, alpha = 0.84, position = position_dodge(width = 0.72)) +
  scale_fill_manual(values = c("Machine learning" = pc("baseline_dark"), "Deep/time-series" = pc("baseline_mid"))) +
  labs(x = NULL, y = "Filtered minus full M-F1(4), pp", fill = "Backbone family", title = "Magnitude of robustness adjustment") +
  theme(legend.position = "bottom")

removed_counts <- robust_df |>
  count(dataset, removed_count, name = "groups")
p9c <- ggplot(removed_counts, aes(factor(removed_count), groups, fill = factor(removed_count))) +
  geom_col(width = 0.62, colour = "white", linewidth = 0.12) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_fill_manual(values = c("0" = "#D8D8D8", "1" = "#B4C0E4", "2" = "#7884B4"), guide = "none") +
  labs(x = "Removed seeds per group", y = "Dataset-method groups", title = "IQR removal is limited")

fig9 <- (p9a / (p9b | p9c)) +
  plot_layout(heights = c(1.05, 0.95), guides = "collect") +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8, face = "bold"), legend.position = "bottom")
save_pub_r(fig9, "fig9_outlier_robustness", width_mm = 183, height_mm = 130)

manifest <- tibble(
  figure = c("Graphical abstract", paste0("Fig. ", 1:9)),
  file_stem = c(
    "fig_graphical_abstract",
    "fig1_cemr_ecg_framework",
    "fig2_dataset_support_context",
    "fig3_20method_framework_gains",
    "fig4_classwise_pattern",
    "fig5_bioadaptive_decoder_behavior",
    "fig6_stage_component_ablation",
    "fig7_confusion_error_anatomy",
    "fig8_stability_practicality",
    "fig9_outlier_robustness"
  ),
  role = c(
    "Evidence funnel summary used as the manuscript graphical abstract",
    "CEMR-ECG evidence flow and interpretable module map",
    "Dataset imbalance and minority-class motivation",
    "Full 20-method framework gains across families and datasets",
    "Class-wise pattern of minority recovery",
    "BioAdaptive decoder priors, multipliers and validation behaviour",
    "Stage and component ablation and error pathways",
    "Confusion matrices and S/F to N error anatomy",
    "Seed stability and practical runtime",
    "IQR outlier-removal robustness check"
  ),
  source = c(
    "Python/matplotlib schematic; source_data/fig_graphical_abstract_python_source_data.csv",
    "Python/matplotlib schematic; source_data/fig1_python_framework_source_data.csv",
    "datasetwise_multimethod_permethod_bioadaptive_detail.csv",
    "datasetwise_multimethod_permethod_bioadaptive_summary.csv",
    "datasetwise_multimethod_permethod_bioadaptive_summary.csv",
    "datasetwise_multimethod_permethod_bioadaptive_detail.csv",
    "cemr_bio_adaptive_decoder_detail.csv; cemr_bio_adaptive_decoder_confusion.json",
    "datasetwise_multimethod_permethod_bioadaptive_confusion.json",
    "datasetwise_multimethod_permethod_bioadaptive_detail.csv; datasetwise_multimethod_permethod_bioadaptive_summary.csv",
    "datasetwise_multimethod_permethod_bioadaptive_summary.csv; datasetwise_multimethod_permethod_bioadaptive_summary_no_outliers.csv"
  )
)
readr::write_csv(manifest, file.path(BACKUP_DIR, "figure_manifest.csv"))
readr::write_csv(manifest, file.path(PRIMARY_DIR, "figure_manifest.csv"))
readr::write_lines(
  c(
    "# CEMR-ECG BSPC Figure Manifest",
    "",
    "Fig. 1 and the graphical abstract were generated in Python/matplotlib; Fig. 2--9 were generated in R/ggplot2. All figures include editable SVG/PDF, PNG previews and 600 dpi TIFF exports.",
    "Main-result figures use the corrected 20-method, three-dataset, five-seed per-method CEMR-ECG tables.",
    "",
    "| Figure | File stem | Role | Source |",
    "| --- | --- | --- | --- |",
    apply(manifest, 1, function(x) paste0("| ", paste(x, collapse = " | "), " |"))
  ),
  file.path(BACKUP_DIR, "figure_manifest.md")
)
file.copy(file.path(BACKUP_DIR, "figure_manifest.md"), file.path(PRIMARY_DIR, "figure_manifest.md"), overwrite = TRUE)

message("Generated CEMR-ECG BSPC figures in: ", PRIMARY_DIR)
message("Backup and source data in: ", BACKUP_DIR)
