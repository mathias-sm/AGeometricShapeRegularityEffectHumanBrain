library(tidyverse)
library(ggplot2)
library(cowplot)
library(patchwork)
library(extrafont)
library(officer)
library(rvg)
library(ggnewscale)
library(afex)
library(broom)
library(broom.mixed)
library(flextable)
theme_set(theme_cowplot() +
          theme(text = element_text(family = "sans", size=9),
                axis.text = element_text(size=9),
                panel.grid.major.x = element_blank() ,
                panel.grid.major.y = element_line(linewidth=.1, color="black"),
                legend.title=element_blank(),
                legend.position="none"))

base_path_betas <- "../bids_dataset/derivatives/rsa/"

##############
# ROIs plots
##############


theories <- c("symbolic", "IT")

dbetas <-
  list.files(path = base_path_betas, pattern="sub-.*reftask-category_just-betas.csv", full.names = TRUE, recursive = TRUE) %>%
  lapply(function(x) {mutate(read.csv(x), fname=x)}) %>%
  bind_rows %>%
  filter(theory %in% theories) %>%
  mutate(theory = ordered(theory, levels=theories)) %>%
  mutate(age_group = if_else(as.numeric(str_sub(subject, start=5)) >= 300, "6 years old", "Adults"))

all_plots <-
  read.csv("../bids_dataset/derivatives/bootstrap_clusters/tables/adults_task-category_ctr-shape1_table_full.csv") %>%
  filter(pval < .5) %>%
  mutate(title = paste0(idxs, " (",X,",",Y,",",Z,"); p=", pval)) %>%
  mutate(roi_id = idxs) %>%
  filter(roi_id %in% c(3, 4, 24, 28, 29)) %>%  # Manual filtering of the relevant ROIs.
  inner_join(dbetas, by="roi_id") %>%
  group_by(roi_id, title) %>%
  group_map(function(d,k) {
  test <- mean(filter(d, roi_contrast == "shape1")$value)
  if (is.na(test)) {
    return(NULL)
  }
  pl.betas <-
    d %>%
    filter(roi_contrast == "shape1") %>%
    #mutate(age_group = paste0("Beta in the GLM (", age_group, ")")) %>%
    #mutate(age_group = ordered(age_group, levels=unique(age_group))) %>%
    group_by(age_group, subject, task, theory) %>%
    summarize(value = mean(value), .groups="keep") %>%
    group_by(age_group, task, theory) %>%
    summarize(se=sd(value)/sqrt(length(value)),
              value = mean(value),
              .groups="keep") %>%
    ggplot(aes(x = interaction(age_group, task), y = value)) +
    new_scale_fill() +
    geom_bar(aes(fill=interaction(age_group, task)), width=1, stat="identity") +
    geom_errorbar(aes(ymin=value-se, ymax=value+se), width=0, linewidth=.5) +
    ylab("") +
    ggtitle(k$title[[1]]) +
    facet_wrap(theory~.) +
    #theme(panel.background = element_blank(),
          #axis.ticks.x=element_blank()) +
    #theme(strip.text.x = element_blank()) +
    #theme(strip.background = element_blank()) +
    #theme(panel.grid.major.y = element_blank()) +
    theme(plot.margin = margin(0,0,6,0))
  pl.betas
  })

all.rois.pl <- plot_grid(plotlist=all_plots, ncol=1)

all_slides <<- read_pptx("./blank_A4.pptx")
lapply(1:length(all_plots),
       function(idx) {
         if (!is.null(all_plots[[idx]])) {
           all_slides <<-
             all_slides %>%
              add_slide(layout = "Title and Content", master = "Office Theme") %>%
              ph_with(dml(ggobj = all_plots[[idx]]), location = ph_location(left = 0.5, top = 3, width = 2*1.59, height = 4*.95*(5/4)))
         }
       }) -> useless
all_slides %>% print(target="./figures/category_clusters.pptx")

##############
# RSA rois plots
##############

dbetas.rsa <-
  list.files(path=base_path_betas, pattern="sub-.*rsa.*.csv", full.names = TRUE, recursive = TRUE) %>%
  lapply(function(x) {mutate(read.csv(x), fname=x)}) %>%
  bind_rows %>%
  filter(theory %in% theories) %>%
  mutate(theory = ordered(theory, levels=theories)) %>%
  mutate(age_group = if_else(as.numeric(str_sub(subject, start=5)) >= 300, "6 years old", "Adults"))

cl.IT <-
  read.csv("../bids_dataset/derivatives/rsa/sub-average/tables/pop-adults_task-geometry_theory-IT_table_full.csv") %>%
  filter(pval < .05) %>%
  mutate(title = paste0(idxs, " (",X,",",Y,",",Z,"); p=", pval, "; IT")) %>%
  mutate(roi_id = idxs) %>%
  mutate(theory = "IT")

cl.symbolic <-
  read.csv("../bids_dataset/derivatives/rsa/sub-average/tables/pop-adults_task-geometry_theory-symbolic_table_full.csv") %>%
  filter(pval < .05) %>%
  mutate(title = paste0(idxs, " (",X,",",Y,",",Z,"); p=", pval, "; symbolic")) %>%
  mutate(roi_id = idxs) %>%
  mutate(theory = "symbolic")


all_plots <-
  bind_rows(cl.IT, cl.symbolic) %>%
  inner_join(dbetas.rsa, by=c("theory", "roi_id")) %>%
  group_by(roi_id, title, theory) %>%
  group_map(function(d,k) {
    pval <-
      d %>%
      filter(age_group != "Adults") %>%
      pull(value) %>%
      t.test(alternative="greater") %>%
      tidy %>%
      pull(p.value)
    correct_factor = if_else(k$theory == "IT", 7, 16)
    if (pval < .05/correct_factor) {
      print(pval)
      pl.betas <-
        d %>%
        group_by(age_group, subject, task) %>%
        summarize(value = mean(value), .groups="keep") %>%
        group_by(age_group, task) %>%
        summarize(se=sd(value)/sqrt(length(value)),
                  value = mean(value),
                  .groups="keep") %>%
        ggplot(aes(x = interaction(age_group, task), y = value, fill=interaction(age_group, task))) +
        geom_bar(width=1, stat="identity", position="dodge") +
        geom_errorbar(aes(ymin=value-se, ymax=value+se), width=0, linewidth=.5, position=position_dodge(width=0.9)) +
        ylab("") +
        ggtitle(k$title[[1]]) +
        theme(panel.background = element_blank()) +
        theme(strip.background = element_blank()) +
        theme(panel.grid.major.y = element_blank()) +
        theme(plot.margin = margin(0,0,6,0))
      pl.betas
    }
  })

all.rois.pl <- plot_grid(plotlist=all_plots, ncol=1)

all_slides <<- read_pptx("./blank_A4.pptx")
lapply(1:length(all_plots),
       function(idx) {
         if (!is.null(all_plots[[idx]])) {
           all_slides <<-
             all_slides %>%
              add_slide(layout = "Title and Content", master = "Office Theme") %>%
              ph_with(dml(ggobj = all_plots[[idx]]), location = ph_location(left = 0.5, top = 3, width = 2*1.59, height = 4*.95*(5/4)))
         }
       }) -> useless

all_slides %>% print(target="./figures/rsa_clusters.pptx")


cl.IT.kids <-
  read.csv("../bids_dataset/derivatives/rsa/sub-average/tables/pop-kids_task-geometry_theory-IT_table_full.csv") %>%
  filter(pval < .1) %>%
  mutate(title = paste0(idxs, " (",X,",",Y,",",Z,"); p=", pval, "; IT")) %>%
  mutate(roi_id = idxs) %>%
  mutate(theory = "IT")

table.to.print <-
  bind_rows(cl.symbolic, cl.IT) %>%
  mutate(pop = "adults") %>%
  bind_rows(mutate(cl.IT.kids, pop="children")) %>%
  select(-X.1, -Cluster.ID, -title, -idxs, -weight, -roi_id) %>%
  select(pop, theory, X, Y, Y, everything()) %>%
  mutate(pval = if_else(pval<.01, "p<.01", paste0("p=", sub('.', '', sprintf("%.2f", signif(pval, 2))))),
         Peak.Stat = signif(Peak.Stat, 3),
         Cluster.Size..mm3. = round(Cluster.Size..mm3. / 1000, 1)) %>%
  rename(#`MNI Coordinates` = loc,
         `Peak t-value` = `Peak.Stat`,
         `Volume in cm³` = `Cluster.Size..mm3.`,
         `p` = pval)

ft <-
  table.to.print %>%
  flextable %>%
  merge_v(j = "pop", target = "pop") %>%
  merge_v(j = "theory", target = "theory")

save_as_docx(ft, path="./figures/table_both.docx")
