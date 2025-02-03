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

cats <- c("shape1", "shape3", "number", "word", "Chinese", "face", "tool", "house")
derain_palette <- c("#97c684", "#6f9969", "#efc86e", "#aab5d5", "#808fe1", "#5c66a8", "#454a74", "#203050")
base_path <- "../bids_dataset/derivatives/extracted_betas/"

##############
# ROIs plots
##############

dbetas <-
  list.files(path=base_path, pattern="sub-.*_task-category_reftask-category_just-betas.csv") %>%
  lapply(function(x) {mutate(read.csv(paste0(base_path, x)), fname=x)}) %>%
  bind_rows %>%
  filter(name %in% cats) %>%
  mutate(name = ordered(name, levels=cats)) %>%
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
    mutate(age_group = paste0("Beta in the GLM (", age_group, ")")) %>%
    mutate(age_group = ordered(age_group, levels=unique(age_group))) %>%
    group_by(age_group, subject, task, name) %>%
    summarize(value = mean(value), .groups="keep") %>%
    group_by(age_group, task, name) %>%
    summarize(se=sd(value)/sqrt(length(value)),
              value = mean(value),
              .groups="keep") %>%
    ggplot(aes(x = name, y = value)) +
    new_scale_fill() +
    geom_bar(aes(fill=name), width=1, stat="identity") +
    geom_errorbar(aes(ymin=value-se, ymax=value+se), width=0, linewidth=.5) +
    scale_fill_manual(values=derain_palette) +
    ylab("") +
    ggtitle(k$title[[1]]) +
    facet_wrap(.~age_group, ncol=2) +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          panel.background = element_blank(),
          axis.ticks.x=element_blank()) +
    theme(strip.text.x = element_blank()) +
    theme(strip.background = element_blank()) +
    theme(panel.grid.major.y = element_blank()) +
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
              ph_with(dml(ggobj = all_plots[[idx]]), location = ph_location(left = 0.5, top = 3, width = 1.59, height = .95*(5/4)))
         }
       }) -> useless
all_slides %>% print(target="./figures/category_clusters.pptx")

##############
# Ventral plots
##############

relevants <-
  read.csv("../bids_dataset/derivatives/bootstrap_clusters/tables/ventral_adults.csv", header = F) %>%
  rename(age_group = V1, contrast = V2, idxs = V3) %>%
  group_by(contrast, idxs) %>%
  group_modify(function(d,k) {
    read.csv(paste0("../bids_dataset/derivatives/bootstrap_clusters/tables/adults_task-category_ctr-",k$contrast[[1]],"_table_full.csv")) %>%
      filter(idxs == k$idxs[[1]]) %>%
      mutate(side = if_else(X > 0, "right", "left")) %>%
      select(-idxs)
  }) %>%
  rename(roi_id = idxs, roi_contrast=contrast)

pl.betas <-
  relevants %>%
  inner_join(dbetas) %>%
  mutate(age_group = paste0("Beta in the GLM (", age_group, ")")) %>%
  mutate(age_group = ordered(age_group, levels=unique(age_group))) %>%
  group_by(side, age_group, roi_contrast, roi_id, subject, task, name) %>%
  summarize(value = mean(value), .groups="keep") %>%
  group_by(side, roi_contrast, roi_id, age_group, task, name) %>%
  summarize(se=sd(value)/sqrt(length(value)),
            value = mean(value),
            .groups="keep") %>%
  group_by(side, roi_contrast, roi_id) %>%
  group_map(function(d,k) {
    title <- paste0(k$roi_contrast, ", ", k$side)
    ggplot(d, aes(x = name, y = value)) +
    geom_bar(aes(fill=name), width=1, stat="identity") +
    geom_errorbar(aes(ymin=value-se, ymax=value+se), width=0, linewidth=.5) +
    scale_fill_manual(values=derain_palette) +
    ylab("") +
    facet_wrap(.~age_group, ncol=2) +
    ggtitle(title) +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          panel.background = element_blank(),
          axis.ticks.x=element_blank()) +
    theme(strip.text.x = element_blank()) +
    theme(strip.background = element_blank()) +
    theme(panel.grid.major.y = element_blank()) +
    theme(plot.margin = margin(0,0,6,0))
  })

all.rois.pl <- plot_grid(plotlist=pl.betas, ncol=1)

all_slides <<- read_pptx("./blank_A4.pptx")
lapply(1:length(pl.betas),
       function(idx) {
         all_slides <<-
           all_slides %>%
            add_slide(layout = "Title and Content", master = "Office Theme") %>%
            ph_with(value = idx, location=ph_location_type(type="title")) %>%
            ph_with(dml(ggobj = pl.betas[[idx]]), location = ph_location(left = 0.5, top = 3, width = 1.59, height = .95*(5/4)))
       }) -> useless
all_slides %>% print(target="./figures/ventral_clusters.pptx")

dbetas %>%
  inner_join(relevants) %>%
  filter(roi_contrast == "vwfa") %>%
  group_by(side, roi_contrast, roi_id, task) %>%
  group_modify(function(d,k) {
   contrast <- case_when(
     k$roi_contrast == "vwfa" ~ "word - Chinese",
     T ~ "ERROR")
   mdl <-
     d %>%
     pivot_wider(names_from=name, values_from=value) %>%
     mutate(ctr = eval(parse(text=contrast))) %>%
     ungroup
   adults <- tidy(t.test(filter(mdl, age_group == "Adults")$ctr, alternative = "greater"))
   kids <- tidy(t.test(filter(mdl, age_group != "Adults")$ctr, alternative = "greater"))
   rbind(mutate(adults, pop="Adults"),
         mutate(kids, pop="6 years old"))
  })

dbetas %>%
  inner_join(relevants) %>%
  filter(roi_contrast == "vwfa") %>%
  group_by(side, roi_contrast, roi_id, task) %>%
  group_modify(function(d,k) {
   contrast <- case_when(
     k$roi_contrast == "vwfa" ~ ".5*(shape1 + shape3) - .5*(face + Chinese)",
     T ~ "ERROR")
   mdl <-
     d %>%
     pivot_wider(names_from=name, values_from=value) %>%
     mutate(ctr = eval(parse(text=contrast))) %>%
     ungroup
   adults <- tidy(t.test(filter(mdl, age_group == "Adults")$ctr, alternative = "greater"))
   kids <- tidy(t.test(filter(mdl, age_group != "Adults")$ctr, alternative = "greater"))
   rbind(mutate(adults, pop="Adults"),
         mutate(kids, pop="6 years old"))
  })


dbetas %>%
  inner_join(relevants) %>%
  filter(roi_contrast == "house") %>%
  group_by(side, roi_contrast, roi_id, task) %>%
  group_modify(function(d,k) {
   contrast <- case_when(
     k$roi_contrast == "house" ~ ".5*(shape1 + shape3) - face",
     T ~ "ERROR")
   mdl <-
     d %>%
     pivot_wider(names_from=name, values_from=value) %>%
     mutate(ctr = eval(parse(text=contrast))) %>%
     ungroup
   adults <- tidy(t.test(filter(mdl, age_group == "Adults")$ctr, alternative = "greater"))
   kids <- tidy(t.test(filter(mdl, age_group != "Adults")$ctr, alternative = "greater"))
   rbind(mutate(adults, pop="Adults"),
         mutate(kids, pop="6 years old"))
  })



lapply(c("house", "face", "tool", "word"),
function(other) {
  mdl <-
    dbetas %>%
    inner_join(relevants) %>%
    filter(roi_contrast == "vwfa") %>%
    pivot_wider(names_from=name, values_from=value) %>%
    mutate(other = eval(parse(text=other))) %>%
    mutate(ctr = shape1 - other) %>%
    ungroup
   adults <- tidy(t.test(filter(mdl, age_group == "Adults")$ctr, alternative = "greater"))
   kids <- tidy(t.test(filter(mdl, age_group != "Adults")$ctr, alternative = "greater"))
   rbind(mutate(adults, pop="Adults", other=other),
         mutate(kids, pop="6 years old", other=other))
  }) %>%
  bind_rows %>%
  select(other, pop, statistic, p.value) %>%
  arrange(other) %>%
  mutate(statistic = round(statistic, 7),
         p.value = round(p.value, 8)) %>%
  rename(Other = other,
         `Age Group` = pop,
         `t-value` = statistic,
         `p-value` = p.value)


############
# Now the plot but with geom in cat ROIs
############

shapes_in_order <- c("square", "rectangle", "isoTrapezoid", "parallelogram", "losange", "kite", "rightKite", "rustedHinge", "hinge", "trapezoid", "random")
shapes_color  <- hcl(h = seq(15, 375, length = 12), l = 65, c = 100)[1:11]
behavior <- data.frame(name=shapes_in_order, model=c(0.0747863247863248, 0.0651709401709402, 0.150793650793651, 0.178266178266178, 0.214285714285714, 0.230006105006105, 0.256715506715507, 0.284696784696785, 0.302808302808303, 0.357244607244607, 0.421560846560847))
theory <- data.frame(name=shapes_in_order, model=1-((c(18, 14,  7,  7, 11,  3,  5,  8,  7,  5,  2) - 2) / 16))

dbetas_geom <-
  list.files(path=base_path, pattern="sub-.*_task-geometry.*_reftask-category_just-betas.csv") %>%
  lapply(function(x) {mutate(read.csv(paste0(base_path, x)), fname=x)}) %>%
  bind_rows %>%
  mutate(name = ordered(name, levels=shapes_in_order)) %>%
  mutate(age_group = if_else(as.numeric(str_sub(subject, start=5)) >= 300, "6 years old", "Adults")) %>%
  mutate(value = if_else(task == "geometryHard", value/10, value))

all_plots <-
  read.csv("../bids_dataset/derivatives/bootstrap_clusters/tables/adults_task-category_ctr-shape1_table_full.csv") %>%
  filter(pval < .5) %>%
  mutate(title = paste0("(",X,",",Y,",",Z,")")) %>%
  mutate(roi_id = idxs) %>%
  filter(roi_id %in% c(3, 4, 24, 28, 29)) %>%  # Manual filtering of the relevant ROIs.
  inner_join(dbetas_geom, by="roi_id") %>%
  filter(subject != "sub-308") %>%
  group_by(roi_id, title) %>%
  group_map(function(d,k) {
    test <- mean(filter(d, roi_contrast == "shape1")$value)
    if (is.na(test)) {
      return(NULL)
    }
    data <-
      d %>%
      filter(roi_contrast == "shape1") %>%
      mutate(age_group_str = paste0("Beta in the GLM (", age_group, ")")) %>%
      mutate(age_group_str = ordered(age_group, levels=unique(age_group))) %>%
      group_by(age_group, age_group_str, subject, task, name) %>%
      summarize(value = mean(value), .groups="keep") %>%
      group_by(age_group, age_group_str, task, name) %>%
      inner_join(behavior)
    adults_e <- mixed(value ~ model + (model | subject), data=filter(data, age_group=="Adults", task=="geometry"))
    adults_h <- mixed(value ~ model + (model | subject), data=filter(data, age_group=="Adults", task=="geometryHard"))
    kids <- mixed(value ~ model + (model | subject), data=filter(data, age_group!="Adults"))
    full <- mixed(value ~ model + (1 | task/subject) + (1 | age_group/subject), data=data)
    models <-
      bind_rows(mutate(filter(tidy(adults_e$full_model), effect == "fixed"), age_group="Adults", task="geometry"),
                mutate(filter(tidy(adults_h$full_model), effect == "fixed"), age_group="Adults", task="geometryHard"),
                mutate(filter(tidy(kids$full_model), effect == "fixed"), age_group="6 years old", task="geometry")) %>%
      mutate(age_group_str = paste0("Beta in the GLM (", age_group, ")")) %>%
      mutate(age_group_str = ordered(age_group, levels=unique(age_group))) %>%
      filter(term == "model") %>%
      mutate(pstr = paste0("p=", round(p.value, 2)))
    pl.betas <-
      data %>%
      group_by(age_group, age_group_str, task, name, model) %>%
      summarize(se=sd(value)/sqrt(length(value)),
                value = mean(value),
                .groups="keep") %>%
      mutate(name = ordered(name, levels=shapes_in_order)) %>%
      ggplot(aes(x = model, y = value, color = name)) +
      geom_point() +
      geom_smooth(aes(group=NA), method="lm", se=F, color="black") +
      scale_color_manual(values=shapes_color, drop = FALSE) +
      ylab("") +
      ggtitle(k$title[[1]]) +
      geom_text(data=models, aes(x=.5,y=.5,label=pstr,color=NA)) +
      facet_wrap(task~age_group_str, ncol=2) +
      theme(axis.title.x=element_blank(),
            axis.text.x=element_blank(),
            panel.background = element_blank(),
            axis.ticks.x=element_blank()) +
      theme(strip.text.x = element_blank()) +
      theme(strip.background = element_blank()) +
      theme(panel.grid.major.y = element_blank()) +
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
              ph_with(dml(ggobj = all_plots[[idx]]), location = ph_location(left = 0.5, top = 3, width = 1.59, height = 2*.95*(5/4)))
         }
       }) -> useless
all_slides %>% print(target="./figures/geometry_category_clusters.pptx")


#########
# Now the plot but with geom in geom ROIs
########

dbetas_geom <-
  list.files(path=base_path, pattern="sub-.*_task-geometry.*_reftask-geometry_just-betas.csv") %>%
  lapply(function(x) {mutate(read.csv(paste0(base_path, x)), fname=x)}) %>%
  bind_rows %>%
  mutate(name = ordered(name, levels=shapes_in_order)) %>%
  mutate(age_group = if_else(as.numeric(str_sub(subject, start=5)) >= 300, "6 years old", "Adults")) %>%
  mutate(value = if_else(task == "geometryHard", value/10, value)) %>%
  filter(task != "geometryHard")


read.csv("../bids_dataset/derivatives/bootstrap_clusters/tables/adults_task-geometry_ctr-geom_behavior_online_table_full.csv") %>%
  filter(pval < .05) %>%
  mutate(title = paste0("(",X,",",Y,",",Z,")")) %>%
  mutate(roi_id = idxs) %>%
  inner_join(dbetas_geom, by="roi_id") %>%
  filter(subject != "sub-308") %>%
  mutate(roi_id = ordered(roi_id, levels=unique(roi_id))) %>%
  select(roi_id, Peak.Stat) %>%
  unique


all_plots <-
  read.csv("../bids_dataset/derivatives/bootstrap_clusters/tables/adults_task-geometry_ctr-geom_behavior_online_table_full.csv") %>%
  filter(pval < .05) %>%
  mutate(title = paste0("(",X,",",Y,",",Z,")")) %>%
  mutate(roi_id = idxs) %>%
  inner_join(dbetas_geom, by="roi_id") %>%
  filter(subject != "sub-308") %>%
  mutate(roi_id = ordered(roi_id, levels=unique(roi_id))) %>%
  arrange(roi_id) %>%
  group_by(roi_id, title) %>%
  group_map(function(d,k) {
    test <- mean(filter(d, roi_contrast == "geom_behavior_online")$value)
    if (is.na(test)) {
      return(NULL)
    }
    data <-
      d %>%
      filter(roi_contrast == "geom_behavior_online") %>%
      mutate(age_group_str = paste0("Beta in the GLM (", age_group, ")")) %>%
      mutate(age_group_str = ordered(age_group, levels=unique(age_group))) %>%
      group_by(age_group, age_group_str, subject, task, name) %>%
      summarize(value = mean(value), .groups="keep") %>%
      group_by(age_group, age_group_str, task, name) %>%
      inner_join(behavior)

    adults_e <- mixed(value ~ model + (model | subject), data=filter(data, age_group=="Adults", task=="geometry"))
    kids <- mixed(value ~ model + (model | subject), data=filter(data, age_group!="Adults"))

    models <-
      bind_rows(mutate(filter(tidy(adults_e$full_model), effect == "fixed"), age_group="Adults", task="geometry"),
                mutate(filter(tidy(kids$full_model), effect == "fixed"), age_group="6 years old", task="geometry")) %>%
      mutate(age_group_str = paste0("Beta in the GLM (", age_group, ")")) %>%
      mutate(age_group_str = ordered(age_group, levels=unique(age_group))) %>%
      filter(term == "model") %>%
      mutate(pstr = paste0("p=", round(14*p.value/2, 2))) # 14 is number of multiple comp., 2 is because one tailed

    if (all(models$p.value < .05)) {
      print(paste0(k$title[[1]], round(models$p.value[[1]],3), round(models$p.value[[2]],3)))
      pl.betas <-
        data %>%
        group_by(age_group, age_group_str, task, name, model) %>%
        summarize(se=sd(value)/sqrt(length(value)),
                  value = mean(value),
                  .groups="keep") %>%
        mutate(name = ordered(name, levels=shapes_in_order)) %>%
        ggplot(aes(x = model, y = value, color = name)) +
        geom_point() +
        geom_errorbar(aes(ymin=value-se, ymax=value+se), width=0, linewidth=.5) +
        geom_smooth(aes(group=NA), method="lm", se=F, color="black") +
        scale_color_manual(values=shapes_color, drop = FALSE) +
        ylab("") +
        ggtitle(paste0(k$title[[1]], round(models$p.value[[1]],3), round(models$p.value[[2]],3))) +
        geom_text(data=models, aes(x=.25,y=.0,label=pstr,color=NA)) +
        facet_wrap(task~age_group_str, ncol=2) +
        theme(axis.title.x=element_blank()) +
        #theme(axis.text.x=element_blank()) +
        theme(panel.background = element_blank()) +
        #theme(axis.ticks.x=element_blank()) +
        #theme(strip.text.x = element_blank()) +
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
              ph_with(dml(ggobj = all_plots[[idx]]), location = ph_location(left = 0.5, top = 3, width = 2*1.59, height = 2*.95*(5/4)))
         }
       }) -> useless
all_slides %>% print(target="./figures/geometry_geom_clusters.pptx")

## OUTPUT CLUSTER TABLE

cl.adults <-
  read.csv("../bids_dataset/derivatives/bootstrap_clusters/tables/adults_task-category_ctr-shape1_table_full.csv") %>%
  filter(pval < .05) %>%
  mutate(title = paste0(idxs, " (",X,",",Y,",",Z,"); p=", pval)) %>%
  mutate(roi_id = idxs) %>%
  mutate(population = "adults")

cl.kids <-
  read.csv("../bids_dataset/derivatives/bootstrap_clusters/tables/kids_task-category_ctr-shape1_table_full.csv") %>%
  filter(pval < .05) %>%
  mutate(title = paste0(idxs, " (",X,",",Y,",",Z,"); p=", pval)) %>%
  mutate(roi_id = idxs) %>%
  mutate(population = "kids")

table.to.print <-
  bind_rows(cl.adults, cl.kids) %>%
  select(-X.1, -Cluster.ID, -title, -idxs, -weight, -roi_id) %>%
  select(population, X, Y, Y, everything()) %>%
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
  merge_v(j = "population", target = "population")

save_as_docx(ft, path="./figures/table_both.docx")

