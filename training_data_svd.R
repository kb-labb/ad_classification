library(readr)
library(dplyr)
df <- read_csv("~/Desktop/ad_classification/data/annotation/svd_design5_training_ad.csv")

df <- df %>% 
  filter(!is.na(label))

df_design <- arrow::read_feather("~/Desktop/KB_newspapers/data/svd_design5.feather")
df_design$page_id <- stringr::str_match(df_design$id, pattern = ".*#[0-9]-[0-9]{1,2}")[, 1]

df_ads <- df_design %>%
  left_join(df %>% select(page_id, label), by = "page_id") %>%
  filter(training_set == 1) %>%
  # filter(!is.na(label) & label != "mixed") %>%
  select(id, part, page, label, image_path, content)

arrow::write_parquet(df_ads, "~/Desktop/ad_classification/data/svd_ads.parquet")
