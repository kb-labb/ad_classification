library(readr)
library(dplyr)
df <- read_csv("~/Desktop/ad_classification/data/annotation/svd_design5_training_ad.csv")

df <- df %>% 
  filter(!is.na(label))

df_design <- arrow::read_feather("~/Desktop/KB_newspapers/data/svd_design5.feather")
df_design$page_id <- stringr::str_match(df_design$id, pattern = ".*#[0-9]-[0-9]{1,2}")[, 1]

# Observations from pages wheere whole page contains ads or whole page contains editorial content 
df_full_page <- df_design %>%
  left_join(df %>% select(page_id, label), by = "page_id") %>%
  filter(training_set == 1) %>%
  # filter(!is.na(label) & label != "mixed") %>%
  select(id, dark_id, part, page, date, type, label, content, x, y, width, height, 
         page, part, font_size, font_style, year, weekday, image_path, page_image_url)


# Annotations where pages contain Mixed ad and editorial content
# Annotator has only annotated the "ad" content on those pages
df_mixed <- list.files("data/annotation/mixed", full.names = TRUE) %>% 
  purrr::map_df(~read_csv(.)) %>%
  rename(label = tag) %>%
  mutate(label = "ad")

# Update the "mixed" labels with actual ad labels
df_ads <- df_full_page %>%
  rows_update(df_mixed %>% select(id, label), by = "id")

# Set rest of "mixed" labels in mixed pages to editorial label
df_mixed <- df_ads %>%
  group_by(dark_id, part, page) %>%
  filter(!all(label == "ad") & !all(label == "editorial") & !all(label == "mixed")) %>%
  mutate(label = ifelse(label == "mixed", yes = "editorial", no = label)) %>%
  ungroup()

# df_mixed now contains both editorial and ad labels in mixed pages
# Update those labels into main file again
df_final <- df_ads %>%
  rows_update(df_mixed %>% select(id, label), by = "id")

arrow::write_parquet(df_final, "~/Desktop/ad_classification/data/svd_ads.parquet")




# Plots
library(ggplot2)
df_final %>%
  group_by(text_length) %>%
  count() %>%
  arrange(-n)

df_final$text_length <- stringr::str_length(df_final$content)
df_final$text_length[is.na(df_final$text_length)] <- 0

ggplot(df_final, aes(x = text_length)) +
  geom_histogram(aes(y = stat(count/sum(count))), boundary = 0, bins = 50, colour="grey75") +
  scale_x_continuous(limits = c(0, 550), breaks = c(0, 1:10*50), labels = as.character(c(0, 1:10*50))) +
  labs(x = "Text length (number of characters)",
       y = "Proportion",
       title = "Distribution of character lengths for observations") +
  theme_light(base_size=13) + 
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5))


# 
a <- df_full_page %>%
  group_by(dark_id, part, page) %>%
  count() %>%
  arrange(-n)

mean(a$n)
sd(a$n)

ggplot(data=a, aes(x = n)) +
  geom_histogram(aes(y = stat(count/sum(count))),
                 bins = 40, colour = "grey75", boundary=0) + 
  theme_light(base_size = 13) +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5)) +
  labs(x = "Segmented boxes",
       y = "Proportion",
       title = "Distribution of segmented OCR boxes per page") +
  scale_x_continuous(limits = c(0, 188), breaks = c(0, 1:10*20), labels = as.character(c(0, 1:10*20)))