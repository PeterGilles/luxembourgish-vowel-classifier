# create new emudb from collection of TextGrids and Wavs
library(dplyr)
library(reticulate)
library(purrr)
library(stringr)
library(emuR)

db <- load_emuDB("/Users/peter.gilles/Library/CloudStorage/OneDrive-UniversityofLuxembourg/Schnëssen_audio_ASR/Schnessen_ASR_emuDB")

# prepare export for vowel classifier

# Define the vowel labels (no quotes inside the string)
targets <- c("oː", "aː", "eː", "ɑɪ", "æːɪ", "ɜɪ", "əʊ", "ɑʊ", "æːʊ")
query_string <- paste0("[MAU==", paste(targets, collapse = "|"), "]")

# Perform the query
vowel_segs <- query(db, query_string)

# remove stopwords
# load file with stopwords
stopwords <- rio::import("/Users/peter.gilles/Documents/_Daten/R_scripts/workflow_create_emuDB/stopwords.txt")

vowel3 <- vowel_segs %>%
  # add column for ORT
  mutate(ORT = requery_hier(db, labels, level = "ORT", timeRefSegmentLevel = "MAU")$labels) 

%>%
  # column for stopwords / content words & check if ORT is stopword or content word
  mutate(stopword = if_else(tolower(ORT) %in% stopwords$stopword, "stopword", "content word")) %>%
  # filter out stopwords
  filter(stopword != "stopword")

#serve(db, seglist = vowel_segs)

# Count tokens per speaker
speaker_stats <- vowel_segs |>
  filter(end-start >=100 & end-start <= 300) |>
  group_by(bundle, labels) |>
  summarise(n = n()) |>
  tidyr::pivot_wider(names_from = labels, values_from = n, values_fill = 0) |>
  filter(eː >= 50, `ɑɪ` >= 50, `æːɪ` >= 50)

print(speaker_stats)

sozialdaten <- rio::import("../vowel_analysis/sozialdaten.csv") %>% 
  rename(id = ID) %>%
  filter(!Muttersprache == "Neen") %>%
  filter(!Geschlecht == "Aner") %>%
  filter(!Dialektgebiet == "") %>%
  dplyr::select(id, Decade = Alter, Gender = Geschlecht, Dialektgebiet, Gemeng = Gemeng_alt, Education = Ausbildung, Auswiel, device_id) %>%
  #mutate(id  = paste0("Speaker_", as.character(id))) %>%
  mutate(id  = as.character(id)) %>%
  mutate(Decade = as.factor(Decade)) %>%
  mutate(Gender = as.factor(Gender)) %>%
  mutate(Dialektgebiet = as.factor(Dialektgebiet)) %>%
  mutate(Gemeng = as.factor(Gemeng)) %>%
  # define max frequency parameters for formant extraction for male and female
  mutate(ceiling = ifelse(Gender == "Weiblech", 5500.0, 5000.0)) %>%
  mutate(Decade = recode(Decade, `≤ 24`= 2003, `25 bis 34` = 1993, `35 bis 44` = 1983, `45 bis 54` = 1973, `55 bis 64` = 1963, `65+` = 1953))  %>% 
  mutate(Gender = recode(Gender, `Weiblech` = "Female", `Männlech` = "Male")) %>%
  mutate(Education = recode(Education, 
                            `Etudes techniques supérieures (BTS, brevet de maîtrise)` = "Lycée technique",
                            `13e/14e (Lycée technique)` = "Lycée technique",
                            `CCP/DAP (CATP)` = "Lycée technique",
                            `9e (Lycée technique)` = "Lycée technique",
                            `Primaire/Grondschoul` = "just Primaire",
                            `5e (Lycée classique)` = "Lycée classique",
                            `1e (Lycée classique)` = "Lycée classique" )) %>%
  mutate(Education = factor(Education, levels = c("just Primaire", "Lycée technique", "Lycée classique", "Fachhéichschoul/Universitéit")))

selected_speakers <- vowel_segs %>%
  filter(end-start >=100 & end-start <= 300) |>
  mutate(id = stringr::word(bundle, 3, sep ="_")) %>%
  # join Sozialdaten
  inner_join(sozialdaten) %>%
  filter(Decade == 1993, Dialektgebiet == "Zentrum", Education == "Fachhéichschoul/Universitéit")


# Use the Python environment where pydub/librosa are installed
use_virtualenv("../.venv", required = TRUE)
py_install(c("librosa", "pydub", "soundfile"))  # if not installed

# Use Python env with pydub installed
pydub <- import("pydub")
AudioSegment <- pydub$AudioSegment

# Export function with pydub
export_segment_pydub <- function(session, bundle, start, end, label,
                                 base_path = "/Users/peter.gilles/Library/CloudStorage/OneDrive-UniversityofLuxembourg/Schnëssen_audio_ASR/Schnessen_ASR_emuDB/",
                                 export_dir = "exported_vowels") {
  wav_path <- file.path(base_path, paste0(session, "_ses"), paste0(bundle, "_bndl"), paste0(bundle, ".wav"))
  if (!file.exists(wav_path)) {
    message("Missing file: ", wav_path)
    return(NULL)
  }
  
  audio <- AudioSegment$from_wav(wav_path)
  start_ms <- as.integer(start)
  end_ms <- as.integer(end)
  segment <- audio[start_ms:end_ms]
  
  fname <- str_glue("{session}_{bundle}_{round(start, 2)}_{round(end, 2)}_{label}.wav")
  out_path <- file.path(export_dir, fname)
  # if file already exists, skip
  if (file.exists(out_path)) {
    message("File already exists: ", out_path)
    return(NULL)
  }
  segment$export(out_path, format = "wav")
  message("✅ Exported: ", out_path)
}

dir.create("exported_vowels", showWarnings = FALSE)

selected_speakers %>%
  select(session, bundle, start, end, label = labels) %>%
  pwalk(export_segment_pydub)


