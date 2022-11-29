# clarify_kpi3

To reproduce the result of Normalized Famililar Cancer Frequency and Famililar Cancer Connectedness

1. Data table preparation: Using the the queries in sql.ipynb file to do query over https://labs.tib.eu/sdm/mysql/db_structure.php?server=1&db=SLCG_UPM_v5.0

2. Analysis: Using the python code from kpi3_jaccard.ipynb to 
   a. preprocess csv data files
   b. analysis of Normalized Famililar Cancer Frequency by using `family_cancer_frequency()` function
   c. analysis of Famililar Cancer Connectedness by using `jaccard_x_is_cancer()` function
   d. change the `age_threshold` then run the code from code block `Generate all box plot` for box plots of different age threshold
   
3. The final box plots is available from 👉 this [Google Drive](https://drive.google.com/drive/folders/1U3JEYoJuvgvvOfuGKktOQ3bhSzU_PKDw) or 👉 this [Notion Page](https://www.notion.so/Comparing-age-threshold-55-60-65-a930e78e3f2c4be59b5a506c549f810f)
