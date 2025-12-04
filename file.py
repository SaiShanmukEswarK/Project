#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
        .appName("CSV_to_Parquet_PUMS")
        .getOrCreate()
)
sc = spark.sparkContext

# print("Spark master:", spark.sparkContext.master)
# should show: spark://harrisburg.cs.colostate.edu:30176


# In[7]:


# from pyspark.sql import SparkSession


# spark = (SparkSession.builder
#         .appName("CSV_to_Parquet_PUMS")
#         .getOrCreate())


base = "file:///s/chopin/b/grad/C837217249/project"

input_dir    = f"{base}/fd"
output_dir   = f"{base}/parque_data"
targets_base = f"{base}/target_Datasets"
parquet_base = f"{base}/parque_data"
out_base     = f"{base}/complete"
out_panel    = f"{base}/final_state_panel"
final_single = f"{base}/final_panel_single"


years = [2019, 2020, 2021, 2022, 2023]

for year in years:
    csv_path = f"{input_dir}/pums_{year}.csv"
    parquet_path = f"{output_dir}/pums_{year}.parquet"
    
    print(f"Reading {csv_path} ...")
    df = (
        spark.read
             .option("header", "true")       
             .option("inferSchema", "true")  
             .csv(csv_path)
    )

    print(f"Writing {parquet_path} ...")
    (
        df.write
          .mode("overwrite")   
          .parquet(parquet_path)
    )

print("Done")



# In[9]:


# spark.sparkContext.master


# In[60]:


df_all = spark.read.parquet("file:///s/chopin/b/grad/C837217249/project/parque_data/pums_*.parquet")


# In[11]:


# from pyspark.sql import SparkSession

# spark = (
#     SparkSession.builder
#     .appName("Explore_PUMS")
#     .master("spark://harrisburg.cs.colostate.edu:30176")
#     .getOrCreate()
# )

# from pyspark.sql import SparkSession

# spark = (
#     SparkSession.builder
#         .appName("CSV_to_Parquet_PUMS")
#         .master("spark://harrisburg.cs.colostate.edu:30176")
#         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
#         .config("spark.driver.memory", "4g")
#         .config("spark.executor.memory", "4g")
#         .getOrCreate()
# )

# print("Spark master:", spark.sparkContext.master)


# In[ ]:





# In[1]:


# from pyspark.sql import SparkSession

# spark = (
#     SparkSession.builder
#     .appName("Explore_PUMS")
#     .getOrCreate()
# )

df = spark.read.parquet(
    "file:///s/chopin/b/grad/C837217249/project/parque_data/pums_*.parquet"
)


# In[2]:


df.printSchema()


# In[3]:


# spark.sparkContext.master
# spark.sparkContext.uiWebUrl


# In[4]:


# spark.sparkContext.master

# In[62]:


import os

base = "/s/chopin/b/grad/C837217249/project/parque_data"
for root, dirs, files in os.walk(base):
    print(root)
    for f in files:
        print("   ", f)


# In[63]:


# import pandas as pd
# rows = df.limit(50).collect()


# # In[64]:


# rows = pd.DataFrame(rows)
# rows


# In[65]:


df = spark.read.parquet(
    "file:///s/chopin/b/grad/C837217249/project/parque_data/pums_*.parquet"
)


df.printSchema()
df.show(5)


# In[66]:


# total rows
total_rows = df.count()
print("Total rows:", total_rows)

# total columns
print("Total columns:", len(df.columns))


# In[67]:


from pyspark.sql.functions import col, sum as _sum, count, expr

total_rows = df.count()

# null for single row
null_counts_row = df.select([
    _sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
])


stack_expr = "stack({n}, {pairs}) as (column, null_count)".format(
    n=len(df.columns),
    pairs=", ".join([f"'{c}', `{c}`" for c in df.columns])
)

nulls_long = null_counts_row.selectExpr(stack_expr)


null_report = (
    nulls_long
    .withColumn("non_null_count", expr(f"{total_rows} - null_count"))
    .withColumn("null_percent", (col("null_count") / total_rows) * 100)
    .orderBy(col("null_percent").desc())
)

null_report.show(35)


# In[84]:


from pyspark.sql import functions as F

base_path = "file:///s/chopin/b/grad/C837217249/project/parque_data"
years = [2019, 2020, 2021, 2022, 2023]

for year in years:
    print("\n" + "="*80)
    print(f"YEAR: {year}")
    print("="*80)

    path = f"{base_path}/pums_{year}.parquet"
    df_y = spark.read.parquet(path)


    total_rows = df_y.count()
    print(f"Total rows in {year}: {total_rows}")

    null_count_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df_y.columns
    ]
    null_counts_wide = df_y.agg(*null_count_exprs)


    stack_expr = "stack({n}, {pairs}) as (column, null_count)".format(
        n=len(df_y.columns),
        pairs=", ".join([f"'{c}', `{c}`" for c in df_y.columns])
    )

    nulls_long = null_counts_wide.selectExpr(stack_expr)


    null_report = (
        nulls_long
        .withColumn("non_null_count", F.lit(total_rows) - F.col("null_count"))
        .withColumn("null_percent", (F.col("null_count") / F.lit(total_rows)) * 100)
        .orderBy(F.col("null_percent").desc())
    )

    print(f"\nNull report per column for {year}")
    null_report.show(50, truncate=False)  


# In[69]:


path_2022 = "file:///s/chopin/b/grad/C837217249/project/parque_data/pums_2022.parquet"
df22 = spark.read.parquet(path_2022)

df22.printSchema()
df22.show(5)


# In[39]:


df22.describe("AGEP").show()


# # Target dataset

# In[70]:


import os

# spark = SparkSession.builder.appName("Merge_PUMS_and_Targets").getOrCreate()

targets_base = "file:///s/chopin/b/grad/C837217249/project/target_Datasets"
years = [2019, 2020, 2021, 2022, 2023]

state_names = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
    "Delaware","District of Columbia","Florida","Georgia","Hawaii","Idaho","Illinois",
    "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts",
    "Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
    "New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota",
    "Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina",
    "South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington",
    "West Virginia","Wisconsin","Wyoming",
]

def load_target(year: int):
    path = os.path.join(targets_base, f"Table_{year}.csv")
    print(f"Loading {path} ...")

    raw = spark.read.text(path)

    body_rdd = (
        raw.rdd
           .zipWithIndex()
           .filter(lambda x: x[1] >= 3)      
           .map(lambda x: x[0][0])
    )

    df = spark.read.csv(body_rdd, header=True, inferSchema=False)

    df = df.filter(F.col("GeoFips").rlike(r"^[0-9]+$"))

    df = df.withColumn("GeoFips_int", F.col("GeoFips").cast("int"))
    df = df.filter(F.col("GeoFips_int").isNotNull())
    df = df.filter(F.col("GeoFips_int") != 0)

    df = df.filter(F.col("GeoName").isin(state_names))

    df = df.withColumn("ST", (F.col("GeoFips_int") / 1000).cast("int"))

    value_col_name = str(year)
    df = df.withColumn("YEAR", F.lit(year))
    df = df.withColumn("PCE_PC", F.col(value_col_name).cast("double"))

    df = df.select(
        "YEAR",
        "ST",
        F.col("GeoName").alias("STATE_NAME"),
        "PCE_PC"
    )

    return df

df_target_all = None
for y in years:
    df_y = load_target(y)
    if df_target_all is None:
        df_target_all = df_y
    else:
        df_target_all = df_target_all.unionByName(df_y)

print("Combined target row count:", df_target_all.count())
df_target_all.orderBy("YEAR", "ST").show(60, truncate=False)
df_target_all.printSchema()


# In[71]:


from pyspark.sql import SparkSession, functions as F
from functools import reduce


parquet_base = "file:///s/chopin/b/grad/C837217249/project/parque_data"
years = [2019, 2020, 2021, 2022, 2023]

dfs = []
for y in years:
    path = f"{parquet_base}/pums_{y}.parquet"
    print(f"Reading PUMS parquet for {y}: {path}")
    
    df_y = spark.read.parquet(path)


    if "YEAR" not in df_y.columns:
        df_y = df_y.withColumn("YEAR", F.lit(y))
    else:

        df_y = df_y.withColumn("YEAR", F.col("YEAR").cast("int"))

    dfs.append(df_y)


df_pums_all = reduce(
    lambda a, b: a.unionByName(b, allowMissingColumns=True),
    dfs
)

print("Total PUMS rows (all years):", df_pums_all.count())
print("Number of columns:", len(df_pums_all.columns))
df_pums_all.printSchema()


# In[72]:


df_target_all = df_target_all.withColumnRenamed("PCE_PC", "TOTAL_HEALTH_SPENDING")


df_target_small = df_target_all.select("YEAR", "ST", "STATE_NAME", "TOTAL_HEALTH_SPENDING")

print("Target rows:", df_target_small.count())


df_pums_with_target = (
    df_pums_all
    .join(F.broadcast(df_target_small), on=["YEAR", "ST"], how="left")
)

print("Joined rows:", df_pums_with_target.count())
df_pums_with_target.printSchema()
df_pums_with_target.show(5)


# In[ ]:





# In[73]:


from pyspark.sql import functions as F

out_base = "file:///s/chopin/b/grad/C837217249/project/complete"
years = [2019, 2020, 2021, 2022, 2023]

for y in years:
    out_path = f"{out_base}/pums_{y}.parquet"
    print(f"Writing year {y} to {out_path} ...")
    
    (
        df_pums_with_target
        .filter(F.col("YEAR") == y)

        .write
        .mode("overwrite")
        .parquet(out_path)
    )

print("Done writing per-year Parquet datasets.")


# In[74]:


path_2022 = "file:///s/chopin/b/grad/C837217249/project/complete/pums_2022.parquet"

df22 = spark.read.parquet(path_2022)

print("Rows in 2022 parquet:", df22.count())
print("Columns:", len(df22.columns))
df22.printSchema()


# In[75]:


out_base = "file:///s/chopin/b/grad/C837217249/project/complete"
path_2022 = f"{out_base}/pums_2022.parquet"

df22 = spark.read.parquet(path_2022)

print("Rows in 2022 parquet:", df22.count())
print("Columns:", len(df22.columns))
df22.printSchema()


# In[76]:


df22.select("YEAR", "ST", "STATE_NAME", "TOTAL_HEALTH_SPENDING").show(30, False)


# In[77]:


from pyspark.sql import SparkSession, functions as F
from functools import reduce


# spark = SparkSession.builder.appName("StateLevel_HealthSpending_Panel").getOrCreate()


parquet_base = "file:///s/chopin/b/grad/C837217249/project/complete"
years = [2019, 2020, 2021, 2022, 2023]

dfs = []
for y in years:
    path = f"{parquet_base}/pums_{y}.parquet"
    print(f"Reading PUMS+target parquet for {y}: {path}")
    
    df_y = spark.read.parquet(path)

    if "YEAR" not in df_y.columns:
        df_y = df_y.withColumn("YEAR", F.lit(y))
    else:
        df_y = df_y.withColumn("YEAR", F.col("YEAR").cast("int"))

    dfs.append(df_y)


df_all = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dfs)

print("Total micro rows (all years):", df_all.count())
print("Columns:", len(df_all.columns))
df_all.printSchema()

#  Aggregating to state–year weighted sums
group_cols = ["YEAR", "ST"]

agg = (
    df_all
    .groupBy(*group_cols)
    .agg(
        # ID
        F.first("STATE_NAME").alias("STATE_NAME"),
        F.first("TOTAL_HEALTH_SPENDING").alias("TOTAL_HEALTH_SPENDING"),

        # Total weighted population
        F.sum("PWGTP").alias("wt_pop"),

        # Age
        F.sum(F.when(F.col("AGEP") >= 65, F.col("PWGTP")).otherwise(0)).alias("wt_age_65_plus"),
        F.sum(F.when(F.col("AGEP") <= 17, F.col("PWGTP")).otherwise(0)).alias("wt_age_0_17"),

        # Sex
        F.sum(F.when(F.col("SEX") == 2, F.col("PWGTP")).otherwise(0)).alias("wt_female"),

        # Marital status
        F.sum(F.when(F.col("MAR").isin(1, 2, 3), F.col("PWGTP")).otherwise(0)).alias("wt_ever_married"),

        # Insurance coverage
        F.sum(F.when(F.col("HICOV") == 2, F.col("PWGTP")).otherwise(0)).alias("wt_uninsured"),

        F.sum(F.when(F.col("HINS1") == 1, F.col("PWGTP")).otherwise(0)).alias("wt_hins1_employer"),
        F.sum(F.when(F.col("HINS2") == 1, F.col("PWGTP")).otherwise(0)).alias("wt_hins2_direct"),
        F.sum(F.when(F.col("HINS3") == 1, F.col("PWGTP")).otherwise(0)).alias("wt_hins3_medicare"),
        F.sum(F.when(F.col("HINS4") == 1, F.col("PWGTP")).otherwise(0)).alias("wt_hins4_medicaid"),
        F.sum(F.when(F.col("HINS5") == 1, F.col("PWGTP")).otherwise(0)).alias("wt_hins5_tricare"),
        F.sum(F.when(F.col("HINS6") == 1, F.col("PWGTP")).otherwise(0)).alias("wt_hins6_va"),

        # Poverty
        F.sum(F.when(F.col("POVPIP") < 138, F.col("PWGTP")).otherwise(0)).alias("wt_pov_lt138"),

        # Income & transfers
        F.sum(F.col("PWGTP") * F.col("PINCP")).alias("wt_sum_pincome"),
        F.sum(F.col("PWGTP") * F.col("WAGP")).alias("wt_sum_wagp"),

        F.sum(F.col("PWGTP") * F.col("SEMP")).alias("wt_sum_semp"),
        F.sum(F.when(F.col("SEMP") > 0, F.col("PWGTP")).otherwise(0)).alias("wt_semp_positive"),

        F.sum(F.col("PWGTP") * F.col("SSP")).alias("wt_sum_ssp"),
        F.sum(F.when(F.col("RETP") > 0, F.col("PWGTP")).otherwise(0)).alias("wt_ret_income_any"),

        # Labor force
        F.sum(F.when(F.col("ESR").isin(3, 6), F.col("PWGTP")).otherwise(0)).alias("wt_esr_unemp_or_nilf"),
        F.sum(F.when(F.col("ESR").isin(1, 2, 3, 4, 5), F.col("PWGTP")).otherwise(0)).alias("wt_in_labor_force"),

        # Education
        F.sum(F.when(F.col("SCHL") >= 21, F.col("PWGTP")).otherwise(0)).alias("wt_bach_plus"),

        # Disability
        F.sum(F.when(F.col("DOUT") == 1, F.col("PWGTP")).otherwise(0)).alias("wt_dout_diff"),
        F.sum(F.when(F.col("DPHY") == 1, F.col("PWGTP")).otherwise(0)).alias("wt_dphy_diff"),
        F.sum(F.when(F.col("DIS") == 1, F.col("PWGTP")).otherwise(0)).alias("wt_disabled_any"),

        # Fertility
        F.sum(
            F.when(
                (F.col("SEX") == 2) & (F.col("AGEP").between(15, 50)) & (F.col("FER") == 1),
                F.col("PWGTP")
            ).otherwise(0)
        ).alias("wt_births_women_15_50"),
        F.sum(
            F.when(
                (F.col("SEX") == 2) & (F.col("AGEP").between(15, 50)),
                F.col("PWGTP")
            ).otherwise(0)
        ).alias("wt_women_15_50"),

        # Broadband / access
        F.sum(F.when(F.col("BROADBND") == 1, F.col("PWGTP")).otherwise(0)).alias("wt_broadband_yes"),

        # Vehicles
        F.sum(F.col("PWGTP") * F.col("VEH")).alias("wt_sum_veh"),

        # Military
        F.sum(F.when(F.col("MIL").isin(2, 3), F.col("PWGTP")).otherwise(0)).alias("wt_veterans"),
        F.sum(F.when(F.col("MIL") == 1, F.col("PWGTP")).otherwise(0)).alias("wt_active_duty"),
    )
)

#  Turning weighted sums into per-capita
features = (
    agg
    .withColumn("AGEP_65plus_share",       F.col("wt_age_65_plus") / F.col("wt_pop"))
    .withColumn("AGEP_0_17_share",         F.col("wt_age_0_17")     / F.col("wt_pop"))
    .withColumn("SEX_female_share",        F.col("wt_female")       / F.col("wt_pop"))
    .withColumn("MAR_ever_married_share",  F.col("wt_ever_married") / F.col("wt_pop"))

    .withColumn("HICOV_uninsured_share",   F.col("wt_uninsured")      / F.col("wt_pop"))
    .withColumn("HINS1_employer_share",    F.col("wt_hins1_employer") / F.col("wt_pop"))
    .withColumn("HINS2_direct_share",      F.col("wt_hins2_direct")   / F.col("wt_pop"))
    .withColumn("HINS3_medicare_share",    F.col("wt_hins3_medicare") / F.col("wt_pop"))
    .withColumn("HINS4_medicaid_share",    F.col("wt_hins4_medicaid") / F.col("wt_pop"))
    .withColumn("HINS5_tricare_share",     F.col("wt_hins5_tricare")  / F.col("wt_pop"))
    .withColumn("HINS6_va_share",          F.col("wt_hins6_va")       / F.col("wt_pop"))

    .withColumn("POVPIP_lt138_share",      F.col("wt_pov_lt138") / F.col("wt_pop"))
    .withColumn("PINCP_mean",              F.col("wt_sum_pincome") / F.col("wt_pop"))
    .withColumn("WAGP_mean",               F.col("wt_sum_wagp")    / F.col("wt_pop"))
    .withColumn(
        "SEMP_mean_if_positive",
        F.when(F.col("wt_semp_positive") > 0,
               F.col("wt_sum_semp") / F.col("wt_semp_positive")
        ).otherwise(None)
    )
    .withColumn("SSP_per_capita",          F.col("wt_sum_ssp") / F.col("wt_pop"))
    .withColumn("RETP_positive_share",     F.col("wt_ret_income_any") / F.col("wt_pop"))

    .withColumn("ESR_unemp_or_nilf_share", F.col("wt_esr_unemp_or_nilf") / F.col("wt_pop"))
    .withColumn("ESR_in_labor_force_share",F.col("wt_in_labor_force") / F.col("wt_pop"))

    .withColumn("SCHL_bach_plus_share",    F.col("wt_bach_plus") / F.col("wt_pop"))

    .withColumn("DOUT_diff_share",         F.col("wt_dout_diff")    / F.col("wt_pop"))
    .withColumn("DPHY_diff_share",         F.col("wt_dphy_diff")    / F.col("wt_pop"))
    .withColumn("DIS_any_share",           F.col("wt_disabled_any") / F.col("wt_pop"))

    .withColumn(
        "FER_births_per_1000_women_15_50",
        F.when(F.col("wt_women_15_50") > 0,
               1000.0 * F.col("wt_births_women_15_50") / F.col("wt_women_15_50")
        ).otherwise(None)
    )

    .withColumn("BROADBND_yes_share",      F.col("wt_broadband_yes") / F.col("wt_pop"))
    .withColumn("VEH_mean_per_person",     F.col("wt_sum_veh") / F.col("wt_pop"))
    .withColumn("MIL_veteran_share",       F.col("wt_veterans") / F.col("wt_pop"))
    .withColumn("MIL_active_duty_share",   F.col("wt_active_duty") / F.col("wt_pop"))

    .select(
        "YEAR", "ST", "STATE_NAME", "TOTAL_HEALTH_SPENDING",
        "AGEP_65plus_share", "AGEP_0_17_share", "SEX_female_share",
        "MAR_ever_married_share",
        "HICOV_uninsured_share",
        "HINS1_employer_share", "HINS2_direct_share",
        "HINS3_medicare_share", "HINS4_medicaid_share",
        "HINS5_tricare_share", "HINS6_va_share",
        "POVPIP_lt138_share",
        "PINCP_mean", "WAGP_mean", "SEMP_mean_if_positive",
        "SSP_per_capita", "RETP_positive_share",
        "ESR_unemp_or_nilf_share", "ESR_in_labor_force_share",
        "SCHL_bach_plus_share",
        "DOUT_diff_share", "DPHY_diff_share", "DIS_any_share",
        "FER_births_per_1000_women_15_50",
        "BROADBND_yes_share", "VEH_mean_per_person",
        "MIL_veteran_share", "MIL_active_duty_share"
    )
)

print("State-year rows", features.count())
features.orderBy("YEAR", "ST").show(10, truncate=False)


out_panel = "file:///s/chopin/b/grad/C837217249/project/final_state_panel"
features.write.mode("overwrite").option("header", True).csv(out_panel)
print("State-level panel saved to:", out_panel)


# In[80]:


df = spark.read.csv("file:///s/chopin/b/grad/C837217249/project/final_state_panel", header=True, inferSchema=True)
df.printSchema()
df.show(5)
df.count()


# In[82]:


df.coalesce(1).write.mode("overwrite").option("header", True).csv("file:///s/chopin/b/grad/C837217249/project/final_panel_single")
# Print worker nodes used by this Spark job
print("\n=== EXECUTOR NODES USED ===")

# sc._jsc is the JavaSparkContext — this lets us query cluster info
executor_infos = sc._jsc.sc().statusTracker().getExecutorInfos()

nodes = set()
for info in executor_infos:
    host = info.host()
    # skip "driver"
    if host not in ("driver", "localhost"):
        nodes.add(host)

print("Worker nodes that executed tasks:")
for n in sorted(nodes):
    print("  -", n)

print("Total executors:", len(nodes))
print("Driver running on:", spark.sparkContext.master)
print("Driver host:", spark.sparkContext.sparkUser())


spark.stop()
# In[ ]:
