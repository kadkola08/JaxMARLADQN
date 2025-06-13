import wandb
import datetime

api = wandb.Api()
map_name = "HEURISTICENEMYSMAX_SMACV2_10_UNITS_FIX_POS"

start_date = datetime.datetime(2025, 4, 30) 
end_date = datetime.datetime(2025, 5, 5) 

start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

def clean_column_names(df):
    df_clean = df.copy()

    new_columns = {}
    for col in df.columns:
        if '/' in col:
            new_name = col.split('/')[-1]
            new_columns[col] = new_name
        else:
            new_columns[col] = col

    df_clean = df_clean.rename(columns=new_columns)

    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]

    return df_clean

runs = api.runs(
                    "anuragk1/ADQN",
                    filters={
                        "tags" : {
                             "$all" : [map_name, "QMIX_RNN32"],
                             # "$in" : [map_name, ],
                             # "$nin" : ["MOD_QMIX"]
                        },
                        "config.SEED": {"$gt": 9},
                        # "created_at": {
                        #     "$gte": start_date_str,
                        #     "$lt": end_date_str     
                        # }
                    }
                )

print(runs)
print(len(runs))
for i, run in enumerate(runs):
    print(i)
    # Get history data (all logged metrics over time)
    history = run.history()
    history_clean = clean_column_names(history)
    # breakpoint()
    history_clean.to_csv(f"./exps/mod_qmix_{map_name}_i_{i}.csv", index=False)

