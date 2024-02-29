import pickle
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os

def parse_EEUD(data_path : Path, save_path : Path):
    """
    Parse the EEUD data and save it as a pickle file
    ## Params:
    data_path: Path to the EEUD data
    save_path: Path to save the parsed data as pickle file
    """
    # dict to store households data
    data = {}
    for file in tqdm(os.listdir(data_path)):
        name = "EEUD_"+file.split(".")[0][1:]
        if file.endswith(".csv"):
            # special cases because files are not in a consistent format
            if name == "EEUD_20" or name == "EEUD_17" or name == "EEUD_19" or name =="EEUD_22" or name == "EEUD_23" or name == "EEUD_18" or name =="EEUD_16":
                df = pd.read_csv(data_path / file, header=39).drop(columns=["No"])
            elif name == "EEUD_21":
                df = pd.read_csv(data_path / file, header=39).drop(columns=["No", "Unnamed: 3"])
            elif name == "EEUD_15":
                df = pd.read_csv(data_path / file ,on_bad_lines="warn", header=44).drop(columns=["No", "Unnamed: 7","Unnamed: 8"])
            elif name == "EEUD_13":
                df = pd.read_csv(data_path / file, header=45)
                # cols are shifted by one
                cols = df.columns[1:]
                df.drop(columns=cols[-1], inplace=True)
                df.columns = cols
            else:
                df = pd.read_csv(data_path / file, header=45).drop(columns=["No"])
            # set the date time as index and sort the index and remove duplicates
            df[' Date Time'] = pd.to_datetime(df[" Date Time"])
            df.set_index(" Date Time", inplace=True)
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep="first")]
            # convert to watts
            df *= 1000
            # resample to 1min and fill the missing values
            df = df.resample("1min").ffill()
            df.dropna(inplace=True)

            curr_data = {}
            for c in df.columns:
                device_name = c.split("(")[0].strip().lower()
                if device_name == "main":
                    device_name = "aggregate"
                print(c, device_name, name)
                curr_data[device_name] = pd.DataFrame(df[c])
                

            data[name] = curr_data
        elif file.endswith(".xls"):
            xls = pd.ExcelFile(data_path / file)
            sheet_names = xls.sheet_names
            dfs = []

            for sheet in sheet_names[1:-1]:
                
                s = sheet.split("    ")
                # construct datetime index
                curr_df = xls.parse(sheet)
                curr_df["Year"] = int(s[0])
                curr_df["Month"] = int(s[1])
                curr_df["index"] = pd.to_datetime(curr_df[["Day", "Hour", "Minute", "Month", "Year"]])
                curr_df.drop(columns=["Day", "Hour", "Minute", "Year", "Month"], inplace=True)
                curr_df.set_index("index", inplace=True)
                curr_df.sort_index(inplace=True)
                curr_df = curr_df[~curr_df.index.duplicated(keep="first")]

                #convert to watts
                curr_df *= 1000
                # resample to 1min and fill the missing values
                curr_df = curr_df.resample("1min").ffill(limit=2)
                curr_df.dropna(inplace=True)
                # append to the list of dataframes
                dfs.append(curr_df)
            # concat the dataframes
            df = pd.concat(dfs, axis=0)
            for c in df.columns:
                device_name = c.split("(")[0].strip().lower()
                if device_name == "main":
                    device_name = "aggregate"
                curr_data[device_name] = pd.DataFrame(df[c])
            data[name] = df
    # save with pickle
    with open(save_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)



