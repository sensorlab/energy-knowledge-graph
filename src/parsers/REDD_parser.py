import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from nilmtk import DataSet
import pandas as pd
from pathlib import Path
from src.helper import save_to_pickle


######################DATASET INFO#########################################
# sampling rate: 1s
# length: 1 month
# unit: watts
# households: 6
# submetered
# Location: USA (Massachusetts)
# Source: https://people.csail.mit.edu/mattjj/papers/kddsust2011.pdf

def load_redd_dataset(path: Path) -> list:
    try:
        dataset = DataSet(path)

        samples = []
        # iterate over buildings and devices
        for building_idx, building in dataset.buildings.items():
            aggregate = next(building.elec.mains()[1].load()) + next(building.elec.mains()[1].load())
            for meter in building.elec.all_meters():
                data = list(meter.load())
                assert len(data) == 1

                assert len(meter.appliances) < 2
                # store the good sections of the data for each device and aggregate
                sample = (
                    building_idx,
                    list([a.type["type"] for a in meter.appliances]),
                    data,
                    meter.good_sections(),
                    aggregate,
                )

                samples.append(sample)

        return samples

    except Exception as e:
        dataset.store.close()
        raise e


# format the data into a dictionary of dictionaries and concat the dataframes
def data_preparation(dataset: list) -> dict:
    out_data = {}
    for idx, appliances, data, good_sections, aggregate in dataset:
        name = "REDD_{}".format(idx)
        if name not in out_data:
            out_data[name] = {"aggregate": aggregate}
        if not appliances:
            continue

        appliance = appliances[0]
        data = data[0]
        if appliance == "unknown":
            continue
        samples = [data[good.start : good.end] for good in good_sections]
        print(name, appliance, len(samples))
        out_data[name][appliance] = pd.concat(samples, axis=0)

    return out_data


def parse_REDD(data_path: str, save_path: str) -> None:
    data_path: Path = Path(data_path).resolve()
    assert data_path.exists(), f"Path '{data_path}' does not exist!"

    dataset = load_redd_dataset(data_path / "redd.h5")
    prepared_data = data_preparation(dataset)

    # resample the data to 7s and fill the missing values with the previous value and convert to kWh
    for house in prepared_data:
        for meter in prepared_data[house]:
            df = prepared_data[house][meter]
            prepared_data[house][meter] = df.resample("7s").ffill(limit=1).fillna(0)

    save_to_pickle(prepared_data, save_path)
