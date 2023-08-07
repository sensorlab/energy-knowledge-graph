import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from nilmtk import DataSet
import pandas as pd

from helper_functions import save_to_pickle, watts2kwh



def load_dataset(path):
    try:
        dataset = DataSet(path)

        samples = []
        for building_idx, building in dataset.buildings.items():
            aggregate = next(building.elec.mains()[1].load()) + next(building.elec.mains()[1].load())
            for meter in building.elec.all_meters():

                data = list(meter.load())
                assert len(data) == 1

                assert len(meter.appliances) < 2

                # TODO: Poglej s kje jemlje sample Jakob.
                sample = (building_idx, list([a.type['type'] for a in meter.appliances]), data, meter.good_sections(), aggregate)

                samples.append(sample)
                
        return samples

    except Exception as e:
        dataset.store.close()
        raise e
        


def data_preparation(dataset):
    
    out_data = {}
    for (idx, appliances, data, good_sections, aggregate) in dataset:
        name = "IAWE_{}".format(idx)
        if name not in out_data:
           out_data[name] = {
                "aggregate": aggregate
           }
        if not appliances:
            continue
        
        
        appliance = appliances[0]
        data = data[0]
        if appliance == "unknown":
            continue
        samples = [data[good.start:good.end] for good in good_sections]
        print(name, appliance, len(samples))
        # X[appliance].extend(samples)
        df = pd.concat(samples, axis=0)
        out_data[name][appliance] = pd.concat(samples, axis=0)
        
    # for appliance, samples in X.items():
    #     print(appliance, len(samples))
        
    return out_data


def parse_IAWE(data_path : str, save_path : str):
    dataset = load_dataset(data_path +"iawe.h5")
    prepared_data = data_preparation(dataset)

    for house in prepared_data:
        for meter in prepared_data[house]:
            df = pd.DataFrame(prepared_data[house][meter]["power"]["active"])
            prepared_data[house][meter] = watts2kwh(df, 60/3600)


    save_to_pickle(prepared_data, save_path)