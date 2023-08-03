# Energy Knowledge Graph

The projects builds knowledge graph for energy consumption.

## Installation / Use

If PostgresML is already deployed on remote machine, go to step 3. If database is already populated with the data, go to step 6.

The scripts are described in more detail in the [scripts README](./scripts/README.md).


1. clone PostgresML [repository](https://github.com/postgresml/postgresml) `git clone https://github.com/postgresml/postgresml`
2. Navigate to postgresml directory `cd ./postgresml` and run `docker-compose up --build`
3. Start separate terminal and clone this [repository](https://github.com/sensorlab/energy-knowledge-graph) `git clone https://github.com/sensorlab/energy-knowledge-graph`
4. Navigate into energy-knowledge-graph directory, enter conda or virtualenv, and install dependecies with `pip install -e .`
5. Make sure that `./data/raw` contains required datasets and run script to preprocess the data `python ./scripts/parsers run_parsers.py <path to data> <path to save folder>`
5. Make sure that `./data/metadata` contains required datasets and run script to generate metadata `python ./scripts/generate_metadata <path to data> <path to save> [--save]`

6. Make sure that `./data/` contains required datasets and run script to populate database `python ./scripts/database-reset.py`
7. Access PostgreSQL at port `:5433`, PosgresML dashboard at port `:8000`, and PostgresML documentation at port `:8001`