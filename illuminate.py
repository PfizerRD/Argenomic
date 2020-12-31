import sys
import hydra
import pandas as pd
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import PandasTools as pdtl

from dask import bag
from dask.distributed import Client
from distributed.protocol import serialize, deserialize
import pickle

from argenomic.operations import crossover, mutator
from argenomic.mechanism import descriptor, fitness
from argenomic.infrastructure import archive, arbiter

class illumination:
    def __init__(self, config) -> None:
        self.data_file = config.data_file
        self.batch_size = config.batch_size
        self.initial_size = config.initial_size
        self.generations = config.generations

        self.mutator = mutator()
        self.crossover = crossover()
        self.arbiter = arbiter(config.arbiter)
        self.descriptor = descriptor(config.descriptor)
        self.archive = archive(config.archive, config.descriptor)
        self.fitness = fitness(config)

        logger.debug("serialize(self.descriptor): \n{}\n".format(serialize(self.descriptor)), flush=True)
        logger.debug("serialize(self.fitness): \n{}\n".format(serialize(self.fitness)), flush=True)
        logger.debug("pickle.dumps(self.fitness): \n{}\n".format(pickle.dumps(self.fitness)), flush=True)
        logger.debug("pickle.loads(pickle.dumps(self.fitness)): \n{}\n"
                     .format(pickle.loads(pickle.dumps(self.fitness))), flush=True)

        self.client = Client(n_workers=config.workers, threads_per_worker=config.threads)
        return None

    def __call__(self) -> None:
        logger.debug("initial_population() ...", flush=True)
        self.initial_population()
        print("FINISHED initial_population() ...", flush=True)
        for generation in range(self.generations):
            print("="*30 + "\nGeneration {}".format(generation+1), flush=True)
            print("SMILES,ROCS_Score,Time", flush=True)
            print("started generate_molecules() ...", flush=True)
            molecules = self.generate_molecules()
            molecules, descriptors, fitnesses = self.process_molecules(molecules)
            self.archive.add_to_archive(molecules, descriptors, fitnesses)
            self.archive.store_statistics(generation)
            self.archive.store_archive(generation)
        return None

    def initial_population(self) -> None:
        print("read data_file to dataframe ...", flush=True)
        dataframe = pd.read_csv(hydra.utils.to_absolute_path(self.data_file))
        print("add mol column to df ...", flush=True)
        pdtl.AddMoleculeColumnToFrame(dataframe, 'smiles', 'molecule')
        print("sample molecules from df ...", flush=True)
        molecules = dataframe['molecule'].sample(n=self.initial_size).tolist()
        print("filter unique_molecules using arbiter ...", flush=True)
        molecules = self.arbiter(self.unique_molecules(molecules))
        print("process molecules ...", flush=True)
        molecules, descriptors, fitnesses = self.process_molecules(molecules)
        print("FINISHED processing molecules ...", flush=True)
        print("add molecules to archive ...", flush=True)
        self.archive.add_to_archive(molecules, descriptors, fitnesses)
        return None

    def generate_molecules(self) -> None:
        molecules = []
        print("sample molecules from archive ...", flush=True)
        sample_molecules = self.archive.sample(self.batch_size)
        print("sample molecule-pairs from archive ...", flush=True)
        sample_molecule_pairs = self.archive.sample_pairs(self.batch_size)
        for molecule in sample_molecules:
            print("apply mutation to molecules from archive ...", flush=True)
            molecules.extend(self.mutator(molecule))
        for molecule_pair in sample_molecule_pairs:
            print("apply crossover to molecule-pairs from archive ...", flush=True)
            molecules.extend(self.crossover(molecule_pair))
        print("keep molecules from archive which pass all filters ...", flush=True)
        molecules = self.arbiter(self.unique_molecules(molecules))
        return molecules

    def process_molecules(self, molecules: List[Chem.Mol]) -> Tuple[List[List[float]],List[float]]:
        print("="*30 + "\nprocess_molecules\n", flush=True)
        print("calculate descriptors using dask.bag ...", flush=True)
        descriptors = bag.map(self.descriptor, bag.from_sequence(molecules)).compute()
        print("check if all descriptors pass requirements ...", flush=True)
        molecules, descriptors = zip(*[(molecule, descriptor) for molecule, descriptor in zip(molecules, descriptors)\
                if all(1.0 > property > 0.0 for property in descriptor)])
        print("convert mols and descriptors into two lists ...", flush=True)
        molecules, descriptors = list(molecules), list(descriptors)
        print("calculate fitness of molecuels using dask.bag ...", flush=True)
        fitnesses = bag.map(self.fitness, bag.from_sequence(molecules)).compute()
        print("FINISHED calculating fitness of molecuels using dask.bag ...", flush=True)
        return molecules, descriptors, fitnesses

    @staticmethod
    def unique_molecules(molecules: List[Chem.Mol]) -> List[Chem.Mol]:
        molecules = [Chem.MolFromSmiles(Chem.MolToSmiles(molecule)) for molecule in molecules if molecule is not None]
        molecule_records = [(molecule, Chem.MolToSmiles(molecule)) for molecule in molecules if molecule is not None]
        molecule_dataframe = pd.DataFrame(molecule_records, columns = ['molecules', 'smiles'])
        molecule_dataframe.drop_duplicates('smiles', inplace = True)
        return molecule_dataframe['molecules']


@hydra.main(config_path="configuration", config_name="config.yaml")
def launch(config) -> None:
    print(config.pretty())
    if config.path_to_remove is not None and config.path_to_remove in sys.path:
        sys.path.remove(config.path_to_remove)
    current_instance = illumination(config)
    current_instance()
    current_instance.client.close()

if __name__ == "__main__":
    launch()