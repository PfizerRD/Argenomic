import sys
import numpy as np
import pandas as pd
from typing import List, Tuple

from openeye import oechem, oeshape, oeomega

from rdkit import Chem
from rdkit import rdBase
from rdkit import RDConfig
rdBase.DisableLog('rdApp.error')

from rdkit.Chem import AllChem
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdmolfiles import MolToSmiles
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

class descriptor:
    """
    A strategy class for calculating the descriptor vector of a molecule.
    """
    def __init__(self, config_descriptor) -> None:
        self.properties = []
        self.ranges = config_descriptor.ranges
        self.property_names = config_descriptor.properties
        for name in self.property_names:
            if "." in name:
                module, fuction = name.split(".")
                module = getattr(sys.modules[__name__], module)
                self.properties.append(getattr(module, fuction))
            else:
                self.properties.append(name)
        return None

    def __call__(self, molecule: Chem.Mol) -> List[float]:
        """
        Calculating the descriptor vector of a molecule.
        """
        descriptor = []
        for property, range in zip(self.properties, self.ranges):
            descriptor.append(self.rescale(property(molecule), range))
        return descriptor

    @staticmethod
    def rescale(feature: List[float], range: List[float]) -> List[float]:
        """
        Rescaling the feature to the unit range.
        """
        rescaled_feature = (feature - range[0])/(range[1] - range[0])
        return rescaled_feature

class fitness:
    """
    A strategy class for calculating the fitness of a molecule.
    """
    def __init__(self, config_fitness, config_spec_params=None) -> None:
        self.memoized_cache = dict()
        if 'CFP' in config_fitness.type:
            self.fingerprint_type = config_fitness.type
            self.target = Chem.MolFromSmiles(config_fitness.target)
            self.target_fingerprint = self.get_fingerprint(self.target, self.fingerprint_type)
        elif 'ROCS' in config_fitness.type:
            self.param_dict = {
                'rocs_type': config_spec_params.rocs_type,
                'max_confs': config_spec_params.max_confs,
                'max_centers': config_spec_params.max_centers,
                'force_flip': config_spec_params.force_flip,
                'enum_nitrogen': config_spec_params.enum_nitrogen,
            }
            omegaOpts = oeomega.OEOmegaOptions()
            omegaOpts.SetMaxConfs(self.param_dict['max_confs'])
            self._omega = oeomega.OEOmega(omegaOpts)
            self._ref_overlay = self._calc_ref_overlay(config_fitness.target)
        return None

    def __call__(self, molecule: Chem.Mol) -> float:
        #print("Molecule,SMILES,ROCS_Score,Time", flush=True)
        fit_smi = MolToSmiles(molecule)
        try:
            if fit_smi in self.memoized_cache:
                fitness_score = self.memoized_cache[fit_smi]
            else:
                fit_list = self._get_enantiomers_from_smi(fit_smi)
                fitness_score = self._calc_rocs_score(fit_list)
                self.memoized_cache[fit_smi] = fitness_score
        except:
            fitness_score = 0
        return fitness_score

    def get_fingerprint(self, molecule: Chem.Mol, fingerprint_type: str):
        method_name = 'get_' + fingerprint_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception('{} is not a supported fingerprint type.'.format(fingerprint_type))
        return method(molecule)

    def get_ECFP4(self, molecule: Chem.Mol):
        return AllChem.GetMorganFingerprint(molecule, 2)

    def get_ECFP6(self, molecule: Chem.Mol):
        return AllChem.GetMorganFingerprint(molecule, 3)

    def get_FCFP4(self, molecule: Chem.Mol):
        return AllChem.GetMorganFingerprint(molecule, 2, useFeatures=True)

    def get_FCFP6(self, molecule: Chem.Mol):
        return AllChem.GetMorganFingerprint(molecule, 3, useFeatures=True)

    def _calc_ref_overlay(self, ref3d):
        ref_mol = self._get_ref_from_3D_file(ref3d)
        ref_overlay = self._create_shape_overlay(ref_mol)
        return ref_overlay

    def _get_ref_from_3D_file(self, ref3d):
        reffs = oechem.oemolistream(ref3d)
        reffs.SetConfTest(oechem.OEIsomericConfTest())  # read all conformers from input SDF file
        refmol = oechem.OEMol()
        oechem.OEReadMolecule(reffs, refmol)
        return refmol

    @classmethod
    def _create_shape_overlay(cls, ref_mol):
        prep = oeshape.OEOverlapPrep()
        prep.Prep(ref_mol)
        overlay = oeshape.OEMultiRefOverlay()
        overlay.SetupRef(ref_mol)
        return overlay

    def _get_enantiomers_from_mol(self, mol):
        mol_list = []
        for enant in oeomega.OEFlipper(mol, self.param_dict['max_centers'],
                                       self.param_dict['force_flip'],
                                       self.param_dict['enum_nitrogen']):
            enant = oechem.OEMol(enant)
            ret_code = self._omega.Build(enant)
            if ret_code == oeomega.OEOmegaReturnCode_Success:
                mol_list.append(oechem.OEMol(enant))
            else:
                raise ValueError("Unable to build enantiomers!")       
        return mol_list

    def _get_enantiomers_from_smi(self, smi):
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smi)
        try:
            mol_list = self._get_enantiomers_from_mol(mol)
        except:
            raise ValueError("Unable to build enantiomers!")       
        return mol_list

    def _calc_rocs_score(self, fit_list):
        best_score = -np.inf
        # available options: "shape_only" or "shape_and_color"
        rocs_type = self.param_dict['rocs_type']
        try:
            overlay = self._ref_overlay
            for fitmol in fit_list:
                prep = oeshape.OEOverlapPrep()
                prep.Prep(fitmol)
                score = oeshape.OEBestOverlayScore()
                overlay.BestOverlay(score, fitmol, oeshape.OEHighestTanimoto())
                if rocs_type == "shape_only" and score.GetTanimoto() > best_score:
                    best_score = score.GetTanimoto()
                elif rocs_type == "shape_and_color" and score.GetTanimotoCombo() > best_score:
                    best_score = score.GetTanimotoCombo()
                elif rocs_type not in ("shape_only", "shape_and_color"):
                    raise ValueError("Invalid ROCS score type!")
        except:
            raise ValueError("Unable to calculate ROCS score!")
        return best_score

