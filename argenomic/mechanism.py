import sys
import numpy as np
import pandas as pd
from typing import List, Tuple
import time

from openeye import oechem, oeshape, oeomega, oemedchem

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
            module, fuction = name.split(".")
            module = getattr(sys.modules[__name__], module)
            self.properties.append(getattr(module, fuction))
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
    def __init__(self, config) -> None:
        self.memoized_cache = dict()
        if 'CFP' in config.fitness.type:
            self.fingerprint_type = config.fitness.type
            self.target = Chem.MolFromSmiles(config.fitness.target)
            self.target_fingerprint = self.get_fingerprint(self.target, self.fingerprint_type)
        elif 'ROCS' in config.fitness.type:
            self.param_dict = {
                'rocs_type': config.spec_params.rocs_type,
                'max_confs': config.spec_params.max_confs,
                'max_centers': config.spec_params.max_centers,
                'force_flip': config.spec_params.force_flip,
                'enum_nitrogen': config.spec_params.enum_nitrogen,
            }
            self._ref3d = config.fitness.target
            self._ref_topo_scaffold = self._get_Topo_BemisMurcko_framework_from_3d_ref()
            self._penalty_weight = config.penalty_weight
        return None

    def __call__(self, molecule: Chem.Mol) -> float:
        start_time = time.time()
        time_taken = 0
        fit_smi = Chem.MolToSmiles(molecule)
        try:
            if fit_smi in self.memoized_cache:
                fitness_score = self.memoized_cache[fit_smi]
            else:
                fit_confs = self._get_enantiomers_from_smi(fit_smi)
                fitness_score = self._calc_rocs_score(fit_confs)
                self.memoized_cache[fit_smi] = fitness_score
                time_taken = time.time() - start_time
        except:
            fitness_score = 0
        print("{},{},{}".format(fit_smi, round(fitness_score, 2), round(time_taken, 1)), flush=True)
        return fitness_score

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

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

    def _calc_ref_overlay(self):
        ref_mol = self._get_ref_from_3D_file()
        ref_overlay = self._create_shape_overlay(ref_mol)
        return ref_overlay

    def _get_ref_from_3D_file(self):
        reffs = oechem.oemolistream(self._ref3d)
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
            omegaOpts = oeomega.OEOmegaOptions()
            omegaOpts.SetMaxConfs(self.param_dict['max_confs'])
            _omega = oeomega.OEOmega(omegaOpts)
            ret_code = _omega.Build(enant)
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

    def _get_Topo_BemisMurcko_framework_from_3d_ref(self):
        ref_mol = self._get_ref_from_3D_file()
        ref_BM_framework = None
        adjustHCount = True
        for frag in oemedchem.OEGetBemisMurcko(ref_mol):
            fragment = oechem.OEGraphMol()
            oechem.OESubsetMol(fragment, ref_mol, frag, adjustHCount)
            if ".".join(r.GetName() for r in frag.GetRoles()) == 'Framework':
                oechem.OEUncolorMol(fragment)  # make scaffold generic
                ref_BM_framework = oechem.OEMolToSmiles(fragment)
        return ref_BM_framework

    @classmethod
    def _get_Topo_BemisMurcko_framework_from_fit_smi(cls, fit_smi):
        fit_mol = oechem.OEGraphMol()
        oechem.OESmilesToMol(fit_mol, fit_smi)
        fit_BM_framework = None
        adjustHCount = True
        for frag in oemedchem.OEGetBemisMurcko(fit_mol):
            fragment = oechem.OEGraphMol()
            oechem.OESubsetMol(fragment, fit_mol, frag, adjustHCount)
            if ".".join(r.GetName() for r in frag.GetRoles()) == 'Framework':
                oechem.OEUncolorMol(fragment)  # make scaffold generic
                fit_BM_framework = oechem.OEMolToSmiles(fragment)
        return fit_BM_framework

    def _get_num_atoms_in_ref_scaffold(self):
        #ref_scaffold = self._get_Topo_BemisMurcko_framework_from_3d_ref()
        ref_scaffold = self._ref_topo_scaffold
        mol = oechem.OEGraphMol()
        oechem.OEParseSmiles(mol, ref_scaffold)
        return mol.NumAtoms()

    def _get_num_atoms_max_common_substructure(self, fit_smi):
        ref_scaffold = self._ref_topo_scaffold
        fit_scaffold = self._get_Topo_BemisMurcko_framework_from_fit_smi(fit_smi)
        pattern = oechem.OEGraphMol()
        target = oechem.OEGraphMol()
        oechem.OESmilesToMol(pattern, ref_scaffold)
        oechem.OESmilesToMol(target, fit_scaffold)

        atomexpr = oechem.OEExprOpts_DefaultAtoms
        bondexpr = oechem.OEExprOpts_DefaultBonds

        mcss = oechem.OEMCSSearch(pattern, atomexpr, bondexpr, oechem.OEMCSType_Approximate)
        mcss.SetMCSFunc(oechem.OEMCSMaxAtoms())

        unique = True
        top_match_num_atoms = 0
        for match in mcss.Match(target, unique):
            if match.NumAtoms() > top_match_num_atoms:
                top_match_num_atoms = match.NumAtoms()
        return top_match_num_atoms

    def _calc_rocs_score(self, fit_confs):
        best_score = -np.inf
        # available options: "shape_only" or "shape_and_color"
        rocs_type = self.param_dict['rocs_type']
        penalty = 0
        try:
            overlay = self._calc_ref_overlay()
            for i, fitmol in enumerate(fit_confs):
                prep = oeshape.OEOverlapPrep()
                prep.Prep(fitmol)
                score = oeshape.OEBestOverlayScore()
                overlay.BestOverlay(score, fitmol, oeshape.OEHighestTanimoto())
                if self._penalty_weight != 0 and i == 0:
                    fit_smi = oechem.OEMolToSmiles(fitmol)
                    num_atoms_in_mcss = self._get_num_atoms_max_common_substructure(fit_smi)
                    num_atoms_in_ref_scaffold = self._get_num_atoms_in_ref_scaffold()
                    penalty = num_atoms_in_mcss / num_atoms_in_ref_scaffold
                if rocs_type == "shape_only" and score.GetTanimoto() > best_score:
                    best_score = score.GetTanimoto() - 0.5 * self._penalty_weight * penalty
                elif rocs_type == "shape_and_color" and score.GetTanimotoCombo() > best_score:
                    best_score = score.GetTanimotoCombo() - self._penalty_weight * penalty
                elif rocs_type not in ("shape_only", "shape_and_color"):
                    raise ValueError("Invalid ROCS score type!")
        except:
            raise ValueError("Unable to calculate ROCS score!")
        return max(0, best_score)
