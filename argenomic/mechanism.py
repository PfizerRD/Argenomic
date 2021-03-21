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
from rdkit import DataStructs
from rdkit.Chem.rdmolfiles import MolToSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric, GetScaffoldForMol
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
            self._ref_smi = self._get_ref_smi()
            self._segment_thresholds = config.fitness.thresholds
            self._segment_penalty_weights = config.fitness.penalty_weights
            self._segment_smis = config.fitness.segment_smis
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
            print("ERROR in main _calc_rocs_score method --> result: fitness_score = 0", flush=True)
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
        ref_mol = self._get_ref_mol_from_3D_file()
        ref_overlay = self._create_shape_overlay(ref_mol)
        return ref_overlay

    def _get_ref_mol_from_3D_file(self):
        reffs = oechem.oemolistream(self._ref3d)
        reffs.SetConfTest(oechem.OEIsomericConfTest())  # read all conformers from input SDF file
        refmol = oechem.OEMol()
        oechem.OEReadMolecule(reffs, refmol)
        return refmol

    def _get_ref_smi(self):
        ref_mol = self._get_ref_mol_from_3D_file()
        ref_smi = oechem.OEMolToSmiles(ref_mol)
        return ref_smi

    @classmethod
    def _create_shape_overlay(cls, ref_mol):
        prep = oeshape.OEOverlapPrep()
        prep.Prep(ref_mol)
        overlay = oeshape.OEMultiRefOverlay()
        overlay.SetupRef(ref_mol)
        return overlay

    def _get_ref_mol_segments(self):
        _segment_smis = self._segment_smis

        print("Length of _segment_smis: {}".format(len(_segment_smis)))
        print("Type of _segment_smis: {}".format(type(_segment_smis)))

        _segment_mols = []
        unique = True
        ss = oechem.OESubSearch()

        # perform a substructure search to find the portion of the refmol
        for seg_smi in _segment_smis:
            if not ss.Init(seg_smi):
                oechem.OEThrow.Fatal("Unable to parse SMILES: {}".format(seg_smi))

        refmol = self._get_ref_mol_from_3D_file()
        oechem.OEPrepareSearch(refmol, ss)

        for seg_smi in _segment_smis:
            ss.Init(seg_smi)
            if not ss.SingleMatch(refmol):
                oechem.OEThrow.Fatal("SMILES fails to match refmol")

            for match in ss.Match(refmol, unique):
                # create match atom/bond set for target atoms
                abset = oechem.OEAtomBondSet()
                for ma in match.GetTargetAtoms():
                    abset.AddAtom(ma)

                for mb in match.GetTargetBonds():
                    abset.AddBond(mb)

                # create match subgraph
                subRef = oechem.OEMol()
                oechem.OESubsetMol(subRef, abset, True)

            _segment_mols.append(subRef)

        print("Length of _segment_mols: {}".format(len(_segment_mols)))
        print("Type of _segment_mols: {}".format(type(_segment_mols)))

        return _segment_mols

    def _calc_segment_overlays(self):
        segment_mols = self._get_ref_mol_segments()

        print("Length of segment_mols (for overlay): {}".format(len(segment_mols)))
        print("Type of segment_mols (for overlay): {}".format(type(segment_mols)))

        segment_overlays = self._create_shape_overlays_segments(segment_mols)

        print("Length of segment_overlays: {}".format(len(segment_overlays)))
        print("Type of segment_overlays: {}".format(type(segment_overlays)))

        return segment_overlays

    @classmethod
    def _create_shape_overlays_segments(cls, segment_mols):
        segment_overlays = []
        for seg_mol in segment_mols:
            prep = oeshape.OEOverlapPrep()
            prep.Prep(seg_mol)
            overlay = oeshape.OEMultiRefOverlay()
            overlay.SetupRef(seg_mol)
            segment_overlays.append(overlay)
        return segment_overlays

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

    def _calc_rocs_score(self, fit_confs):
        best_score = -np.inf
        # available options: "shape_only" or "shape_and_color"
        rocs_type = self.param_dict['rocs_type']
        try:
            overlay = self._calc_ref_overlay()
            best_index = -1
            for i, fitmol in enumerate(fit_confs):
                prep = oeshape.OEOverlapPrep()
                prep.Prep(fitmol)
                score = oeshape.OEBestOverlayScore()
                overlay.BestOverlay(score, fitmol, oeshape.OEHighestRefTverskyCombo())
                if rocs_type == "shape_only" and score.GetRefTversky() > best_score:
                    best_score = score.GetRefTversky()
                    best_index = i
                elif rocs_type == "shape_and_color" and score.GetRefTverskyCombo() > best_score:
                    best_score = score.GetRefTverskyCombo()
                    best_index = i
                elif rocs_type not in ("shape_only", "shape_and_color"):
                    raise ValueError("Invalid ROCS score type!")
            fitness_score = best_score

            print("fitness_score before segment penalties: {}".format(fitness_score), flush=True)

            print("FINISHED calculating whole Ref overlay.", flush=True)

            print("\nLength of self._segment_thresholds: {}".format(len(self._segment_thresholds)))
            print("Type of self._segment_thresholds: {}".format(type(self._segment_thresholds)))

            print("Length of self._segment_penalty_weights: {}".format(len(self._segment_penalty_weights)))
            print("Type of self._segment_penalty_weights: {}".format(type(self._segment_penalty_weights)))

            print("Length of self._segment_smis: {}".format(len(self._segment_smis)))
            print("Type of self._segment_smis: {}".format(type(self._segment_smis)))

            for k, seg in enumerate(self._segment_smis):
                print("segment {}: {}".format(k, seg), flush=True)
            
            #print("Length of self._segment_overlays: {}".format(len(self._segment_overlays)), flush=True)
            #print("Type of self._segment_overlays: {}\n".format(type(self._segment_overlays)))

            if len(self._segment_smis) > 0:
                print("BEFORE the segment loop.", flush=True)
                prep = oeshape.OEOverlapPrep()
                print("Created prep object.", flush=True)
                print("best_index: {}".format(best_index), flush=True)
                print("type of fit_confs: {}".format(type(fit_confs)), flush=True)
                print("fit_confs: {}\n".format(fit_confs), flush=True)
                #print("fit_confs(best_index): {}".format(fit_confs(best_index)), flush=True)
                #print("type of fit_confs(best_index): {}".format(type(fit_confs(best_index))), flush=True)

                for i, fitmol in enumerate(fit_confs):
                    if i == best_index:
                        prep.Prep(fitmol)
                        print("Added best conf to prep.", flush=True)
                        
                        for j, seg_overlay in enumerate(self._calc_segment_overlays()):
                            print("\nSTARTED the segment loop.", flush=True)
                            print("seg_overlay[{}]:  {}\n".format(j, seg_overlay), flush=True)
                            seg_score = oeshape.OEBestOverlayScore()
                            print("01. the segment loop.", flush=True)
                            seg_overlay.BestOverlay(seg_score, fitmol, oeshape.OEHighestRefTversky())
                            print("02. the segment loop.", flush=True)
                            seg_ref_Tversky_score = seg_score.GetRefTversky()
                            print("03. the segment loop.", flush=True)
                            print("seg_ref_Tversky_score {}: {}".format(j, seg_ref_Tversky_score), flush=True)

                            print("FINISHED calculating seg_ref_Tversky_score.", flush=True)

                            upperbound = 2

                            if seg_ref_Tversky_score > self._segment_thresholds[j]:
                                if seg_ref_Tversky_score > upperbound:
                                    seg_ref_Tversky_score = upperbound
                                penalty = (upperbound - seg_ref_Tversky_score) / (upperbound - self._segment_thresholds[j])
                                weighted_penalty = self._segment_penalty_weights[j] * penalty
                                fitness_score *= weighted_penalty
                        break

            print("fitness_score after segment penalties: {}".format(fitness_score), flush=True)

        except:
            raise ValueError("Unable to calculate ROCS score!")
        return max(0, fitness_score)
