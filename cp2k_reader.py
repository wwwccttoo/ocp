import os
import re
from itertools import chain

import ase
import ase.io
import numpy as np
from ase.constraints import FixAtoms
from monty.io import zopen
from pycp2k import CP2K
from pymatgen.core.units import Ha_to_eV
from pymatgen.io.cp2k.outputs import Cp2kOutput
from pymatgen.io.cp2k.utils import postprocessor


class Cp2kResHold:
    # this is a class for single atom (not trajectory)
    def __init__(self, outfile, xyzfile):
        self.outfile_address = outfile
        self.xyzfile_address = xyzfile
        self.outfile = Cp2kOutput(outfile)
        self.xyzfile = ase.io.read(xyzfile)
        # parse the energy, note it will automatically read the energy from the trajectory file
        # for static_run, it will read the energy from main output file instead
        self.outfile.parse_energies()
        # parse the forces from the force file or from the main file
        # the energy is in the unit of Ha/Bohr, which should be converted to eV/Angstrom in ase
        self.outfile.parse_forces()
        # setup a simple calculator for assigning forces
        self.delete_proton = False

        self.atoms_to_keep = []

    def set_calculator(self):
        self.calc = ase.calculators.singlepoint.SinglePointDFTCalculator(
            self.xyzfile
        )
        converted_forces = (
            np.array(self.outfile.data["forces"])
            * ase.units.Hartree
            / ase.units.Bohr
        )
        self.calc.results = {
            "energy": self.outfile.final_energy,
            "forces": converted_forces[0][self.atoms_to_keep]
            if len(converted_forces.shape) == 3
            else converted_forces[self.atoms_to_keep],
        }
        self.xyzfile._calc = self.calc

    def set_constraints_and_cell(self, inputfile, delete_proton=False):
        # set the constraints of the ASE objective
        # make sure to set the cell and pbc first, otherwist ASE will think the molecule changes and decline getting energy and forces
        self.inputfile = CP2K()
        self.inputfile.parse(inputfile)
        # extract constraints
        c = FixAtoms(
            indices=[
                int(_) - 1
                for _ in self.inputfile.CP2K_INPUT.MOTION.CONSTRAINT.FIXED_ATOMS_list[
                    0
                ]
                .List[0]
                .split(" ")
            ]
        )
        self.xyzfile.set_constraint(c)
        # extract cell lengths
        cell_ABC = self.inputfile.CP2K_INPUT.FORCE_EVAL_list[0].SUBSYS.CELL.Abc
        cell_ABC = [float(_) for _ in cell_ABC.split(" ")]
        # extract cell angles
        cell_angles = self.inputfile.CP2K_INPUT.FORCE_EVAL_list[
            0
        ].SUBSYS.CELL.Alpha_beta_gamma
        cell_angles = [float(_) for _ in cell_angles.split(" ")]
        # extract cell pbc
        cell_pbc = [0, 0, 0]
        for _ in ["X", "Y", "Z"]:
            if (
                _
                in self.inputfile.CP2K_INPUT.FORCE_EVAL_list[
                    0
                ].SUBSYS.CELL.Periodic
            ):
                cell_pbc[ord(_) - ord("X")] = 1
        self.xyzfile.set_cell(cell_ABC + cell_angles)
        self.xyzfile.set_pbc(cell_pbc)
        self.atoms_to_keep = []
        if delete_proton:
            self.delete_proton = True
            atoms_to_del = []
            for atom_id_to_del, atom in enumerate(self.xyzfile):
                if atom.symbol == "X":
                    atoms_to_del.append(atom_id_to_del)
                else:
                    self.atoms_to_keep.append(atom_id_to_del)
            del self.xyzfile[atoms_to_del]
        else:
            self.atoms_to_keep = list(range(len(self.xyzfile)))


def building_of_one_metal(metal_address, delete_proton=False):
    try:
        base_case = Cp2kResHold(
            metal_address + "/base/out.txt", metal_address + "/base/test.xyz"
        )
    except Exception:
        print(
            "Failed to find base (pure surface case), search them in the parent folder..."
        )
        if "charge" in metal_address:
            levels = metal_address.split("/")
            for l_id, l in enumerate(levels):
                if "charge" in l:
                    levels[l_id] = "charge_inputs_ori"
            base_case = Cp2kResHold(
                "/".join(levels) + "/base/out.txt",
                "/".join(levels) + "/base/test.xyz",
            )
            print("Find the base case! It is " + "/".join(levels) + ".")
        elif "neutral" in metal_address:
            levels = metal_address.split("/")
            for l_id, l in enumerate(levels):
                if "neutral" in l:
                    levels[l_id] = "neutral_inputs_ori"
            base_case = Cp2kResHold(
                "/".join(levels) + "/base/out.txt",
                "/".join(levels) + "/base/test.xyz",
            )
            print("Find the base case! It is " + "/".join(levels) + ".")
        else:
            print(
                f"Cannot analyse {metal_address}, please check it. Skipping it..."
            )
            return []

    base_case.set_calculator()
    base_atom = base_case.xyzfile
    refer_energy = base_atom.get_potential_energy()
    structure_list = []
    for f in os.scandir(metal_address):
        levels = f.path.split("/")
        if (
            "DS_Store" not in levels[-1]
            and "." not in levels[-1]
            and levels[-1] != "base"
        ):
            try:
                # locate and retrieve the energy of the molecule
                cur_molecule = Cp2kResHold(
                    "/".join(levels[:-3]) + "/gas/" + levels[-1] + "/out.txt",
                    "/".join(levels[:-3]) + "/gas/" + levels[-1] + "/test.xyz",
                )
                cur_molecule.set_calculator()
                cur_molecule_atom = cur_molecule.xyzfile

                cur_comb = Cp2kResHold(
                    f.path + "/out.txt", f.path + "/test.xyz"
                )
                cur_comb.set_constraints_and_cell(
                    f.path + "/in.al2o3", delete_proton=delete_proton
                )
                cur_comb.set_calculator()
                cur_atom = cur_comb.xyzfile
                cur_atom._calc.results["energy"] = (
                    cur_atom._calc.results["energy"]
                    - refer_energy
                    - cur_molecule_atom.get_potential_energy()
                )
                structure_list.append(cur_atom)
            except FileNotFoundError:
                print("Please check this file: " + metal_address + "!")
                print(
                    f"Ignoring the structure with adsorbed molecule of {levels[-1]} due to the missing of output file!"
                )
    return structure_list


def gather_all_data(store_address, include_metal_species):
    ans = []
    for metal in include_metal_species:
        for f in os.scandir(store_address):
            if f.is_dir() and "." not in f.path and "gas" not in f.path:
                ans = ans + building_of_one_metal(f.path + "/" + metal)
    return ans


class Cp2kTrajHold:
    # this is a class for trajectory
    def __init__(self, outfile, xyzfile, init_xyzfile=None):
        self.outfile_address = outfile
        self.xyzfile_address = xyzfile
        self.init_xyzfile = init_xyzfile
        self.outfile = Cp2kOutput(outfile)
        # if it is a trajectory file, the final one is redundant.
        self.xyzfile = ase.io.read(xyzfile, index=":")[:-1]
        if init_xyzfile:
            self.xyzfile = [ase.io.read(init_xyzfile)] + self.xyzfile
        # set the proton symbol
        for xyz in self.xyzfile:
            for atom in xyz:
                if atom.position[-1] > 30:
                    atom.symbol = "X"
        # parse the energy, note it will automatically read the energy from the trajectory file
        # for static_run, it will read the energy from main output file instead
        self.outfile.parse_energies()
        # parse the forces from the force file or from the main file
        # the energy is in the unit of Ha/Bohr, which should be converted to eV/Angstrom in ase
        self.outfile.parse_forces()
        # check each SCF convergence
        self.outfile.convergence()
        # re-extract the energies
        self.extract_energies()
        # re-extract the forces
        self.extract_forces()
        # setup a simple calculator for assigning forces
        self.delete_proton = False

        self.atoms_to_keep = []

    def set_calculator(self):
        # check if there is the initial_xyz file
        if self.init_xyzfile:
            energies = self.outfile.data["total_energy"]
            forces = self.outfile.data["forces"]
            # this is reversed
            convergence = self.outfile.data["scf_converged"][::-1]
            # check if the Geopt is converged, this is also reversed for multiply file (unsupported)

        else:
            # if no initial_xyz is found, the first element should be dropped
            # this is corresponding to the initial file
            energies = self.outfile.data["total_energy"][1:]
            forces = self.outfile.data["forces"][1:]
            convergence = self.outfile.data["scf_converged"][::-1][1:]
            # check if the Geopt is converged, this is also reversed for multiply file (unsupported)

        self.converged_xyzfile = []
        # print(self.xyzfile_address)
        # print(len(forces))
        # print(len(self.xyzfile))
        for i in range(len(self.xyzfile)):
            if not convergence[i]:
                continue
            calc = ase.calculators.singlepoint.SinglePointDFTCalculator(
                self.xyzfile[i]
            )
            converted_forces = (
                np.array(forces[i]) * ase.units.Hartree / ase.units.Bohr
            )
            calc.results = {
                "energy": energies[i],
                "forces": converted_forces[0][self.atoms_to_keep]
                if len(converted_forces.shape) == 3
                else converted_forces[self.atoms_to_keep],
            }
            self.xyzfile[i]._calc = calc
            self.converged_xyzfile.append(self.xyzfile[i])

    def set_constraints_and_cell(self, inputfile, delete_proton=False):
        # this should be called earlier than set_calculator
        # set the constraints of the ASE objective
        # make sure to set the cell and pbc first, otherwist ASE will think the molecule changes and decline getting energy and forces
        self.inputfile = CP2K()
        self.inputfile.parse(inputfile)
        # extract constraints
        c = FixAtoms(
            indices=[
                int(_) - 1
                for _ in self.inputfile.CP2K_INPUT.MOTION.CONSTRAINT.FIXED_ATOMS_list[
                    0
                ]
                .List[0]
                .split(" ")
            ]
        )
        for xyz in self.xyzfile:
            xyz.set_constraint(c)
        # extract cell lengths
        cell_ABC = self.inputfile.CP2K_INPUT.FORCE_EVAL_list[0].SUBSYS.CELL.Abc
        cell_ABC = [float(_) for _ in cell_ABC.split(" ")]
        # extract cell angles
        cell_angles = self.inputfile.CP2K_INPUT.FORCE_EVAL_list[
            0
        ].SUBSYS.CELL.Alpha_beta_gamma
        cell_angles = [float(_) for _ in cell_angles.split(" ")]
        # extract cell pbc
        cell_pbc = [0, 0, 0]
        for _ in ["X", "Y", "Z"]:
            if (
                _
                in self.inputfile.CP2K_INPUT.FORCE_EVAL_list[
                    0
                ].SUBSYS.CELL.Periodic
            ):
                cell_pbc[ord(_) - ord("X")] = 1
        for xyz in self.xyzfile:
            xyz.set_cell(cell_ABC + cell_angles)
            xyz.set_pbc(cell_pbc)
        self.atoms_to_keep = []
        if delete_proton:
            self.delete_proton = True
            atoms_to_del = []
            for atom_id_to_del, atom in enumerate(self.xyzfile[0]):
                if atom.symbol == "X":
                    atoms_to_del.append(atom_id_to_del)
                else:
                    self.atoms_to_keep.append(atom_id_to_del)
            for xyz in self.xyzfile:
                del xyz[atoms_to_del]
        else:
            self.atoms_to_keep = list(range(len(self.xyzfile[0])))

    def extract_energies(self):
        with zopen(self.outfile_address, "rt") as f:
            text = f.read()
        toten_pattern = re.compile(r"Total FORCE_EVAL.*\s(-?\d+.\d+)")
        energies = []
        for mt in toten_pattern.finditer(text):
            d = mt.groupdict()
            if len(d) > 0:
                processed_line = {k: postprocessor(v) for k, v in d.items()}
            else:
                processed_line = [postprocessor(v) for v in mt.groups()]

            energies.append(processed_line)
        self.outfile.data["total_energy"] = energies[:]
        self.outfile.data["total_energy"] = list(
            chain.from_iterable(
                np.multiply(
                    self.outfile.data.get("total_energy", []), Ha_to_eV
                )
            )
        )

    def extract_forces(self):
        with zopen(self.outfile_address, "rt") as f:
            text = f.read()
        header_pattern = r"ATOMIC FORCES.+Z"
        row_pattern = (
            r"\s+\d+\s+\d+\s+\w+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
        )
        footer_pattern = r"SUM OF ATOMIC FORCES"
        table_pattern_text = (
            header_pattern
            + r"\s*^(?P<table_body>(?:\s+"
            + row_pattern
            + r")+)\s+"
            + footer_pattern
        )
        table_pattern = re.compile(
            table_pattern_text, re.MULTILINE | re.DOTALL
        )
        rp = re.compile(row_pattern)
        tables = []
        for mt in table_pattern.finditer(text):
            section_text = (
                mt.group()
            )  # or mt.group('table_body') if you've named this capturing group
            for row_mt in rp.finditer(section_text):
                d = row_mt.groupdict()
                if len(d) > 0:
                    processed_line = {
                        k: postprocessor(v) for k, v in d.items()
                    }
                else:
                    processed_line = [
                        postprocessor(v) for v in row_mt.groups()
                    ]
                tables.append(processed_line)
        num_SCF = len(self.outfile.data["scf_converged"])
        num_atoms = len(self.outfile.data["forces"][0])
        # it should be num_SCF - 1
        # this is caused by the code itself, which reveals the minimum energy if the geopt is converged
        # we do not have to do the check if the geopt is not converged
        if not self.outfile.data["geo_opt_not_converged"]:
            assert (num_SCF - 1) * num_atoms == len(
                tables
            ), f"Check your file! outfile address is {self.outfile_address}"
            forces = []
            for i in range(num_SCF - 1):
                forces.append(tables[num_atoms * i : num_atoms * (i + 1)][:])
        else:
            print("This geometry optimization is not converged!")
            print(f"Check {self.outfile_address}!!!")
            assert (num_SCF) * num_atoms == len(
                tables
            ), f"Check your file! outfile address is {self.outfile_address}"
            forces = []
            for i in range(num_SCF):
                forces.append(tables[num_atoms * i : num_atoms * (i + 1)][:])
        self.outfile.data["forces"] = forces[:]

    def extract_mulliken(self):
        with zopen(self.outfile_address, "rt") as f:
            text = f.read()
        header_pattern = r"Mulliken Population Analysis"
        row_pattern = r"\s+(\d+)\s+(\w+)\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
        footer_pattern = r"Total charge and spin"
        # First, extract the entire block of the table
        table_blocks_pattern = f"{header_pattern}.*?{footer_pattern}"
        table_blocks = re.findall(table_blocks_pattern, text, flags=re.DOTALL)

        # Then, for each block, extract the rows
        tables = []
        for block in table_blocks:
            rows = re.findall(row_pattern, block, flags=re.MULTILINE)
            for row in rows:
                tables.append(row)  # Process each row as needed
        num_SCF = len(self.outfile.data["scf_converged"])
        num_atoms = len(self.outfile.data["forces"][0])
        # it should be num_SCF - 1
        # this is caused by the code itself, which reveals the minimum energy if the geopt is converged
        # we do not have to do the check if the geopt is not converged
        if not self.outfile.data["geo_opt_not_converged"]:
            assert (num_SCF) * num_atoms == len(
                tables
            ), f"Check your file! outfile address is {self.outfile_address}"
            mulliken = []
            for i in range(num_SCF):
                mulliken.append(tables[num_atoms * i : num_atoms * (i + 1)][:])
        else:
            print("This geometry optimization is not converged!")
            print(f"Check {self.outfile_address}!!!")
            assert (num_SCF) * num_atoms == len(
                tables
            ), f"Check your file! outfile address is {self.outfile_address}"
            mulliken = []
            for i in range(num_SCF):
                mulliken.append(tables[num_atoms * i : num_atoms * (i + 1)][:])
        self.outfile.data["mulliken"] = mulliken[:]

    def extract_hirshfeld(self):
        with zopen(self.outfile_address, "rt") as f:
            text = f.read()
        header_pattern = r"Hirshfeld Charges"
        row_pattern = r"\s+(\d+)\s+(\w+)\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
        footer_pattern = r"Total Charge"
        # First, extract the entire block of the table
        table_blocks_pattern = f"{header_pattern}.*?{footer_pattern}"
        table_blocks = re.findall(table_blocks_pattern, text, flags=re.DOTALL)

        # Then, for each block, extract the rows
        tables = []
        for block in table_blocks:
            rows = re.findall(row_pattern, block, flags=re.MULTILINE)
            for row in rows:
                tables.append(row)  # Process each row as needed
        num_SCF = len(self.outfile.data["scf_converged"])
        num_atoms = len(self.outfile.data["forces"][0])
        # it should be num_SCF - 1
        # this is caused by the code itself, which reveals the minimum energy if the geopt is converged
        # we do not have to do the check if the geopt is not converged
        if not self.outfile.data["geo_opt_not_converged"]:
            assert (num_SCF) * num_atoms == len(
                tables
            ), f"Check your file! outfile address is {self.outfile_address}"
            hirshfeld = []
            for i in range(num_SCF):
                hirshfeld.append(
                    tables[num_atoms * i : num_atoms * (i + 1)][:]
                )
        else:
            print("This geometry optimization is not converged!")
            print(f"Check {self.outfile_address}!!!")
            assert (num_SCF) * num_atoms == len(
                tables
            ), f"Check your file! outfile address is {self.outfile_address}"
            hirshfeld = []
            for i in range(num_SCF):
                hirshfeld.append(
                    tables[num_atoms * i : num_atoms * (i + 1)][:]
                )
        self.outfile.data["hirshfeld"] = hirshfeld[:]


def building_of_one_metal_traj(
    metal_address, molecule_address, delete_proton=False
):
    base_case = Cp2kOutput(metal_address + "/base/out.txt")
    base_case.parse_energies()
    refer_energy = base_case.final_energy
    structure_list = []
    for f in os.scandir(metal_address):
        levels = f.path.split("/")
        if (
            "DS_Store" not in levels[-1]
            and "." not in levels[-1]
            and levels[-1] != "base"
        ):
            try:
                # locate and retrieve the energy of the molecule
                cur_molecule = Cp2kOutput(
                    molecule_address + "/" + levels[-1] + "/out.txt"
                )
                cur_molecule.parse_energies()
                cur_molecule_energy = cur_molecule.final_energy

                cur_comb = Cp2kTrajHold(
                    f.path + "/out.txt",
                    f.path + "/Al2O3-pos-1.xyz",
                    f.path + "/test.xyz",
                )
                cur_comb.set_constraints_and_cell(
                    f.path + "/in.al2o3", delete_proton=delete_proton
                )
                cur_comb.set_calculator()
                cur_xyzfile = cur_comb.converged_xyzfile
                for cur_atom in cur_xyzfile:
                    cur_atom._calc.results["energy"] = (
                        cur_atom._calc.results["energy"]
                        - refer_energy
                        - cur_molecule_energy
                    )
                    structure_list.append(cur_atom)
            except FileNotFoundError:
                print("Please check this file: " + metal_address + "!")
                print(
                    f"Ignoring the structure with adsorbed molecule of {levels[-1]} due to the missing of output file!"
                )
    return structure_list


def gather_all_data_traj(store_address, include_metal_species):
    ans = []
    for metal in include_metal_species:
        for f in os.scandir(store_address):
            if f.is_dir() and "." not in f.path and "gas" not in f.path:
                ans = ans + building_of_one_metal_traj(f.path + "/" + metal)
    return ans


def building_of_one_metal_special(
    metal_res_address,
    base_res_address,
    molecule_res_address,
    molecule_occur=1,
    delete_proton=False,
):
    base_case = Cp2kOutput(base_res_address)
    base_case.parse_energies()
    refer_energy = base_case.final_energy
    cur_molecule = Cp2kOutput(molecule_res_address)
    cur_molecule.parse_energies()
    cur_molecule_energy = cur_molecule.final_energy

    cur_comb = Cp2kResHold(
        metal_res_address + "/out.txt", metal_res_address + "/test.xyz"
    )
    cur_comb.set_constraints_and_cell(
        metal_res_address + "/in.al2o3", delete_proton=delete_proton
    )
    cur_comb.set_calculator()
    print(molecule_occur)
    cur_comb.xyzfile._calc.results["energy"] = (
        cur_comb.xyzfile._calc.results["energy"]
        - refer_energy
        - molecule_occur * cur_molecule_energy
    )
    return cur_comb.xyzfile


def building_of_one_metal_special_traj(
    metal_res_address,
    base_res_address,
    molecule_res_address,
    molecule_occur=1,
    delete_proton=False,
):
    base_case = Cp2kOutput(base_res_address + "/out.txt")
    base_case.parse_energies()
    refer_energy = base_case.final_energy
    structure_list = []
    try:
        # locate and retrieve the energy of the molecule
        cur_molecule = Cp2kOutput(molecule_res_address + "/out.txt")
        cur_molecule.parse_energies()
        cur_molecule_energy = cur_molecule.final_energy

        cur_comb = Cp2kTrajHold(
            metal_res_address + "/out.txt",
            metal_res_address + "/Al2O3-pos-1.xyz",
            metal_res_address + "/test.xyz",
        )
        cur_comb.set_constraints_and_cell(
            metal_res_address + "/in.al2o3", delete_proton=delete_proton
        )
        cur_comb.set_calculator()
        cur_xyzfile = cur_comb.converged_xyzfile
        for cur_atom in cur_xyzfile:
            cur_atom._calc.results["energy"] = (
                cur_atom._calc.results["energy"]
                - refer_energy
                - cur_molecule_energy * molecule_occur
            )
            structure_list.append(cur_atom)
    except FileNotFoundError:
        print("Please check this file: " + metal_res_address + "!")
        print(
            f"Ignoring the structure with adsorbed molecule of {metal_res_address} due to the missing of output file!"
        )
    return structure_list
