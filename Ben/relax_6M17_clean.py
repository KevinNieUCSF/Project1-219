import os
import re
import pandas as pd
import pyrosetta
from pyrosetta import *
pyrosetta.init('''
               -ex1
               -ex2
               -score:weights ref2015_cart
               -use_input_sc
               -ignore_unrecognized_res           
               ''')
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task import operation
from pyrosetta.rosetta.core.select import residue_selector


pdb_file = '../Kevin/6M17_clean.pdb'

scorefxn = create_score_function('ref2015_cart')
pose = pose_from_pdb(pdb_file)
pose.pdb_info().name('6M17_clean')
pose_relaxed = pose.clone()
pose_relaxed.pdb_info().name('6M17_relaxed')

# to run FastRelax, make a task factory, a movemap factory, and add them to a FastRelax protocol object

# Make task factory
tf = TaskFactory()
tf.push_back(operation.InitializeFromCommandline())
tf.push_back(operation.IncludeCurrent())
tf.push_back(operation.NoRepackDisulfides())
tf.push_back(operation.OperateOnResidueSubset(
        operation.RestrictToRepackingRLT(),
        residue_selector.TrueResidueSelector()))

# make movemap factory
mmf = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
mmf.all_bb(setting=True)
mmf.all_bondangles(setting=True)
mmf.all_bondlengths(setting=True)
mmf.all_chi(setting=True)
mmf.all_jumps(setting=True)
mmf.set_cartesian(setting=True)

# make fastrelax protocol
fr = pyrosetta.rosetta.protocols.relax.FastRelax(standard_repeats=5)
fr.set_scorefxn(scorefxn)
fr.set_task_factory(tf)
fr.set_movemap_factory(mmf)
fr.cartesian(True)
fr.min_type("lbfgs_armijo_nonmonotone")

# run fastrelax and dump pose
fr.apply(pose_relaxed)
pose_relaxed.dump_pdb('./6M17_relaxed.pdb')