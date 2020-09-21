"""
Copyright 2020 Quaranta Roberto

This file is part of Simulation_of_plant_growth_and_features_extractions.

Simulation_of_plant_growth_and_features_extractions is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Simulation_of_plant_growth_and_features_extractions is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Simulation_of_plant_growth_and_features_extractions.  If not, see <https://www.gnu.org/licenses/>.
"""
"""
The purpose of this script is to create the plant object before the simulation
As the xml examples are not well writen, creating plant characteristics by changing attribute of a default plant
is easier
"""
import plantbox as pb  # CPlantBox Python Binding
import numpy as np


def create_basil():
	plant = pb.Plant()
	p0 = pb.RootRandomParameter(plant)  # with default values,
	p1 = pb.RootRandomParameter(plant)  # all standard deviations are 0
	s2 = pb.StemRandomParameter(plant)
	s3 = pb.StemRandomParameter(plant)
	l1 = pb.LeafRandomParameter(plant)

	p0.name = "taproot"
	p0.a = 0.2  # [cm] radius
	p0.subType = 1  # [-] index starts at 1
	p0.lb = 5  # [cm] basal zone
	p0.la = 10  # [cm] apical zone
	p0.lmax = 30  # [cm] maximal root length, number of lateral branching nodes = round((lmax-lb-la)/ln) + 1
	p0.ln = 1.  # [cm] inter-lateral distance (16 branching nodes)
	p0.theta = 0.  # [rad]
	p0.r = 1  # [cm/day] initial growth rate
	p0.dx = 10  # [cm] axial resolution
	p0.successor = [2]  # add successors
	p0.successorP = [1]  # probability that successor emerges
	p0.tropismT = pb.TropismType.gravi  #
	p0.tropismN = 1.8  # [-] strength of tropism
	p0.tropismS = 0.2  # [rad/cm] maximal bending

	p1.name = "lateral"
	p1.a = 0.1  # [cm] radius
	p1.subType = 2  # [1] index starts at 1
	p1.lmax = 15  # # [cm] apical zone
	p1.lmaxs = 0.15  # [cm] standard deviation of the apical zone
	p1.theta = 90. / 180. * np.pi  # [rad]
	p1.r = 2  # initial growth rate
	p1.dx = 1  # [cm] axial resolution
	p1.tropismT = pb.TropismType.gravi  # exo
	p1.tropismN = 2  # [-] strength of tropism
	p1.tropismS = 0.1  # [rad/cm] maximal bending

	s2.name = "mainstem"
	s2.subType = 1
	s2.lmax = 25
	# s2.lmaxs = 1
	s2.lb = 3
	s2.la = 0
	s2.ln = 3  # This value is normally the Inter-lateral distance [cm], but with decussate plant, this value is
	# multiplied by 2
	s2.lnf = 5  # This value means "successors in a decussate position"
	s2.RotBeta = 1
	s2.BetaDev = 0
	s2.InitBeta = 0
	s2.gf = 1
	s2.successor = [2]
	s2.successorP = [1]
	s2.tropismT = 4
	# s2.theta = 1/6
	s2.theta = 0
	s2.tropismN = 18
	s2.tropismS = 0.01

	s3.name = "invisible"  # The invisible stem representing the bud of the leaf
	s3.subType = 2
	s3.la = 0
	s3.ln = 0
	s3.lmax = 5
	s3.RotBeta = 1
	s3.BetaDev = 0
	s3.InitBeta = 0.5
	s3.tropismS = 0
	s3.lnf = 5

	l1.name = 'basil'
	l1.subType = 2
	l1.lb = 2
	l1.la = 0.2
	l1.lmax = 5
	l1.r = 0.5
	l1.RotBeta = 0.5
	l1.BetaDev = 0
	l1.InitBeta = 0.5
	l1.tropismT = 1
	l1.tropismN = 5
	l1.tropismS = 0.1
	l1.theta = 0.35
	l1.thetas = 0.05
	l1.gf = 1
	l1.lnf = 5  # If not precised, it will not be possible to do the opposite decussate arrangement

	plant.setOrganRandomParameter(p0)
	plant.setOrganRandomParameter(p1)
	plant.setOrganRandomParameter(s2)
	plant.setOrganRandomParameter(s3)
	plant.setOrganRandomParameter(l1)

	srp = pb.SeedRandomParameter(plant)  # with default values
	# srp.seedPos = pb.Vector3d(10, 10, -3.)  # [cm] seed position
	srp.seedPos = pb.Vector3d(0, 0, -3.)  # [cm] seed position
	srp.maxB = 0  # [-] number of basal roots (neglecting basal roots and shoot borne)
	srp.firstB = 10.  # [day] first emergence of a basal root
	srp.delayB = 3.  # [day] delay between the emergence of basal roots
	plant.setOrganRandomParameter(srp)

	return plant


def create_arabidopsis():
	plant = pb.Plant()
	p0 = pb.RootRandomParameter(plant)  # with default values,
	p1 = pb.RootRandomParameter(plant)  # all standard deviations are 0
	s1 = pb.StemRandomParameter(plant)
	s2 = pb.StemRandomParameter(plant)
	s3 = pb.StemRandomParameter(plant)
	s4 = pb.StemRandomParameter(plant)
	l1 = pb.LeafRandomParameter(plant)

	p0.name = "taproot"
	p0.a = 0.2  # [cm] radius
	p0.subType = 1  # [-] index starts at 1
	p0.lb = 5  # [cm] basal zone
	p0.la = 10  # [cm] apical zone
	p0.lmax = 30  # [cm] maximal root length, number of lateral branching nodes = round((lmax-lb-la)/ln) + 1
	p0.ln = 1.  # [cm] inter-lateral distance (16 branching nodes)
	p0.theta = 0.  # [rad]
	p0.r = 1  # [cm/day] initial growth rate
	p0.dx = 10  # [cm] axial resolution
	p0.successor = [2]  # add successors
	p0.successorP = [1]  # probability that successor emerges
	p0.tropismT = pb.TropismType.gravi  #
	p0.tropismN = 1.8  # [-] strength of tropism
	p0.tropismS = 0.2  # [rad/cm] maximal bending

	p1.name = "lateral"
	p1.a = 0.1  # [cm] radius
	p1.subType = 2  # [1] index starts at 1
	p1.lmax = 15  # # [cm] apical zone
	p1.lmaxs = 0.15  # [cm] standard deviation of the apical zone
	p1.theta = 90. / 180. * np.pi  # [rad]
	p1.r = 2  # initial growth rate
	p1.dx = 1  # [cm] axial resolution
	p1.tropismT = pb.TropismType.gravi  # exo
	p1.tropismN = 2  # [-] strength of tropism
	p1.tropismS = 0.1  # [rad/cm] maximal bending

	s1.name = "mainstem"
	s1.subType = 1
	s1.lmax = 25
	s1.lmaxs = 1
	s1.lb = 0
	s1.lbs = 0.1
	s1.la = 3
	s1.las = 0.1
	s1.ln = 3
	s1.lns = 0.2
	s1.lnf = 0  # 1 peut être cool
	s1.RotBeta = 0
	s1.BetaDev = 0
	s1.InitBeta = 0
	s1.gf = 1
	s1.successor = [3, 2]
	s1.successorP = [0.3, 0.7]
	# s1.successor = [3]
	# s1.successorP = [1]
	s1.tropismT = 1
	s1.theta = 0.
	s1.thetas = 0.05
	s1.tropismN = 2
	s1.tropismS = 0.005
	s1.r = 1
	s1.rs = 0.05

	s2.name = "secondary_stem"
	s2.subType = 3
	s2.lmax = 7
	s2.lmaxs = 0.2
	s2.lb = 0
	s2.la = 2
	s2.ln = 2
	s2.lns = 0.1
	s2.lnf = 0
	s2.RotBeta = 2 / 3
	s2.BetaDev = 0.1
	s2.InitBeta = 0
	s2.gf = 1
	s2.successor = [2]
	s2.successorP = [1]
	s2.tropismT = 4
	s2.theta = 0.3
	s2.thetas = 0.02
	s2.tropismN = 18
	s2.tropismS = 0.01
	s2.r = 0.5
	s2.rs = 0.02

	s3.name = "invisible stem"
	s3.subType = 2
	s3.lmax = 5
	s3.la = 0
	s3.ln = 0
	s3.RotBeta = 2/3
	s3.BetaDev = 0
	s3.InitBeta = 0.
	s3.lb = 0
	s3.la = 0
	s3.ln = 0
	s3.lnf = 0
	s3.RotBeta = 2 / 3
	s3.BetaDev = 0
	s3.InitBeta = 0
	s3.gf = 1
	s3.tropismT = 1
	s3.theta = 0.3
	s3.tropismN = 18
	s3.tropismS = 0.01
	s3.r = 1

	l1.name = 'leaf_under_second_stem'
	l1.subType = 2
	l1.lb = 2
	l1.la = 0.2
	l1.lmax = 5
	l1.lmaxs = 0.1
	l1.r = 0.5
	l1.rs = 0.02
	l1.RotBeta = 0.5
	l1.BetaDev = 0
	l1.InitBeta = 0.
	l1.tropismT = 1
	l1.tropismN = 5
	l1.tropismS = 0.15
	l1.theta = 0.35
	l1.gf = 1

	s4.name = "rosette"  # Cannot be a leaf organ because we can only create 1 leaf organ subType
	s4.subType = 4
	s4.lmax = 10
	s4.lmaxs = 0.2
	s4.lb = 1
	s4.la = 0
	s4.ln = 2
	s4.lnf = 0
	s4.RotBeta = 3 / 5
	s4.BetaDev = 0.02
	s4.InitBeta = 0
	s4.gf = 1
	s4.tropismT = 1
	s4.theta = 0.3
	s4.thetas = 0.02
	s4.tropismN = 18
	s4.tropismS = 0.01
	s4.r = 0.5
	s4.rs = 0.02

	plant.setOrganRandomParameter(p0)
	plant.setOrganRandomParameter(p1)
	plant.setOrganRandomParameter(s1)
	plant.setOrganRandomParameter(s2)
	plant.setOrganRandomParameter(s3)
	plant.setOrganRandomParameter(s4)
	plant.setOrganRandomParameter(l1)

	srp = pb.SeedRandomParameter(plant)  # with default values
	srp.seedPos = pb.Vector3d(0., 0., -3.)  # [cm] seed position
	srp.maxB = 0  # [-] number of basal roots (neglecting basal roots and shoot borne)
	srp.firstB = 10.  # [day] first emergence of a basal root
	srp.delayB = 3.  # [day] delay between the emergence of basal roots
	srp.maxTil = 10
	plant.setOrganRandomParameter(srp)

	return plant
