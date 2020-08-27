### As the xml examples are not well writen, creating plant characteristics by changing attribute of a default plant
### is easier
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
	s2.lnf = 5
	s2.RotBeta = 1
	s2.BetaDev = 0
	s2.InitBeta = 0
	s2.gf = 1
	s2.successor = [2]
	s2.successorP = [1]
	s2.tropismT = 4
	s2.theta = 0
	s2.tropismN = 18
	s2.tropismS = 0.01

	s3.name = "invisible"
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
	l1.lnf = 5

	plant.setOrganRandomParameter(p0)
	plant.setOrganRandomParameter(p1)
	plant.setOrganRandomParameter(s2)
	plant.setOrganRandomParameter(s3)
	plant.setOrganRandomParameter(l1)

	srp = pb.SeedRandomParameter(plant)  # with default values
	srp.seedPos = pb.Vector3d(0., 0., -3.)  # [cm] seed position
	srp.maxB = 0  # [-] number of basal roots (neglecting basal roots and shoot borne)
	srp.firstB = 10.  # [day] first emergence of a basal root
	srp.delayB = 3.  # [day] delay between the emergence of basal roots
	plant.setOrganRandomParameter(srp)

	return plant


def create_arabidopsis():  # The full of bugs arabidopsis (below is a correct version)
	plant = pb.Plant()
	p0 = pb.RootRandomParameter(plant)  # with default values,
	p1 = pb.RootRandomParameter(plant)  # all standard deviations are 0
	s1 = pb.StemRandomParameter(plant)
	s2 = pb.StemRandomParameter(plant)
	s3 = pb.StemRandomParameter(plant)
	s4 = pb.StemRandomParameter(plant)
	s5 = pb.StemRandomParameter(plant)
	l1 = pb.LeafRandomParameter(plant)
	l2 = pb.LeafRandomParameter(plant)
	l4 = pb.LeafRandomParameter(plant)

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
	s1.lmax = 22.5  # Etrangement la hauteur max est un peu plus petite que 2*lmax ici
	s1.lmax = 50
	# s2.lmaxs = 1
	s1.lb = 2
	s1.la = 5
	s1.ln = 6
	s1.lnf = 0
	s1.RotBeta = 0
	s1.BetaDev = 0
	s1.InitBeta = 0
	s1.gf = 2
	s1.successor = [2]
	s1.successorP = [1]
	s1.tropismT = 4
	s1.theta = 0.
	s1.tropismN = 18
	s1.tropismS = 0.01
	s1.r = 1

	s2.name = "secondary_stem"
	s2.subType = 2
	s2.lmax = 10
	# s2.lmaxs = 1
	s2.lb = 0
	s2.la = 2
	s2.ln = 2
	s2.lnf = 0
	s2.RotBeta = 2/3
	s2.BetaDev = 0
	s2.InitBeta = 0
	s2.gf = 1
	s2.successor = [3]
	s2.successorP = [1]
	s2.tropismT = 4
	s2.theta = 0.3
	s2.tropismN = 18
	s2.tropismS = 0.01
	s2.r = 0.5

	s3.name = "invisible"
	s3.subType = 3
	s3.lmax = 5
	s3.la = 0
	s3.ln = 0
	s3.RotBeta = 0
	s3.BetaDev = 0
	s3.InitBeta = 0.
	s3.lb = 0
	s3.la = 0
	s3.ln = 0
	s3.lnf = 0
	s3.RotBeta = 2/3
	s3.BetaDev = 0
	s3.InitBeta = 0
	s3.gf = 1
	s3.tropismT = 1
	s3.theta = 0.3
	s3.tropismN = 18
	s3.tropismS = 0.01
	s3.r = 1

	# l1.name = 'firstLeave'
	# l1.subType = 2
	# l1.lmax = 5
	# l1.a = 0
	# l1.r = 0
	# l1.RotBeta = 0
	# l1.BetaDev = 0
	# l1.InitBeta = 0.
	# l1.tropismT = 1
	# l1.tropismN = 5
	# l1.tropismS = 0.15
	# l1.theta = 0.4
	# l1.gf = 1
	l1.name = 'basil'
	l1.subType = 2
	l1.lb = 2
	l1.la = 0.2
	l1.lmax = 5
	l1.r = 0.5
	l1.RotBeta = 0.5
	l1.BetaDev = 0
	l1.InitBeta = 0.
	l1.tropismT = 1
	l1.tropismN = 5
	l1.tropismS = 0.15
	l1.theta = 0.35
	l1.gf = 1

	l2.name = 'second_leaf'
	l2.subType = 3
	l2.lmax = 2
	l2.a = 0.5
	l2.r = 0.5
	l2.RotBeta = 0
	l2.BetaDev = 0
	l2.InitBeta = 0.
	l2.tropismT = 1
	l2.tropismN = 5
	l2.tropismS = 0.15
	l2.theta = 0.2
	l2.gf = 1
	l2.name = 'second_leaf'
	l2.lb = 2
	l2.la = 0.2
	l2.lmax = 5
	l2.r = 0.5
	l2.RotBeta = 0.5
	l2.BetaDev = 0
	l2.InitBeta = 0.
	l2.tropismT = 1
	l2.tropismN = 5
	l2.tropismS = 0.15
	l2.theta = 0.35
	l2.gf = 1

	s4.name = "rosette"
	s4.subType = 4
	s4.lmax = 2
	# s2.lmaxs = 1
	s4.lb = 1
	s4.la = 0
	s4.ln = 2
	s4.lnf = 0
	s4.RotBeta = 1/5
	s4.BetaDev = 0
	s4.InitBeta = 0
	s4.gf = 1
	s4.tropismT = 1
	s4.theta = 0.3
	s4.tropismN = 18
	s4.tropismS = 0.01
	s4.r = 0.5
	# s4.successor = [5]
	# s4.successorP = [1]

	# s5.name = "rosette2"
	# s5.subType = 5
	# s5.lmax = 0
	# # s2.lmaxs = 1
	# s5.lb = 0
	# s5.la = 10
	# s5.ln = 1
	# s5.lnf = 0
	# s5.RotBeta = 2/3
	# s5.BetaDev = 0
	# s5.InitBeta = 0
	# s5.gf = 1
	# s5.tropismT = 1
	# s5.theta = 0.3
	# s5.tropismN = 18
	# s5.tropismS = 0.01
	# s5.r = 0.5
	#
	# l4.name = 'second_leaf'
	# l4.subType = 5
	# # l4.lb = 5
	# # l4.la = 0.2
	# l4.lmax = 5
	# # l4.r = 0.5
	# # l4.RotBeta = 0.5
	# # l4.BetaDev = 0
	# # l4.InitBeta = 0.
	# # l4.tropismT = 1
	# # l4.tropismN = 5
	# # l4.tropismS = 0.15
	# # l4.theta = 0.35
	# # l4.gf = 1

	plant.setOrganRandomParameter(p0)
	plant.setOrganRandomParameter(p1)
	plant.setOrganRandomParameter(s1)
	plant.setOrganRandomParameter(s2)
	plant.setOrganRandomParameter(s3)
	plant.setOrganRandomParameter(s4)
	# rs.setOrganRandomParameter(s5)
	plant.setOrganRandomParameter(l1)
	# rs.setOrganRandomParameter(l2)
	# rs.setOrganRandomParameter(l4)

	srp = pb.SeedRandomParameter(plant)  # with default values
	srp.seedPos = pb.Vector3d(0., 0., -3.)  # [cm] seed position
	srp.maxB = 0  # [-] number of basal roots (neglecting basal roots and shoot borne)
	srp.firstB = 10.  # [day] first emergence of a basal root
	srp.delayB = 3.  # [day] delay between the emergence of basal roots
	srp.maxTil = 10
	plant.setOrganRandomParameter(srp)

	return plant


def creacrea():  # Temporary code use for test purpose, it will eventually become the arabidopsis model.
	plant = pb.Plant()
	p0 = pb.RootRandomParameter(plant)  # with default values,
	p1 = pb.RootRandomParameter(plant)  # all standard deviations are 0
	s1 = pb.StemRandomParameter(plant)
	s2 = pb.StemRandomParameter(plant)
	s3 = pb.StemRandomParameter(plant)
	s4 = pb.StemRandomParameter(plant)
	l1 = pb.LeafRandomParameter(plant)
	l2 = pb.LeafRandomParameter(plant)

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
	s1.lmax = 50
	s1.lb = 2
	s1.la = 1
	s1.ln = 4
	s1.lnf = 0
	s1.RotBeta = 0
	s1.BetaDev = 0
	s1.InitBeta = 0
	s1.gf = 1
	s1.successor = [3, 2]
	s1.successorP = [0.5, 0.5]
	s1.tropismT = 4
	s1.theta = 0.
	s1.tropismN = 18
	s1.tropismS = 0.01
	s1.r = 1

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

	s2.name = "secondary_stem"
	s2.subType = 3
	s2.lmax = 10
	s2.lb = 0
	s2.la = 2
	s2.ln = 2
	s2.lnf = 0
	s2.RotBeta = 2 / 3
	s2.BetaDev = 0
	s2.InitBeta = 0
	s2.gf = 1
	s2.successor = [2]
	s2.successorP = [1]
	s2.tropismT = 4
	s2.theta = 0.3
	s2.tropismN = 18
	s2.tropismS = 0.01
	s2.r = 0.5

	l1.name = 'leaf_under_second_stem'
	l1.subType = 2
	l1.lb = 2
	l1.la = 0.2
	l1.lmax = 5
	l1.r = 0.5
	l1.RotBeta = 0.5
	l1.BetaDev = 0
	l1.InitBeta = 0.
	l1.tropismT = 1
	l1.tropismN = 5
	l1.tropismS = 0.15
	l1.theta = 0.35
	l1.gf = 1

	s4.name = "rosette"
	s4.subType = 4
	s4.lmax = 2
	s4.lb = 1
	s4.la = 0
	s4.ln = 2
	s4.lnf = 0
	s4.RotBeta = 3 / 5
	s4.BetaDev = 0
	s4.InitBeta = 0
	s4.gf = 1
	s4.tropismT = 1
	s4.theta = 0.3
	s4.tropismN = 18
	s4.tropismS = 0.01
	s4.r = 0.5

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
