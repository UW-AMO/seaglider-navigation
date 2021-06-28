from adcp import ProblemData, ProblemConfig, Weights, GliderProblem
from adcp.tests.conftest import standard_sim

ddat, adat, x = standard_sim()
data = ProblemData(ddat, adat)
weights = Weights(1, 1, 1, 1, 1)
config = ProblemConfig(vehicle_vel="otg")
prob = GliderProblem(data, config, weights)
prob.A

print("done!")
