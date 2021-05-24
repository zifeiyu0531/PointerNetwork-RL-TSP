from __future__ import division
import math
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from scipy.spatial.distance import pdist, squareform
import numpy as np




# Distance callback
class CreateDistanceCallback(object):

  def __init__(self, dist_matrix):
    self.matrix = dist_matrix

  def Distance(self, from_node, to_node):
    return self.matrix[from_node][to_node]


# Demand callback
class CreateDemandCallback(object):

  def __init__(self, demands):
    self.matrix = demands

  def Demand(self, from_node, to_node):
    return self.matrix[from_node]


# Service time (proportional to demand) callback
class CreateServiceTimeCallback(object):

  def __init__(self, demands, time_per_demand_unit):
    self.matrix = demands
    self.time_per_demand_unit = time_per_demand_unit

  def ServiceTime(self, from_node, to_node):
    return self.matrix[from_node] * self.time_per_demand_unit


# Create the travel time callback (equals distance divided by speed).
class CreateTravelTimeCallback(object):

  def __init__(self, dist_callback, speed):
    self.dist_callback = dist_callback
    self.speed = speed

  def TravelTime(self, from_node, to_node):
    travel_time = self.dist_callback(from_node, to_node) / self.speed
    return travel_time


# Create total_time callback (equals service time plus travel time).
class CreateTotalTimeCallback(object):

  def __init__(self, service_time_callback, travel_time_callback):
    self.service_time_callback = service_time_callback
    self.travel_time_callback = travel_time_callback

  def TotalTime(self, from_node, to_node):
    service_time = self.service_time_callback(from_node, to_node)
    travel_time = self.travel_time_callback(from_node, to_node)
    return service_time + travel_time


#############################################################
class SolutionCallback(object):
  def __init__(self, model):
    self.model = model
  def __call__(self):
    print(self.model.CostVar().Max())
#############################################################







class Solver(object):

    def __init__(self,max_length,speed):
        self.max_length = max_length
        self.depot = max_length
        self.speed = speed
        self.num_vehicles = 1

    def run(self, dist_matrix, demands, tw_open, tw_close):

        # Setting the number of locations (including depot)
        num_locations = len(demands)

        # Create routing model
        routing = pywrapcp.RoutingModel(num_locations, self.num_vehicles, self.depot)
        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

        #solution_callback = SolutionCallback(routing)
        #routing.AddAtSolutionCallback(solution_callback)

        # Setting first solution heuristic: the method for finding a first solution to the problem.
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION)

        ################################################
        #                                              #
        #               TO EXTEND SEARCH               #
        #                                              #
        ################################################
        
        # Setting guided local search in order to find the optimal solution
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit_ms = 1000

        # Create the distance callback
        dist_between_locations = CreateDistanceCallback(dist_matrix)
        dist_callback = dist_between_locations.Distance

        routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
        demands_at_locations = CreateDemandCallback(demands)
        demands_callback = demands_at_locations.Demand

        # Adding capacity dimension constraints
        VehicleCapacity = 10000
        NullCapacitySlack = 0
        fix_start_cumul_to_zero = True
        capacity = "Capacity"
        routing.AddDimension(demands_callback, NullCapacitySlack, VehicleCapacity, fix_start_cumul_to_zero, capacity)

        # Add time dimension.
        time_per_demand_unit = 0
        horizon = 10000
        time = "Time"

        service_times = CreateServiceTimeCallback(demands, time_per_demand_unit)
        service_time_callback = service_times.ServiceTime

        travel_times = CreateTravelTimeCallback(dist_callback, self.speed)
        travel_time_callback = travel_times.TravelTime

        total_times = CreateTotalTimeCallback(service_time_callback, travel_time_callback)
        total_time_callback = total_times.TotalTime

        routing.AddDimension(total_time_callback, horizon, horizon, fix_start_cumul_to_zero, time)

        # Add time window constraints.
        time_dimension = routing.GetDimensionOrDie(time)
        for i in range(0,num_locations): 
            time_dimension.CumulVar(i).SetRange(np.asscalar(tw_open[i]), np.asscalar(tw_close[i]))

        # Solve and returns a solution if any.
        assignment = routing.SolveWithParameters(search_parameters)

        if assignment:
            total_distance = assignment.ObjectiveValue()

            # Inspect solution.
            capacity_dimension = routing.GetDimensionOrDie(capacity);
            time_dimension = routing.GetDimensionOrDie(time);

            index = routing.Start(0)
            tour = []
            delivery_time = []
            while not routing.IsEnd(index):
                node_index = routing.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                tmin = assignment.Min(time_var)
                tmax = assignment.Max(time_var)
                index = assignment.Value(routing.NextVar(index))
                tour.append(node_index)        #(node_index, tmin, tmax))
                delivery_time.append(tmin)

            node_index = routing.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            tmin = assignment.Min(time_var)
            tmax = assignment.Max(time_var)
            tour.append(node_index)        #(node_index, tmin, tmax))

            return tour, total_distance, delivery_time

        else:
            print("\n Or-tools: No solution found.")
            return [-1,-1],-1,-1