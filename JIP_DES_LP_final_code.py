# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:44:04 2024

@author: Giacomo
"""

 # Import necessary libraries
import salabim as sim
import random
from gurobipy import Model, GRB, quicksum
import pandas as pd
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows


sim.yieldless(True)  # Make the model yieldless

#Setting parameters important to the research question
price_estimation_precision = 1 #between 0 and 2, 0 if the urban miner is perfectly accurate in its price estimation
selling_date_uncertainty = 7 #in days, urban miner guesses the selling date within half of those many days, the actual selling date is sampled fro a range that big, centred around the urban miner's guess
inspection_cost = 150  # Fixed inspection cost per window set
alpha = 0.5 #adjustment factor for remanufacturing, percentage increase in value

cost_transport = 5 #cost of transportation per unit volume
cost_remanufacturing = 15 # Cost of remanufacturing per window per day
cost_storage = 0.5 # Cost of storage per unit volume per day
# Step 1: Setting up the simulation environment
class WindowSetSimulation(sim.Environment):
    def setup(self):
        # Initialize general parameters for the simulation environment here
        self.time_horizon = 30  # Simulation for 1 year
        self.daily_time_step = 1  # Daily steps
        self.reoptimization_frequency = 7  # Re-optimization every 7 days (parameter to control)
        self.window_sets = []  # Keep track of all window sets
        self.total_revenue = 0  # Track total revenue from all sales
        self.total_cost = 0  # Track total cost of handling all window sets
        self.total_profit = 0  # Track total profit
        self.env.inspection_cost = inspection_cost
        # Selling date range for reusing facility
        self.selling_start = 5 - selling_date_uncertainty  # Minimum days after demolition
        self.selling_end = 5 + selling_date_uncertainty   # Maximum days after demolition
        self.estimation_precision = price_estimation_precision  # Defines the width of the range for price estimation
        self.total_movements = 0  # Initialize movement counter
        
    def print_KPIs(self):
     # Initialize individual cost categories
     total_remanufacturing_cost = 0
     total_storage_cost = 0
     total_transportation_cost = 0
     total_inspection_cost = 0  # Optional: Include if you want to track inspection costs separately
     total_windows = 0
 
     # Iterate over all window sets to accumulate costs
     for window_set in self.window_sets:
         total_remanufacturing_cost += window_set.accumulated_costs.get('remanufacturing', 0)
         total_storage_cost += window_set.accumulated_costs.get('storage', 0)
         total_transportation_cost += window_set.accumulated_costs.get('transportation', 0)
         total_inspection_cost += window_set.accumulated_costs.get('inspection', 0)
         total_windows += window_set.num_windows
 
     # Calculate the overall total cost
     self.total_cost = total_remanufacturing_cost + total_storage_cost + total_transportation_cost + total_inspection_cost
 
     # Calculate profit
     self.total_profit = self.total_revenue - self.total_cost
     
     average_profit_per_window = self.total_profit / total_windows
 
     # Print summary with detailed cost breakdown
     print("\n--- Summary ---")
     print(f"Total Revenue: {self.total_revenue}")
     print(f"Total Cost: {self.total_cost}")
     print(f"  - Remanufacturing Cost: {total_remanufacturing_cost}")
     print(f"  - Storage Cost: {total_storage_cost}")
     print(f"  - Transportation Cost: {total_transportation_cost}")
     print(f"  - Inspection Cost: {total_inspection_cost}")  # Optional
     print(f"Total Profit: {self.total_profit}")
     print(f"Total Number of Windows: {total_windows}")
     print(f"Average Profit per Window: {average_profit_per_window:.2f}")
     print(f"Peak warehouse usage: {self.warehouse.peak_usage}")
     print(f"Total Movements: {self.total_movements}")

# Step 2: Implementing the Window Set Data Structure
class WindowSet(sim.Component):
    def setup(self, id_, material, condition, num_windows, volume, demolition_date, 
              type_, sustainability_class, actual_selling_price_reuse, actual_selling_price_recycle, selling_date_reusing):
        self.name = f"WindowSet_{id_}"  # Use a string name to avoid AttributeError
        self.material = material  # Material characteristic
        self.condition = condition  # Condition characteristic
        self.num_windows = num_windows  # Number of windows in the set
        self.volume = round(volume, 2)  # Volume of the window set rounded to 2 decimal places
        self.type_ = type_  # Type characteristic
        self.sustainability_class = sustainability_class  # Sustainability class
        self.demolition_date = demolition_date  # Date when the window set becomes available
        self.selling_date_reusing = selling_date_reusing  # Date when the window set can be sold to the reusing facility
        self.current_state = "Not Available"  # Initial state
        self.processing_status = None  # Whether it is being remanufactured or not
        self.actual_selling_price_reuse = round(actual_selling_price_reuse, 2)  # Actual selling price for reusing facility rounded to 2 decimal places
        self.actual_selling_price_recycle = round(actual_selling_price_recycle, 2)  # Actual selling price for recycling facility rounded to 2 decimal places
        self.days_in_remanufacturing = 0  # Track days spent in remanufacturing
        self.t_rem = 3  # Required days in remanufacturing to complete processing
        # Calculate estimated selling price for reuse with estimation precision
        random_multiplier = random.uniform(1 - (self.env.estimation_precision / 2), 1 + (self.env.estimation_precision / 2))
        self.estimated_selling_price_reuse = round(self.actual_selling_price_reuse * random_multiplier, 2)
        # Estimated selling time so complicated to ensure it i less then the time horizon
        max_additional_days_1 = max(1, self.env.time_horizon - demolition_date - 3)
        self.estimated_selling_time_reuse = min(
            (self.demolition_date + 5) ,
            self.env.time_horizon - 1
        )

        self.estimated_selling_price_recycle = round(self.actual_selling_price_recycle, 2)  # Perfect estimation for recycling rounded to 2 decimal places
        self.estimated_selling_time_recycle = None
        self.plan = []  # Future actions plan, initially empty
        self.accumulated_costs = {
            'remanufacturing': 0,
            'storage': 0,
            'transportation': 0,
            'inspection': inspection_cost
        }  # Placeholder for accumulated costs
        self.destination = None  # Destination assigned by the scheduler (either "Reusing" or "Recycling")
        self.awaiting_selling_date = False  # Flag for window sets awaiting selling date
        self.processing_start_day = None  # Start day of processing
        self.processing_end_day = None  # End day of processing
        self.env.window_sets.append(self)  # Add this window set to the environment's list
        
    def report(self):
        print(f"\n{self.name} Characteristics:")
        for attr, value in vars(self).items():
            print(f"  {attr}: {value}")

    def __str__(self):
        return (f"{self.name} - State: {self.current_state}, Demolition date: {self.demolition_date}, "
                f"Selling Date Reuse: {self.selling_date_reusing}, Material: {self.material}, Condition: {self.condition}, "
                f"Num Windows: {self.num_windows}, Volume: {self.volume}, Type: {self.type_}, "
                f"Sustainability Class: {self.sustainability_class}, "
                f"Actual Selling Price Reuse: {self.actual_selling_price_reuse}, "
                f"Actual Selling Price Recycle: {self.actual_selling_price_recycle}, "
                f"Destination: {self.destination}")

    def report(self):
        print(self.__str__())

# Step 3: Implementing the Generator Component
class WindowSetGenerator(sim.Component):
    def setup(self):
        self.num_sets_to_generate = 30  # Randomly choose the number of window sets to generate from a range
        self.generate_window_sets()
    
    def process(self):
        pass  # No processing needed in this component

    def generate_window_sets(self):
        window_sets_data = []  # List to store characteristics of each window set
        # Generate window sets with random characteristics
        for i in range(self.num_sets_to_generate):
            id_ = i + 1
            material = random.choice(["Al1", "Al2", "Al3"])
            condition = random.choice(["Poor", "Decent", "Good"])
            num_windows = random.randint(5, 30) 
            volume = round(random.uniform(0.2, 1.0), 2)  
            type_ = random.choice(["Type1", "Type2", "Type3"])
            sustainability_class = random.choice(["SC1", "SC2", "SC3", "SC4"])
            demolition_date = random.randint(1, env.time_horizon - 10)  # Random demolition date within the year
            # Ensure max_additional_days - 3 is at least 1 to avoid an empty range
            max_additional_days = max(env.selling_end, self.env.time_horizon - demolition_date - 3)
            selling_date_reusing = demolition_date + random.randint(env.selling_start, env.selling_end)

            # Calculate actual selling price for reuse facility using polynomial function of characteristics
            actual_selling_price_reuse = (
                (volume * 2) + 
                (60 if condition == "Good" else 40 if condition == "Decent" else 15) +
                (40 if type_ == "Type1" else 25 if type_ == "Type2" else 15) +
                (15 if sustainability_class == "SC1" else 30 if sustainability_class == "SC2" else 45 if sustainability_class == "SC3" else 50)
            ) * num_windows

            # Calculate actual selling price for recycling facility based on volume, material, and number of windows
            actual_selling_price_recycle = (
                (volume * 2) + (30 if material == "Al1" else 50 if material == "Al2" else 60)
            ) * num_windows

            # Create WindowSet instance
            window_set = WindowSet(
                id_=id_, material=material, condition=condition, num_windows=num_windows,
                volume=volume, demolition_date=demolition_date, type_=type_, sustainability_class=sustainability_class,
                actual_selling_price_reuse=actual_selling_price_reuse, actual_selling_price_recycle=actual_selling_price_recycle,
                selling_date_reusing=selling_date_reusing, env=self.env
            )
            # Store data in a dictionary for easy DataFrame conversion
            window_sets_data.append({
                "ID": f"WindowSet_{id_}",
                "Material": material,
                "Condition": condition,
                "Num Windows": num_windows,
                "Volume": volume,
                "Type": type_,
                "Sustainability Class": sustainability_class,
                "Demolition Date": demolition_date,
                "Selling Date Reuse": selling_date_reusing,
                "Actual Selling Price Reuse": round(actual_selling_price_reuse, 2),
                "Actual Selling Price Recycle": round(actual_selling_price_recycle, 2),
                "Estimated Selling Price Reuse": round(window_set.estimated_selling_price_reuse, 2),
                "Estimated Selling Date Reuse": window_set.estimated_selling_time_reuse
            })
        
        # Convert to DataFrame for clear tabular display
        window_sets_df = pd.DataFrame(window_sets_data)
        pd.set_option('display.max_rows', None)  # Show all rows if needed
        pd.set_option('display.max_columns', None)  # Show all columns if needed
        print(window_sets_df)
# Step 6: Implementing the Urban Miner Component (LP-based Schedule)
class UrbanMiner(sim.Component):
    def setup(self):
        self.schedule = {}  # Dictionary to hold the schedule
        self.inspection_cost = self.env.inspection_cost
        self.create_optimal_schedule()
        # Calculate average selling delay based on environment’s selling range
        self.average_selling_delay = (self.env.selling_start + self.env.selling_end) / 2


    def process(self):
        pass  # No processing needed in this component

    def create_optimal_schedule(self):
        # Create a new Gurobi model for optimization
        model = Model("remanufacturing")

        # Parameters
        t_rem = 4
        plan_horizon = env.time_horizon
        I = range(len(self.env.window_sets))  # Window sets indexed from 0 to len(window_sets)-1
        S = ['O', 'W', 'R', 'RU', 'RC'] #set of possible states: origin, warehouse, remanufacturing, reusing facility, recycling facility
        T = range(plan_horizon)  # Time periods for 3 weeks (21 days)
        C_rem = self.env.remanufacturing_facility.capacity  # Remanufacturing capacity
        V_warehouse = self.env.warehouse.capacity  # Warehouse capacity
        c_rem = cost_remanufacturing # Cost of remanufacturing per window per day
        c_storage = cost_storage  # Cost of storage per unit volume per day
        c_transport = cost_transport  # Cost of transportation per unit volume per movement
        # alpha = alpha  # Adjustment factor for reuse price based on remanufacturing

        # Decision variables
        f = model.addVars(I, S, S, T, vtype=GRB.BINARY, name="f")
        s = model.addVars(I, S, T, vtype=GRB.BINARY, name="s")
        R = model.addVars(I, vtype=GRB.BINARY, name="R") # is 1 if window set i has completed remanufacturing

        # Objective function
        model.setObjective(
            quicksum(
                s[i,'RU', plan_horizon-1] * self.env.window_sets[i].estimated_selling_price_reuse * (1 + alpha * R[i])
                + s[i,'RC', plan_horizon-1] * self.env.window_sets[i].actual_selling_price_recycle
                for i in I
            ) - quicksum(
                # Remanufacturing cost
                s[i, 'R', t] * self.env.window_sets[i].num_windows * c_rem
                for i in I for t in T
            ) - quicksum(
                # Storage cost
                s[i, 'W', t] * self.env.window_sets[i].volume * self.env.window_sets[i].num_windows * c_storage
                for i in I for t in T
            ) - quicksum(
                # Transportation cost
                f[i, s1, s2, t] * self.env.window_sets[i].volume * self.env.window_sets[i].num_windows * c_transport
                for i in I for t in T for s1 in S for s2 in S if s1 != s2
            ) - quicksum(
               # Fixed inspection cost for each window set
               self.inspection_cost for i in I
            ),
            GRB.MAXIMIZE
        )

        # Constraints

        # Have to have a state each day after demolition
        for i in I:
            for t in T:
                if t >= self.env.window_sets[i].demolition_date:
                    model.addConstr(quicksum(s[i, s1, t] for s1 in S) == 1, name=f"state_exclusivity_{i}_{t}")

        # Availability based on demolition date
        for i in I:
            for t in T:
                if t < self.env.window_sets[i].demolition_date:
                    for s1 in S:
                        model.addConstr(s[i, s1, t] == 0, name=f"availability_s_{i}_{s1}_{t}")

        # On demolition date stay in the origin
        for i in I:
            model.addConstr(f[i, 'O', 'O', self.env.window_sets[i].demolition_date] == 1)

        # No flow before the day after demolition date
        for i in I:
            for t in T:
                if t < self.env.window_sets[i].demolition_date:
                    model.addConstr(quicksum(f[i, s1, s2, t] for s1 in S for s2 in S) == 0)

        # Flow constraints, assure connection between flow and state
        for i in I:
            for t in T:
                if t > self.env.window_sets[i].demolition_date:
                    for s2 in S:
                        model.addConstr(quicksum(f[i, s1, s2, t] for s1 in S) == s[i, s2, t])

        for i in I:
            for t in T:
                if t > self.env.window_sets[i].demolition_date:
                    for s1 in S:
                        model.addConstr(s[i, s1, t-1] == quicksum(f[i, s1, s2, t] for s2 in S))

        # Have to have exactly one flow each timestep after demolition, aka Flow exclusivity
        for i in I:
            for t in T:
                if t >= self.env.window_sets[i].demolition_date:
                    model.addConstr(quicksum(f[i, s1, s2, t] for s1 in S for s2 in S) == 1)

        # Initial state at demolition date (t = demolition_date)
        for i in I:
            model.addConstr(s[i, 'O', self.env.window_sets[i].demolition_date] == 1, name=f"initial_state_demolition_{i}")

        # No movements back to origin
        for i in I:
            for t in T:
                if t > self.env.window_sets[i].demolition_date:
                    model.addConstr(quicksum(f[i, s1, 'O', t] for s1 in S) == 0, name=f"no_back_to_origin_{i}_{t}")

        # No movements from RU or RC to other states except self-flows
        for i in I:
            for t in T:
                if t >= self.env.window_sets[i].demolition_date:
                    for s1 in ['RU', 'RC']:
                        for s2 in S:
                            if s2 != s1:
                                model.addConstr(
                                    f[i, s1, s2, t] == 0,
                                    name=f"no_from_terminal_except_self_{i}_{s1}_{s2}_{t}"
                                )

        # Movement to RU allowed only after selling date
        for i in I:
            for t in T:
                if t < self.env.window_sets[i].estimated_selling_time_reuse:
                    model.addConstr(quicksum(f[i, s1, 'RU', t] for s1 in S) == 0)

        # R (the decision variable) is 1 only if window set i has been in 'R' state for more than t_rem days in total
        # for i in I:
        #     model.addConstr(R[i] <= quicksum(s[i, 'R', t] for t in T) - t_rem + 1)
         
        for i in I:
            model.addConstr(
                quicksum(s[i, 'R', t] for t in T) >= t_rem * R[i],
                name=f"remanufacturing_completion_{i}"
            )

         
        # Remanufacturing Capacity Constraints
        for t in T:
            model.addConstr(
                quicksum(self.env.window_sets[i].num_windows * s[i, 'R', t] for i in I) <= C_rem,
                name=f"remanufacturing_capacity_{t}"
            )
            
        # Warehouse Capacity Constraints
        for t in T:
            model.addConstr(
                quicksum(self.env.window_sets[i].volume * self.env.window_sets[i].num_windows * s[i, 'W', t] for i in I) <= V_warehouse,
                name=f"warehouse_capacity_{t}"
            )



        # Solve the model
        model.optimize()
        
        # Initialize matrix to store state sequences
        num_sets = len(self.env.window_sets)
        time_horizon = self.env.time_horizon
        state_matrix = [["O" for _ in range(time_horizon)] for _ in range(num_sets)]
    
        # # Populate the matrix based on the state variable s
        # for i in range(num_sets):
        #     for t in range(time_horizon):
        #         # Find the active state for window set i at time t
        #         for state in S:  # Use `state` instead of `s` to avoid conflict with the decision variable `s`
        #             if s[i, state, t].x > 0.5:  # If s_{i,state,t} is active
        #                 state_matrix[i][t] = state  # Set the state in the matrix
        #                 break  # Move to the next time step after finding the active state
    
        # # Print the matrix in a formatted way
        # print("\n--- Optimal Sequence of States (Matrix Form) ---")
        # for i in range(num_sets):
        #     print(f"Window Set {i+1:>2}:", " ".join(f"{state:>3}" for state in state_matrix[i]))
            
        #     # Print and store only non-self flows in the optimal schedule
        #     if model.status == GRB.OPTIMAL:
        #         print("\n--- Optimal Flows ---")
        #         for i in I:
        #             plan = []
        #             for t in T:
        #                 for s1 in S:
        #                     for s2 in S:
        #                         if s1 != s2 and f[i, s1, s2, t].x > 0.5:
        #                             action = {'time': t, 'from': s1, 'to': s2}
        #                             plan.append(action)
        #                             # print(f"Window set {i} moves from {s1} to {s2} at time {t}")
        #             self.schedule[self.env.window_sets[i].name] = plan
                    
         # Print and store only non-self flows in the optimal schedule
        if model.status == GRB.OPTIMAL:
            for i in I:
                plan = []
                flow_str = f"Window set {i + 1} = "  # Start formatted flow string
                for t in T:
                    for s1 in S:
                        for s2 in S:
                            if s1 != s2 and f[i, s1, s2, t].x > 0.5:
                                action = {'time': t, 'from': s1, 'to': s2}
                                plan.append(action)
                                flow_str += f"{s1} -> {s2},  "  # Append each flow transition to the string
                print(flow_str.strip())  # Print formatted flow string
                self.schedule[self.env.window_sets[i].name] = plan    
                
        print("\n--- Remanufacturing Status ---")
        for i in I:
            window_set_name = self.env.window_sets[i].name
            remanufactured = "Yes" if R[i].x >= 0.99 else "No"  # Threshold to account for floating-point precision
            print(f"{window_set_name} is remanufactured: {remanufactured}")

        

    def get_schedule(self):
        return self.schedule


class Scheduler(sim.Component):
    def setup(self, urban_miner):
        self.urban_miner = urban_miner
        self.schedule = self.urban_miner.get_schedule()
        self.cost_per_unit_volume = cost_transport  # Base transportation cost per unit volume

    def move(self, window_set, from_state, to_state):
        # Calculate transportation cost for the entire set (volume per window * number of windows)
        transport_cost = window_set.volume * window_set.num_windows * self.cost_per_unit_volume
        window_set.accumulated_costs['transportation'] += transport_cost
        self.env.total_movements += 1
        print(f"{window_set.name} transportation cost from {from_state} to {to_state}: {transport_cost}")


        # Special handling for reusing facility (RU)
        if to_state == 'RU':
            # Redirect to warehouse if selling date has not arrived
            if self.current_day < window_set.selling_date_reusing:
                print(f"{window_set.name} cannot go to Reusing Facility (RU) yet, moving to Warehouse instead.")
                to_state = 'W'
                window_set.awaiting_selling_date = True  # Set awaiting selling date to True
        # Remove from Warehouse or Remanufacturing Facility if moving away
        if from_state == 'W':
            self.env.warehouse.remove_window_set(window_set)
            print(f"{window_set.name} has been removed from the Warehouse.")
        elif from_state == 'R':
            self.env.remanufacturing_facility.remove_window_set(window_set)
            print(f"{window_set.name} has been removed from the Remanufacturing Facility.")

        # Update current state based on `to_state`
        if to_state == 'W':  # Move to warehouse
            self.env.warehouse.receive_window_set(window_set)
        elif to_state == 'R':  # Move to remanufacturing
            self.env.remanufacturing_facility.receive_window_set(window_set)
        elif to_state == 'RU':  # Move to reusing facility (actual sale)
            window_set.current_state = "Sold to Reusing"
            # Calculate revenue and update total revenue
            sale_revenue = window_set.actual_selling_price_reuse * (1 + alpha) if window_set.processing_status == "Completed" else window_set.actual_selling_price_reuse
            self.env.total_revenue += sale_revenue
            print(f"{window_set.name} has been sold to Reusing Facility. Revenue from sale: {sale_revenue}")
        elif to_state == 'RC':  # Move to recycling facility (actual sale)
            window_set.current_state = "Sold to Recycling"# Calculate revenue and update total revenue
            sale_revenue = window_set.actual_selling_price_recycle
            self.env.total_revenue += sale_revenue
            print(f"{window_set.name} has been sold to Recycling Facility. Revenue from sale: {sale_revenue}")

        # Final state update for all other transitions
        window_set.current_state = to_state

    def process(self):
        while self.env.now() < self.env.time_horizon:
            self.current_day = self.env.now()
            print(f"--- Simulation Day: {int(self.current_day)} ---")
            
            # Check each window set
            for window_set in self.env.window_sets:
                # Implement actions based on the LP schedule
                if window_set.name in self.schedule:
                    for action_plan in self.schedule[window_set.name]:
                        if action_plan['time'] == self.current_day:
                            from_state = action_plan['from']
                            to_state = action_plan['to']
                            # Use `move` function to handle all state transitions and cost calculations
                            self.move(window_set, from_state, to_state)

            self.hold(self.env.daily_time_step)  # Wait for the next day

# Step 7: Implementing the Warehouse Component
class Warehouse(sim.Component):
    def setup(self):
        self.capacity = 1000000  # Capacity exists but is not used in the DES, only in the LP
        self.current_volume = 0
        self.peak_usage = 0  # Track peak usage for KPIs
        self.stored_window_sets = []
        self.c_storage = cost_storage

    def process(self):
        while self.env.now() < self.env.time_horizon:
            # Accumulate storage costs for each window set
            for window_set in self.stored_window_sets:
                storage_cost = window_set.volume * window_set.num_windows * self.c_storage
                window_set.accumulated_costs['storage'] += storage_cost
                print(f"{window_set.name} storage cost for day {self.env.now()}: {storage_cost}")
            
            # Check daily if any window sets are ready to be sold
            for window_set in self.stored_window_sets.copy():  # Use copy to avoid modification during iteration
                if window_set.awaiting_selling_date and self.env.now() >= window_set.selling_date_reusing:
                    # Selling date has arrived; send to Reusing Facility (RU)
                    print(f"{window_set.name} is now ready for reusing.")
                    self.send_to_reusing(window_set)
            # Update peak usage if current volume exceeds the recorded peak
            self.current_volume = sum(ws.volume * ws.num_windows for ws in self.stored_window_sets)
            if self.current_volume > self.peak_usage:
                self.peak_usage = self.current_volume
                print(f"New peak storage is {self.peak_usage}")

            # Hold for a day before rechecking
            self.hold(self.env.daily_time_step)
            
    def receive_window_set(self, window_set):
        # No capacity checks; directly accept the window set
        self.stored_window_sets.append(window_set)
        window_set.current_state = "Warehouse"
        print(f"{window_set.name} has arrived at the Warehouse.")
        

    def remove_window_set(self, window_set):
        if window_set in self.stored_window_sets:
            self.stored_window_sets.remove(window_set)
            self.current_volume -= window_set.volume
            print(f"{window_set.name} has been removed from the Warehouse.")

    def send_to_reusing(self, window_set):
        # Update state, remove from warehouse, and notify Scheduler to handle the move
        window_set.current_state = "Ready for Reusing"
        window_set.awaiting_selling_date = False
        self.remove_window_set(window_set)
        # Call the Scheduler to move the window set to RU
        self.env.scheduler.move(window_set, "W", "RU")

# Step 8: Implementing the Remanufacturing Facility Component
class RemanufacturingFacility(sim.Component):
    def setup(self, c_rem=50):
        self.capacity = 50  # Maximum number of windows that can be processed simultaneously
        self.window_sets_in_process = []
        self.c_rem = cost_remanufacturing  # Cost of remanufacturing per window per day

    def process(self):
        while self.env.now() < self.env.time_horizon:
            self.current_day = self.env.now()
            
            # Update each window set in processing
            for window_set in self.window_sets_in_process.copy():
                # Increment cumulative days spent in remanufacturing
                window_set.days_in_remanufacturing += 1
                
                # Apply remanufacturing cost
                remanufacturing_cost = window_set.num_windows * self.c_rem
                window_set.accumulated_costs['remanufacturing'] += remanufacturing_cost
                print(f"{window_set.name} remanufacturing cost for day {window_set.days_in_remanufacturing}: {remanufacturing_cost}")

                # Check if remanufacturing is completed based on cumulative days
                if window_set.days_in_remanufacturing >= window_set.t_rem:
                    # Mark as completed before removing
                    window_set.processing_status = "Completed"
                    print(f"{window_set.name} has completed remanufacturing after {window_set.days_in_remanufacturing} days.")
                    
                    # Remove the window set from the remanufacturing facility
                    self.remove_window_set(window_set)
                else:
                    # Indicate that the window set is still in processing
                    window_set.processing_status = "Processing"
                    print(f"{window_set.name} is still in remanufacturing, day {window_set.days_in_remanufacturing}.")

            # Wait for the next day
            self.hold(self.env.daily_time_step)

    def receive_window_set(self, window_set):
        # Check remanufacturing capacity before accepting the window set
        total_windows_in_process = sum(ws.num_windows for ws in self.window_sets_in_process)
        if total_windows_in_process + window_set.num_windows <= self.capacity:
            window_set.current_state = "Remanufacturing"
            window_set.processing_status = "Processing"
            window_set.days_in_remanufacturing = 0  # Reset days count upon entry
            self.window_sets_in_process.append(window_set)
            print(f"{window_set.name} has started remanufacturing.")
        else:
            print(f"Remanufacturing Facility is at capacity. Cannot accept {window_set.name}.")

    def remove_window_set(self, window_set):
        if window_set in self.window_sets_in_process:
            self.window_sets_in_process.remove(window_set)
            print(f"{window_set.name} has been removed from the Remanufacturing Facility.")
        else:
            print(f"{window_set.name} is not in the Remanufacturing Facility.")

# Initialize and Run Simulation
env = WindowSetSimulation()
generator = WindowSetGenerator(env=env)
# Initialize facilities before UrbanMiner to avoid AttributeError
env.remanufacturing_facility = RemanufacturingFacility(env=env)
env.warehouse = Warehouse(env=env)
urban_miner = UrbanMiner(env=env)
env.scheduler = Scheduler(env=env, urban_miner=urban_miner)
env.run(till=env.time_horizon)
env.print_KPIs()