# John T.H. Wong
# Hardware: Macbook Air, M1, 2020
# OS: MacOS 15.3.2
# SDE: VS Code, 1.98.2 (Universal)

import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
from IPython.display import display, clear_output
import time
import numpy as np

def uniform_exclusive(a, b):
    # This function draws from a uniform and open interval between two numbers

    # An infinite loop that runs until random.uniform(0,1) draws a number that is not zero or one
    while True:
        x = random.uniform(a, b)
        if a < x < b:
            return x

def generate_crs_elasticities(elasticities_count):
    '''
    This function generates a list of random output elasticities that sum to 1.
    '''

    # Error handling
    if not isinstance(elasticities_count, int) or elasticities_count < 2:
        raise ValueError(
            "elasticities_count must be an integer and 2 or greater"
            )
    
    '''
    Generate the first elasticity.
    If there are only two goods, draw from 1/3 to 2/3.
    If there are >2 goods, draw from 0 to 1/(count - 1)
    The denominator adjustment ensures that elasticities 
    have somewhat evenly large values at the end.
    '''
    if elasticities_count == 2:
        first_elasticity = uniform_exclusive(1/3, 2/3)
    else:
        first_elasticity = 1/(elasticities_count - 1)

    elasticities_list = [first_elasticity]
    # Minus 1 because we already have one value
    for i in range(elasticities_count - 1):
        # Retrieve sum of previous elasticities in the list
        elasticities_sum = sum(elasticities_list)
        inverse = 1 - elasticities_sum
        if i < elasticities_count - 2:
            # Draw a random value between zero and 1/(elasticities_count - 1).
            elasticity = uniform_exclusive(0, 1/(elasticities_count - 1))
        # If it's the last iteration
        else:
            # The elasticitiy takes the value (1 minus the sum)
            elasticity = inverse
        # Add it to the list of elasticities
        elasticities_list.append(elasticity)
    # Scramble so that the value of one index is not always the largest
    # .shuffle is an in-place function
    random.shuffle(elasticities_list) 
    return elasticities_list


class Agent:
    def __init__(
            self, goods_type_count, max_endowment_per_good, 
            equality
            ):
        '''
        This function initializes the agent class.
        goods_type_count defines the variety of goods the agent holds.
        max_endowment_per_good is the max of each good they will 
        initially hold.
        '''
        # .__preferences stores the list of output elasticities
        self.__preferences = generate_crs_elasticities(goods_type_count)
        # .__inventory keeps track of how much of each good the agent holds
        self.__inventory = []
        # Stock the inventory with random integers from 1 to max_endowment.
        for i in range(goods_type_count):
            # Endowment of the last good depends on whether equality is true
            if i == goods_type_count - 1 and equality:
                # If equality is true,
                # last good's endowment is a constant minus the sum of 
                # endowment of all other goods (i.e., existing endowment).
                good_count = (
                    # The product ensures the constant is larger than 
                    # existing endowment.
                    max_endowment_per_good * goods_type_count 
                    - sum(self.__inventory)
                    )
                self.__inventory.append(good_count)
                break
            good_count = int(random.uniform(1, max_endowment_per_good))
            self.__inventory.append(good_count)
        # Shuffle so each agent is not always most endowed with last good.
        random.shuffle(self.__inventory)
        
    
    # Get attribute functions
    def get_pref(self, k):
        return self.__preferences[k]
    def get_inventory(self, k):
        return self.__inventory[k]
    def get_entire_inventory(self):
        return self.__inventory
    def get_entire_pref(self):
        return self.__preferences
    
    # Modify attribute functions
    def chg_inventory(self, k, chg):
        self.__inventory[k] += chg
    
class Market:
    def __init__(self):
        self.agents = []
        self.shuffled_agents_index = []
        self.shuffled_goods_index = []
        self.transacted_goods_tuple = []
        self.transacting_agents_tuple = []
        self.inventory_panel = np.empty((0, 0))
        self.aggregate_utility = []
        self.individual_utilities = []
        self.shuffled_agents_index_vintage = []
        self.friends_dict = {}

    def generate_agents(
            self, agent_count, goods_type_count, max_endowment_per_good, 
            equality
            ):
        # This function generates agents and stores them into .agents.

        # Store some arguments as attribute
        self.agent_count = agent_count
        self.goods_type_count = goods_type_count
        # Clear list of agents
        self.agents = []        
        # Create as many agents as agent_count specifies
        for i in range(agent_count):
            self.agents.append(
                Agent(goods_type_count, max_endowment_per_good, equality)
                )
        # Update .shuffled_agents_index:
        self.shuffled_agents_index = list(range(agent_count))
        # Update .shuffled_goods_index:
        self.shuffled_goods_index = list(range(goods_type_count))

    def generate_network(self, size: int):
        # Error handling
        if self.agents is None:
            raise AttributeError(
                "Agents need to be generated before network can be."
            )   
        if size <= 0:
            raise ValueError(
                "Size must be a positive integer for network to be generated."
                )
        if size >= len(self.agents):
            raise ValueError(
                "Network size must be less than the number of agents."
            )
        
        # Clear existing friends dictionary
        self.friends_dict = {}

        # Set up dictionary for fast retrieval of agents.
        friends_dict = {}
        for i in range(self.agent_count):
            # use list comprehension to get an agents index that excludes i
            agents_index_without_i = (
                 [j for j in range(self.agent_count) if j != i]
            )
            friends = random.sample(agents_index_without_i, size)
            friends_dict[i] = friends

        self.friends_dict = friends_dict

    def clear_transactions(self):
        self.transacted_goods_tuple = []
        self.transacting_agents_tuple = []
        self.inventory_panel = np.empty((0, 0))
        self.aggregate_utility = []
        self.individual_utilities = []
        self.shuffled_agents_index_vintage = []

    '''
    Exchange procedure for an n-agents, p-goods problem: 

    For each "trading day":

    1. Loop across pairwise agents: we will uniformly activate each agent m, where m = 0, 1,..., n - 1. Agent m will pair with each m+z agent, until m+z = n. This subprocedure ensures that we iterate through all (n^2 - n) pairwise agent combinations. After all n-1 agents have gone, we will shuffle the list and start with m = 0 again.

    1a. Agent retrieval: we index each agent initially by i. We store a duplicate of this initial index i in a list within the market, and index this index list by the aforementioned m. In lieu of shuffling .agents, we shuffle this index. We then iterate through this shuffled list, and for each m, retrieves its value (which is a possible value of i, call it i*), and then find the i*-th agent.

    2. Loop across pairwise goods: when paired, agent i will iterate through each good j to see if agent -i is willing to take j. Agent -i will iterate through the same list, starting at j+1, to see if agent i is willing to take the good j+1. This subprocedure ensures that we iterate through all (p^2 - p) pairwise combination of goods. At the end of the loop, we shuffle the list.

    2a. Preference and good retrieval: we index each preference (stored within agent) initially by k, and the initial goods list (stored within market) inherits this index. We store a duplicate of this intiial index k in a list within the market---and index this index list by the aforementioned j. When we shuffle the goods list, we mean we shuffle the index list. The agent actually iterates through the shuffled list, and for each j, retrieves the its value (which is a possible value of k, call it k*), and then finds the k*-th preference and good .

    3. Loop within pairwise goods: for two pairwise goods that successfully trades for one unit, the agents update their pairwise MRS and checks if another trade is possible. The trade continues to occur until the pairwise MRS's of both agents cross.
    '''

    def execute_exchange(
            self, 
            trading_days, 
            strategic_error=None,
            plot_type=None,
            trade_within_network: bool = False,
            ):
        '''
        strategic_error is the probability (CDF) that the two agents will stop trading two goods, even when it will improve their wellbeing. The higher the probability, the more likely the trade will arbitrarily stop.

        In the third step of the exchange procedure, after we know a pairwise combination ought to be exchanged but before the exchange occurs, we draw a random value between zero and one. If the draw value falls below strategic_error (i.e., within the CDF), trading of the current pairwise combination is halted. And the agents move onto trading the next pairwise combination of goods.

        strategic_error does two things:
        1. Conditional on a trading partner, strategic_error ensures that agent i does not just trade on a specific pairwise combination.
        2. Conditional on a pairwise combination that agent i prefers to make trades on, strategic_error ensures that agent i does this trade with more than one partner.
        '''
        if strategic_error is not None:
            if strategic_error >=1 or strategic_error < 0:
                raise ValueError(
                    "strategic_error, if provided, must be between (zero or higher) and (less than one)."
                )
        
        if trade_within_network:
            # is None doesn't work for dictionaries
            if not self.friends_dict:
                raise AttributeError(
                    "Network must be generated first with generate_network() if trade_in_network is set to true."
                )

        # reset transactions
        self.clear_transactions()
        
        initial_transaction_count = len(self.transacted_goods_tuple)
        
        for h in range(trading_days):
            self.loop_across_pairwise_agents(
                strategic_error, 
                plot_type,
                trade_within_network,
                )
            
            # Print number of transactions after each trading day
            current_transaction_count = len(self.transacted_goods_tuple)
            new_transactions = current_transaction_count - initial_transaction_count
            print(f"Trading day {h+1}: {current_transaction_count} total transactions, {new_transactions} since yesterday")
            initial_transaction_count = len(self.transacted_goods_tuple)
            
            # Update plot after every 10 iterations and at the end
            # if h % 100 == 0 or h == trading_days - 1:
            #     self.plot_first_ten_agents_inventory()
            #     time.sleep(2)

            # DEBUG: print shuffled_agents_index
            print(f"this is the {h}-th agents index: {self.shuffled_agents_index}")

    def loop_across_pairwise_agents(self, strategic_error=None,
            plot_type=None,
            trade_within_network: bool = False
            ):
        
        # Shuffle the agents index to randomize activation
        random.shuffle(self.shuffled_agents_index)
        # Store this activation order for future reference
        # Use list() to create a copy to prevent modifying history when shuffling again
        self.shuffled_agents_index_vintage.append(list(self.shuffled_agents_index))

        loops_across_agents = self.agent_count - 1
        for m in range(loops_across_agents):
            i = self.shuffled_agents_index[m]
            agent_i = self.agents[i]

            # Implement network at this stage to supplant the uniform pairing with other agents
            if trade_within_network:
                trading_partners = list(self.friends_dict[i])
                # Randomize friends list
                random.shuffle(trading_partners)
            else:
                trading_partners = (
                    self.shuffled_agents_index[m+1 : self.agent_count]
                )

            for i_plus in trading_partners:
                agent_i_plus = self.agents[i_plus]

                self.loop_across_pairwise_goods(
                    agent_i, agent_i_plus, strategic_error,
                    plot_type,
                    )
                
                if plot_type == "pairwise_goods":
                    self.plot_inventory_of_pairwise_goods()
                    time.sleep(5)
                elif plot_type == "inventory_timeseries":
                    self.plot_timeseries_of_good_inventory()
                    # time.sleep(1)
                elif plot_type == "aggregate_utility":
                    self.plot_aggregate_utility()
                    time.sleep(0.4)
                elif plot_type == "individual_utility":
                    self.plot_individual_utilities()
                    time.sleep(0.4)

    def loop_across_pairwise_goods(
            self, agent_i, agent_i_plus, strategic_error=None,
            plot_type=None,
            ):
        # Check arguments
        if not isinstance(agent_i, Agent) or (
            not isinstance(agent_i_plus, Agent)
            ):
            raise TypeError(
                "agent arguments must be instances of the Agent class"
                )
        random.shuffle(self.shuffled_goods_index)
        loops_across_goods = self.goods_type_count - 1
        for j in range(loops_across_goods):
            k = self.shuffled_goods_index[j]
            for jplus1 in range(j + 1, self.goods_type_count):
                k_plus = self.shuffled_goods_index[jplus1]

                self.loop_within_pairwise_goods(
                    agent_i, agent_i_plus, k, k_plus, strategic_error,
                    plot_type,
                    )
    
    def loop_within_pairwise_goods(
            self, agent_i, agent_i_plus, k, k_plus, strategic_error=None,
            plot_type=None,
            ):
        # Check arguments
        if not isinstance(agent_i, Agent) or (
            not isinstance(agent_i_plus, Agent)
            ):
            raise TypeError(
                "Arguments agent_i and agent_i_plus must be instances of the Agent class."
                )
        if not isinstance(k, int) or not isinstance(k_plus, int):
            raise TypeError("Arguments k and k_plus must be integers.")
        
        # Get agent i's MRS
        mrs_i = self.util_calc("mrs", agent_i, k, k_plus)
        # Get agent i+1's MRS
        mrs_iplus1 = self.util_calc("mrs", agent_i_plus, k, k_plus)
        # Determine who is buying k
        mrs_diff = mrs_i - mrs_iplus1
        
        # For debugging
        transaction_occurred = False
        max_iterations = 100  # Safety limit to prevent infinite loop
        iteration_count = 0
        
        while abs(mrs_diff) > 1e-5 and iteration_count < max_iterations:
            if mrs_diff > 0:
                buyer_of_k = agent_i
                seller_of_k = agent_i_plus
            else:
                buyer_of_k = agent_i_plus
                seller_of_k = agent_i
        
            buyer_decision = self.util_calc(
                "decide_to_trade", buyer_of_k, k, k_plus
                )
            seller_decision = self.util_calc(
                "decide_to_trade", seller_of_k, k_plus, k
                )

            # no need to check whether there is inventory because
            # agents will never start with less than one good, 
            # and if there is one good,
            # the decision will always be false.
            if buyer_decision and seller_decision:
                draw = random.uniform(0, 1)
                if (strategic_error is not None) and (draw < strategic_error):
                    break
                    # If strategic error is not specified, or
                    # the draw exceeds the threshold, python 
                    # proceeds to execute the following else block.
                else:
                    buyer_of_k.chg_inventory(k, 1)
                    buyer_of_k.chg_inventory(k_plus, -1)
                    seller_of_k.chg_inventory(k, -1)
                    seller_of_k.chg_inventory(k_plus, 1)

                    self.transacted_goods_tuple.append(
                        (k, k_plus)
                    )
                    self.transacting_agents_tuple.append(
                        (self.agents.index(buyer_of_k), 
                            self.agents.index(seller_of_k)
                        )
                    )
                    
                    mrs_diff = self.util_calc("mrs", agent_i, k, k_plus) - (
                        self.util_calc("mrs", agent_i_plus, k, k_plus)
                    )
                    transaction_occurred = True
                    iteration_count += 1
                    
                    self.record_inventory()
                    self.record_utility()
                    if (
                        len(self.transacted_goods_tuple) <= 10 and
                        plot_type == "edgeworth"
                        ):
                        self.plot_edgeworth(
                            self.agents.index(agent_i),
                            self.agents.index(agent_i_plus), 
                            k, 
                            k_plus)
                        time.sleep(4)
            else:
                break
                
        # For debugging - if we hit the max iterations, log this
        if iteration_count >= max_iterations:
            print(f"Warning: Max iterations reached in loop_within_pairwise_goods for goods {k} and {k_plus}")


    def util_calc(self, output, agent, good1_index, good2_index):
        '''
        This function makes calculations and decisions related to two goods. It can return a given agent's MRS between two goods. It can also decide whether an agent will be better off if it traded one good for another. 
        '''
        pref_1 = agent.get_pref(good1_index)
        pref_2 = agent.get_pref(good2_index)
        inventory_1 = agent.get_inventory(good1_index)
        inventory_2 = agent.get_inventory(good2_index)
        mrs = pref_1 * inventory_2 / (pref_2 * inventory_1)
        if output == "mrs":
            return mrs
        elif output == "decide_to_trade":
            inventory_1_after = inventory_1 + 1
            inventory_2_after = inventory_2 - 1
            util_before = pow(inventory_1, pref_1) * pow(inventory_2, pref_2)
            util_after = pow(inventory_1_after, pref_1) * (
                pow(inventory_2_after, pref_2)
            )
            if util_after >= util_before:
                return True
            else:
                return False
        else:
            raise TypeError("Output for the calculator isn't specified.")
        
    
    def transactions_to_dataframe(self, by="good"):
        '''
        This function takes transactions data (which are stored as a tuple) and converts them into a wide dataframe, where each row is indexted to transaction count and the columns are either by good or by agent. Each value reports the *cumulative* transaction count of either the good or the agent, as of a given transaction.
        '''
        if by == "good":
            transactions_tuple = self.transacted_goods_tuple
            type_length = self.goods_type_count
        elif by == "agent":
            transactions_tuple = self.transacting_agents_tuple
            type_length = self.agent_count
        else:
            raise ValueError("Parameter 'by' must be either 'good' or 'agent'")
     
        # Check if there are any transactions
        if not transactions_tuple:
            return pd.DataFrame(columns=list(range(type_length)))
            
        # Create lists for each column
        type1 = [pair[0] for pair in transactions_tuple]
        type2 = [pair[1] for pair in transactions_tuple]

        boolean_list = []
        for i in range(type_length):
            boolean_1 = [1 if value == i else 0 for value in type1]
            boolean_2 = [1 if value == i else 0 for value in type2]
            boolean_summed = [x + y for x, y in zip(boolean_1, boolean_2)]
            boolean_list.append(boolean_summed)

        # Transpose the list of lists
        tranposed_boolean_list_tuples = zip(*boolean_list)
        # Turn sub-tuples into sub-lists, and ensure object is a list.
        tranposed_boolean_list = list(map(list, tranposed_boolean_list_tuples))

        boolean_matrix = pd.DataFrame(
            tranposed_boolean_list, 
            columns = list(range(type_length))
                     )
        
        cumsum_matrix = boolean_matrix
        # Iterate across each row starting from the second
        for i in range(1, len(boolean_matrix)):
            # Use .iloc to operate on entire row (across all columns)
            cumsum_matrix.iloc[i] += cumsum_matrix.iloc[i-1]
        
        return cumsum_matrix

    def record_inventory(self):
        '''
        This helper function collects a snapshot of each agent's inventory, with which it updates the market's inventory_panel attribute. We plot the attribute after a transaction.
        '''

        matrix = []
        for agent in self.agents:
            vector = agent.get_entire_inventory()
            # Add an element to the vector whose value is the agent's index
            vector_with_agent = vector + [self.agents.index(agent)]
            matrix.append(vector_with_agent)
        
        # row_stack creates a wide table whose columns are goods
        matrix = np.row_stack(matrix)

        '''
        This if-statement records the time of corresponding to each row without the use of a counter.

        We count the row length of the inventory_panel attribute. Since row length must be (n * T), where n is the number of agents and T is time elasped. We divide row length by n to derive T.

        Then we create a 1 * n vector whose values are T. And we append it column-wise to the inventory matrix.

        Then we append the "timed" inventory matrix to the inventory panel.

        If there are no transactions in the panel, T must be one. We also want to prevent a division by zero error. So we manually set the element values of the time vector to one.
        '''
        if self.inventory_panel.size != 0:
            matrix_nrow = self.inventory_panel.shape[0]
            time = matrix_nrow/self.agent_count + 1
            time_vector = [time]*self.agent_count
            matrix_with_time = np.column_stack((matrix, time_vector))
            self.inventory_panel = np.row_stack(
                (self.inventory_panel, matrix_with_time)
                )
        else:
            time_vector = [1]*self.agent_count
            matrix_with_time = np.column_stack((matrix, time_vector))
            self.inventory_panel = matrix_with_time

    def plot_edgeworth(self, agent1, agent2, good1, good2):

        panel = self.convert_inventory_panel_to_dataframe()
        
        # Separate data for each agent
        agent1_data = panel[panel["agent"] == agent1]
        agent2_data = panel[panel["agent"] == agent2]
        
        # Create the scatter plot
        fig = go.Figure()

        # Use column names directly
        good1_name = f'good_{good1}'
        good2_name = f'good_{good2}'
        
        # Add trace for agent1
        fig.add_trace(go.Scatter(
            x=agent1_data[good1_name],
            y=agent1_data[good2_name],
            mode='markers',
            marker=dict(
                color=agent1_data['time'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Time (Agent 1)")
            ),
            name=f'Agent {agent1}'
        ))

        # Add trace for agent2
        fig.add_trace(go.Scatter(
            x=agent2_data[good1_name],
            y=agent2_data[good2_name],
            mode='markers',
            marker=dict(
                color=agent2_data['time'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Time (Agent 2)")
            ),
            name=f'Agent {agent2}'
        ))

        # Update layout to make the plot square
        fig.update_layout(
            title="Edgeworth Box Plot",
            xaxis_title=f"{good1_name} Inventory",
            yaxis_title=f"{good2_name} Inventory",
            hovermode='closest',
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            )
        )
        clear_output(wait=True)
        fig.show()

    def convert_inventory_panel_to_dataframe(self):
        # Turn .inventory_panel from a list to a dataframe whose size is (number of agents * ticks elapsed) * number of goods.
        panel = pd.DataFrame(self.inventory_panel)

        # The dataframe needs names. The names are the good index, plus agent and time for the last two columns.
        name_vector = []
        for i in range(self.goods_type_count):
            name = f'good_{i}'
            name_vector.append(name)

            if i == self.goods_type_count - 1:
                name_vector.extend(["agent", 'time'])
        panel.columns = name_vector

        return panel
        

    def plot_inventory_of_pairwise_goods(self):
        """
        Creates a biplot of the first twenty agents' inventory for 
        the first two goods using Plotly. 
        Replaces previous plot with the current one in notebook environments.
        """
        # Check if we have enough agents and goods to create the plot
        if len(self.agents) < 1 or self.goods_type_count < 2:
            return
        
        # Get number of agents to plot (up to 20)
        agents_to_plot = min(20, len(self.agents))
        
        # Create data for the plot
        x_values = []  # Good 0 inventory
        y_values = []  # Good 1 inventory
        agent_labels = []
        
        # Collect data for each agent
        for i in range(agents_to_plot):
            agent = self.agents[i]
            x_values.append(agent.get_inventory(0))
            y_values.append(agent.get_inventory(1))
            agent_labels.append(f"Agent {i}")
        
        # Create the scatter plot
        fig = go.FigureWidget()
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers+text',
            text=agent_labels,
            textposition="top center",
            marker=dict(
                size=12,
                color=list(range(agents_to_plot)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Agent Index")
            )
        ))
        
        # Update layout
        fig.update_layout(
            title="First 10 Agents' Inventory of Goods 0 and 1",
            xaxis_title="Good 0 Inventory",
            yaxis_title="Good 1 Inventory",
            hovermode='closest'
        )
        
        # Clear the output before displaying the new plot
        clear_output(wait=True)
        
        # Display the figure (replaces previous output)
        display(fig)

    def plot_timeseries_of_good_inventory(self, good: int = 0):
        '''
        This method plots one column (i.e., for one good) of the inventory_panel. Each line is one agent's inventory of that good over time.

        By default, it plots the holding of good 0.
        '''
        panel = self.convert_inventory_panel_to_dataframe()

        panel_select = panel[[f'good_{good}', 'agent', 'time']]
        
        # # Get unique time values and take only the last 20
        # time_values = sorted(panel_select["time"].unique())
        # if len(time_values) > 20:
        #     time_values = time_values[-20:]
        
        # # Filter data to only include the last 20 time points
        # panel_select = panel_select[panel_select["time"].isin(time_values)]

        # Create a list to store vectors by agent
        agents_data = []
        # Create series by agent
        for i in range(self.agent_count):
            df_filtered = panel_select[panel_select["agent"] == i]
            agents_data.append(df_filtered)

        # Create the plot
        fig = go.FigureWidget()

        # Call a palette
        palette = qualitative.Pastel

        # Loop to add time series
        for i in range(self.agent_count):
            agent_data = agents_data[i]

            fig.add_trace(go.Scatter(
                x=agent_data["time"],
                y=agent_data[f"good_{good}"],
                name=f"Agent {i}",
                mode='lines',
                # Cycle through the list of colors with the modulus operator
                line=dict(color=palette[i % len(palette)]),
            ))

        # Update layout
        fig.update_layout(
            title=f"Time Series of Good {good} Inventory by Agent",
            xaxis_title="Transactions",
            yaxis_title="Good Count",
            hovermode='x unified'
        )
        
        # Set x-axis range from 1 to minimum_length,
        # if the time series is shorter
        min_length = 100
        if agents_data[0].shape[0] < min_length:
            fig.update_xaxes(range=[1, min_length])
        
        # Clear the output before displaying the new plot
        clear_output(wait=True)
        
        # Display the figure (replaces previous output)
        display(fig)

    def record_utility(self):
        utils = []
        for agent in self.agents:
            inventory = agent.get_entire_inventory()
            preferences = agent.get_entire_pref()
            util = np.prod(np.power(inventory, preferences))
            utils.append(util)
        
        self.individual_utilities.append(utils)
        aggregate = sum(utils)
        self.aggregate_utility.append(aggregate)
    
    def plot_aggregate_utility(self):
        fig = go.FigureWidget()

        fig.add_trace(go.Scatter(
            y=self.aggregate_utility,
            x=list(range(1, len(self.aggregate_utility)+1)),
            mode='lines',
            line=dict(color="blue")
        ))

        fig.update_layout(
            title="Aggregate Utility",
            xaxis_title="Transactions",
            hovermode='closest'
        )
        
        # Set x-axis range from 1 to minimum_length,
        # if the time series is shorter
        min_length = 100
        if len(self.aggregate_utility) < min_length:
            fig.update_xaxes(range=[1, min_length])
        
        # Clear the output before displaying the new plot
        clear_output(wait=False)
        
        # Display the figure (replaces previous output)
        display(fig)

    def plot_individual_utilities(self):
        # Check if we have recorded any utilities
        if not self.individual_utilities:
            print("No utility data recorded yet.")
            return
            
        # Create a matrix whose columns are agents and rows are transaction number
        util_panel = np.row_stack(self.individual_utilities)

        # Create plot
        fig = go.FigureWidget()

        # Call a palette
        palette = qualitative.Pastel

        # Add n series for n agents with loop
        for i in range(self.agent_count):
            fig.add_trace(go.Scatter(
                x=list(range(1, util_panel.shape[0]+1)),
                y=util_panel[:,i],
                name=f"Agent {i}",
                mode='lines',
                line=dict(color=palette[i % len(palette)])
            ))

        fig.update_layout(
            title="Individual Utility Time Series",
            xaxis_title="Transactions",
            yaxis_title="Utility",
            hovermode='closest'
        )
        
        # Set x-axis range from 1 to minimum_length,
        # if the time series is shorter
        min_length = 100
        if util_panel.shape[0] < min_length:
            fig.update_xaxes(range=[1, min_length])
        
        # Clear the output before displaying the new plot
        clear_output(wait=False)
        
        # Display the figure (replaces previous output)
        display(fig)