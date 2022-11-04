# Bootstrapped and Bayesian Deep Q-Networks

## How to use
* **Set the number of antennas in the base station**. In `environment.py` change the line `self.M_ULA` to the values of your choice. The code expects M = 4, 8, 16, 32, and 64.
* **Run DQN variants algorithms**. Run the scripts `DQN`, `BoDQN`, `BaDQN`,  and `B2DQN.py` in folder `Codes`. The result is the same as that in folder `Results`.  
* **Show the results**. Run the script `Results_plot.ipynb` in folder `Results` to show `Figure 3`, `Figure 4`, and `Table IV` in the paper.
