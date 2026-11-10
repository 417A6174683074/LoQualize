To use this simulation space:
1. Create your movement in the movements folder, it should be a subclass of Mpvement2D and Space2D presented in the file space.py. 3D space are not yet implemented. An exemple of kalman estimation is presented in each movement file (sinusoid.py and circle.py). The main.py file is the same example as in circle.py.
2. Create your sensors in sensors.py
3. Create your kalman filter in general_filt.py. The kalman filters act as a second physical simulation compare to the movement file. In order to keep coherence the Kalmann filter should have less informations than the movement simulator otherwise it would simply be an omniscient computer, and all these fancy math are useless.
4. Use them in main.py

By default I implemented a circular and sinusoidal directions with IMU sensor and simple satellite localization, and an extended kalman filter adapted to those sensors (consider the acceleration, angular speed and position to predict and correct the position)

!!!Main issue for now: I have no idea about what would be coherent values regarding the precision/error of the devices


## Quantum-gravimeter-aided navigation (added)

The files import each other with relative imports (`from .sensors import ...`), like main.py,
so run them as a package FROM THE FOLDER ABOVE the repo (repo folder named LoQualize):

    cd ..
    python -m LoQualize.main_gravity                          # seamounts, navigation-grade IMU, 1 run
    python -m LoQualize.main_gravity --terrain flat           # flat gravity (map-matching has little to work with)
    python -m LoQualize.main_gravity --seeds 10 --no-show     # Monte Carlo, saves gravity_<terrain>_<grade>.png
    python -m LoQualize.main_gravity --grade tactical         # navigation | tactical | consumer

New files (all at the repo root, next to sensors.py)
- `gravity.py`      : true gravity field (simulator only) + gravity map with errors (filter only), Eotvos effect
- `movements/ship.py`: `ShipRoute`, AIS-like route (straight legs + smooth turns, ~10 kn, hours long)
- `sensors.py`      : new `Gravimeter2D(Measurement_device)` with the same `step()` / `observation()` / `R` interface
- `models.py`       : shared F and process noise for the 11-state model ("honest" mode, see below)
- `presets.py`      : realistic sensor values per IMU grade, GNSS and quantum gravimeter (answers "what values are coherent?")
- `main_gravity.py` : scenario, GNSS for 30 min then jammed; INS-only vs INS+gravimeter on identical sensor data

Changes to existing files
- `general_filt.py`: `np.eye(11)` -> `np.eye(len(self.x))`; optional innovation gate (a device with a `gate` attribute
  gets its measurement skipped when it is > gate sigmas from the prediction; needed for the nonlinear gravity map)


Each figure in the results_examples folder came from one command with 10 runs. They can be reproduced by:

gravity_rough_navigation.png	         python -m LoQualize.main_gravity --terrain rough --grade navigation --seeds 10
gravity_flat_navigation.png	             python -m LoQualize.main_gravity --terrain flat --grade navigation --seeds 10
gravity_rough_tactical.png	             python -m LoQualize.main_gravity --terrain rough --grade tactical --seeds 10