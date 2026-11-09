To use this simulation space:
1. Create your movement in the movements folder, it should be a subclass of Mpvement2D and Space2D presented in the file space.py. 3D space are not yet implemented. An exemple of kalman estimation is presented in each movement file (sinusoid.py and circle.py). The main.py file is the same example as in circle.py.
2. Create your sensors in sensors.py
3. Create your kalman filter in general_filt.py. The kalman filters act as a second physical simulation compare to the movement file. In order to keep coherence the Kalmann filter should have less informations than the movement simulator otherwise it would simply be an omniscient computer, and all these fancy math are useless.
4. Use them in main.py

By default I implemented a circular and sinusoidal directions with IMU sensor and simple satellite localization, and an extended kalman filter adapted to those sensors (consider the acceleration, angular speed and position to predict and correct the position)

!!!Main issue for now: I have no idea about what would be coherent values regarding the precision/error of the devices