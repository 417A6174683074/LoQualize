To use this simulation space:
1. Create your movement in the space.py file
2. Create your sensors in sensors.py
3. Create your kalman filter in general_filt.py
4. Use them in main.py

By default I implemented a circular direction with IMU sensor and simple satellite localization, and an extended kalman filter adapted to those sensors (consider the acceleration, angular speed and position to predict and correct the position)