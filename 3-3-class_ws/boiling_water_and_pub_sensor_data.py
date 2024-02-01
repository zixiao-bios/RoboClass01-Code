#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import Float64

# Set the simulation parameters
initial_temperature = 20  # Initial water temperature in Celsius
boiling_temperature = 100  # Boiling temperature of water in Celsius
simulation_duration = 10  # Duration for simulation in seconds


# Function to model the temperature change: an exponential approach to the boiling point
def temperature_function(time, initial_temp, boiling_temp, duration):
    # This is a simple model using an exponential function where the temperature approaches the boiling point asymptotically.
    # The rate parameter will control how fast the temperature approaches the boiling point.
    rate = 5 / duration  # Rate of temperature increase
    return boiling_temp - (boiling_temp - initial_temp) * np.exp(-rate * time)


def add_gaussian_noise(input_array, mean=0, std_dev=1) -> np.ndarray:
    noise = np.random.normal(mean, std_dev, input_array.shape)
    return input_array + noise


def add_env_noise(temp_array, temp_noise) -> np.ndarray:
    noise = np.linspace(0, temp_noise, temp_array.shape[0])
    return temp_array + noise


def main():
    rospy.init_node('water_sim')

    std_dev = rospy.get_param('/std_dev')
    time_step = rospy.get_param('/delta_t')

    time_array = np.arange(0, simulation_duration + time_step, time_step)
    temperature_array = temperature_function(time_array, initial_temperature, boiling_temperature, simulation_duration)
    real_temp_array = add_env_noise(temperature_array, -10)
    sensor_data_array = add_gaussian_noise(real_temp_array, 0, std_dev)

    pub = rospy.Publisher('water_temp', Float64, queue_size=10)
    rate = rospy.Rate(1 / time_step)

    rospy.sleep(2)

    for i in range(sensor_data_array.shape[0]):
        if rospy.is_shutdown():
            exit(0)
        
        pub.publish(sensor_data_array[i])
        rospy.loginfo(f'measure temp: {sensor_data_array[i]}')

        rate.sleep()


if __name__=='__main__':
    main()
