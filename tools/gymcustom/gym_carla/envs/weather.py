import math
import carla


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self._t)) - 20

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, weather: carla.WeatherParameters, update_freq, speed_factor):
        self.weather = weather
        self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self._storm = Storm(weather.precipitation)
        self.update_freq = update_freq
        self.speed_factor = speed_factor
        self.elapsed_time = 0

    def tick(self, delta_seconds, world):
        """
        Updates the weather for the given world if more than update_freq time has passed.
        Args:
            delta_seconds: how many seconds to add to the running time
            world: the world for which the weather needs to be changed

        Returns: None
        """
        self.elapsed_time += delta_seconds
        if self.elapsed_time >= self.update_freq:
            self._sun.tick(self.elapsed_time * self.speed_factor)
            self._storm.tick(self.elapsed_time * self.speed_factor)
            self.weather.cloudiness = self._storm.clouds
            self.weather.precipitation = self._storm.rain
            self.weather.precipitation_deposits = self._storm.puddles
            self.weather.wind_intensity = self._storm.wind
            self.weather.fog_density = self._storm.fog
            self.weather.wetness = self._storm.wetness
            self.weather.sun_azimuth_angle = self._sun.azimuth
            self.weather.sun_altitude_angle = self._sun.altitude

            world.set_weather(self.weather)
            self.elapsed_time = 0

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)
