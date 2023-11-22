import matplotlib.pyplot as plt
import numpy as np

from trapesium import trapesium;
from segitiga import segitiga;


def plot_membership_function(x, membership_function, label, color):
    plt.plot(x, membership_function, color)
    plt.text(x[np.argmax(membership_function)], 0.8, label, 
             fontweight='bold', fontsize=9, color=color)

# Membership function temperature
x1 = np.arange(0, 111)
freezing = trapesium(x1, 0, 0, 30, 50)
cool = segitiga(x1, 30, 50, 70)
warm = segitiga(x1, 50, 70, 90)
hot = trapesium(x1, 70, 90, 110, 110)

plt.figure(1)
plt.title('Membership function temperature')
plt.xlabel('x1=temperature (F)')
plt.ylabel('Mu(x1)')
plt.grid(True)
plot_membership_function(x1, freezing, 'freezing', 'b')
plot_membership_function(x1, cool, 'cool', 'r')
plot_membership_function(x1, warm, 'warm', 'g')
plot_membership_function(x1, hot, 'hot', 'm')
plt.show()

# Membership function level kecerahan
x2 = np.arange(0, 101)
sunny = trapesium(x2, 0, 0, 20, 40)
cloudy = segitiga(x2, 20, 50, 80)
overcast = trapesium(x2, 60, 80, 100, 100)

plt.figure(2)
plt.title('Membership function level kecerahan')
plt.xlabel('x2=covering sky (%)')
plt.ylabel('Mu(x2)')
plt.grid(True)
plot_membership_function(x2, sunny, 'sunny', 'b')
plot_membership_function(x2, cloudy, 'cloudy', 'r')
plot_membership_function(x2, overcast, 'overcast', 'g')
plt.show()

# Membership function level speed
y = np.arange(0, 101)
slow = trapesium(y, 0, 0, 25, 75)
fast = trapesium(y, 25, 75, 100, 100)

plt.figure(3)
plt.title('Membership function level speed')
plt.xlabel('y=speed (mph)')
plt.ylabel('Mu(y)')
plt.grid(True)
plot_membership_function(y, slow, 'slow', 'b')
plot_membership_function(y, fast, 'fast', 'r')
plt.show()
