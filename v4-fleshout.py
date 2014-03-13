"""Attempt #4 at organizing neuron models

- We specify types of neurons using subclasses of Neuron
- This includes things like LIF vs HH and also Float vs Fixed, Rate vs Spiking
- We build a NeuronPool object which actually has code for running neurons
- We keep a list of known Neuron types around so if we're asked for just
  a Rate neuron, we can pick the first on on the list that matches

- Configuration of parameters is done via descriptors
- NeuronPools use multiple inheritence off neuron types
- build() step is delayed until after constructor, as we don't want that
  to happen until build time

- We initially construct a dummy class that can be fleshed out with
  the actual neuron model.  The dummy class would be made by the
  initial call to nengo.Ensemble() and it wouldn't get fleshed out with
  an actual backend's neural implementation until build time

"""

import numpy as np
import weakref

"""
Neuron type specifications
"""

class FloatParameter(object):
    def __init__(self, default, min=None, max=None):
        self.default = float(default)
        self.min = min
        self.max = max
        self.data = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance, self.default)

    def __set__(self, instance, value):
        if self.min is not None and value < self.min:
            raise AttributeError('parameter value must be >=%g' % self.min)
        if self.max is not None and value > self.max:
            raise AttributeError('parameter value must be <=%g' % self.max)
        self.data[instance] = float(value)


class Neuron(object):
    def __init__(self, **kwargs):
        self._allow_new_attributes = False
        for key, value in kwargs.items():
            setattr(self, key, value)
    def __setattr__(self, key, value):
        if (not key.startswith('_') and not self._allow_new_attributes
                and key not in dir(self)):
            raise AttributeError('Unknown parameter "%s"' % key)
        super(Neuron, self).__setattr__(key, value)


class LIF(Neuron):
    tau_rc = FloatParameter(0.02, min=0)
    tau_ref = FloatParameter(0.002, min=0)


class Rate(Neuron):
    pass


class Spiking(Neuron):
    pass


class Fixed(Neuron):
    pass


class Izhikevich(Neuron):
    a = FloatParameter(0.02)
    b = FloatParameter(0.2)
    c = FloatParameter(-65)
    d = FloatParameter(8)


"""
Base class for neuron pools

Pass in a list of neuron_types to set parameters
"""


class NeuronPool(Neuron):
    def __init__(self, neuron_types=None):
        self._allow_new_attributes = False
        for n in neuron_types:
            for key in dir(n):
                if not key.startswith('_'):
                    setattr(self, key, getattr(n, key))
        self._allow_new_attributes = True

    def build(self, n_neurons):
        raise NotImplementedError('NeuronPools must provide "make"')

    def step(self, dt, J):
        raise NotImplementedError('NeuronPools must provide "step"')



"""
Various neuron models
"""
class LIFRatePool(NeuronPool, LIF, Rate):
    def build(self, n_neurons):
        pass

    def step(self, dt, J):
        old = np.seterr(divide='ignore', invalid='ignore')
        try:
            r = 1.0 / (self.tau_ref + self.tau_rc * np.log1p(1.0 / (J-1)))
            r[J <= 1] = 0
        finally:
            np.seterr(**old)
        return r * dt   # multiply by dt to do rate per timestep


class LIFSpikingPool(NeuronPool, LIF, Spiking):
    def build(self, n_neurons):
        self.voltage = np.zeros(n_neurons)
        self.refractory_time = np.zeros(n_neurons)

    def step(self, dt, J):
        dv = (dt / self.tau_rc) * (J - self.voltage)
        self.voltage += dv

        self.voltage[self.voltage < 0] = 0

        self.refractory_time -= dt

        self.voltage *= (1-self.refractory_time / dt).clip(0, 1)

        spiked = self.voltage > 1

        overshoot = (self.voltage[spiked > 0] - 1) / dv[spiked > 0]
        spiketime = dt * (1 - overshoot)

        self.voltage[spiked > 0] = 0
        self.refractory_time[spiked > 0] = self.tau_ref + spiketime

        return spiked

class LIFFixedPool(NeuronPool, LIF, Spiking, Fixed):
    def build(self, n_neurons):
        self.voltage = np.zeros(n_neurons, dtype='i32')
        self.refractory_time = np.zeros(n_neurons, dtype='u8')

        self.dt = None
        self.lfsr = 1

    def step(self, dt, J):
        if self.dt != dt:
            self.dt = dt
            self.dt_over_tau_rc = int(dt * 0x10000 / self.tau_rc)
            self.ref_steps = int(self.tau_ref / dt)

        J = np.asarray(J * 0x10000, dtype='i32')

        dv = ((J - self.voltage) * self.dt_over_tau_rc) >> 16

        dv[self.refractory_time > 0] = 0

        self.refractory_time[self.refractory_time > 0] -= 1

        self.voltage += dv

        self.voltage[self.voltage < 0] = 0

        spiked = self.voltage > 0x10000

        self.refractory_time[spiked > 0] = self.ref_steps

        # randomly adjust the refractory period to account for overshoot
        for i in np.where(spiked > 0)[0]:
            p = ((self.voltage[i] - 0x10000) << 16) / dv[i]
            if self.lfsr < p:
                self.refractory_time[i] -= 1
            self.lfsr = (self.lfsr >> 1) ^ (-(self.lfsr & 0x1) & 0xB400)

        self.voltage[spiked > 0] = 0

        return spiked






class IzhikevichPool(NeuronPool, Izhikevich, Spiking):
    def build(self, n_neurons):
        self.v = np.zeros(n_neurons) + self.c
        self.u = self.b * self.v

    def step(self, dt, J):
        dv = (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + J) * 1000
        du = (self.a * (self.b * self.v - self.u)) * 1000

        self.v += dv * dt
        self.u += du * dt

        spiked = self.v >= 30
        self.v[spiked > 0] = self.c
        self.u[spiked > 0] = self.u[spiked > 0] + self.d

        return spiked


"""
List of known neuron models, in order of preference
"""
neuron_models = [
    LIFSpikingPool,
    LIFRatePool,
    LIFFixedPool,
    IzhikevichPool,
    ]

"""
Create a pool of neurons, given the required type specifications
"""
import inspect
def create(n_neurons, neuron_type):

    # make sure it's a list
    try:
        len(neuron_type)
    except TypeError:
        neuron_type = [neuron_type]

    # make sure elements in the list are instances, not classes
    for i, type in enumerate(neuron_type):
        if inspect.isclass(type):
            neuron_type[i] = type()

    # look through the list of neuron models to see if we can
    # find a match
    for model in neuron_models:
        for type in neuron_type:
            if not issubclass(model, type.__class__):
                break
        else:
            n = model(neuron_type)
            n.build(n_neurons)
            return n

    raise Exception('Could not find suitable neuron model')


if __name__ == '__main__':

    default = create(100, [])
    spiking = create(100, [LIF, Spiking])
    rate = create(100, [LIF, Rate])
    fixed = create(100, [LIF, Fixed])
    iz = create(100, [Izhikevich])
    #iz = create(100, [Izhikevich(a=0.02, b=0.2, c=-50, d=2)])


    J = np.linspace(-2, 10, 100)
    dt = 0.001
    T = 1
    default_data = []
    spiking_data = []
    rate_data = []
    iz_data = []
    fixed_data = []

    v = []
    for i in range(int(T/dt)):
        default_data.append(default.step(dt, J))
        spiking_data.append(spiking.step(dt, J))
        rate_data.append(rate.step(dt, J))
        iz_data.append(iz.step(dt, J))
        fixed_data.append(fixed.step(dt, J))
        v.append(fixed.voltage[-1])

    default_tuning = np.sum(default_data, axis=0)/T
    spiking_tuning = np.sum(spiking_data, axis=0)/T
    rate_tuning = np.sum(rate_data, axis=0)/T
    iz_tuning = np.sum(iz_data, axis=0)/T
    fixed_tuning = np.sum(fixed_data, axis=0)/T

    import pylab
    pylab.subplot(2, 1, 1)
    pylab.plot(J, default_tuning, label='default')
    pylab.plot(J, spiking_tuning, label='spiking LIF')
    pylab.plot(J, rate_tuning, label='rate LIF')
    pylab.plot(J, iz_tuning, label='Iz')
    pylab.plot(J, fixed_tuning, label='fixed LIF')
    pylab.legend(loc='best')
    pylab.subplot(2, 1, 2)
    pylab.plot(v)
    #pylab.plot(np.array(fixed_data)[:,-1])
    pylab.show()

