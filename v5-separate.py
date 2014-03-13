"""Attempt #5 at organizing neuron models

- We specify types of neurons using subclasses of Neuron
- This includes things like LIF vs HH and also Float vs Fixed, Rate vs Spiking
- We build a NeuronPool object which actually has code for running neurons
- We keep a list of known Neuron types around so if we're asked for just
  a Rate neuron, we can pick the first on on the list that matches

- Configuration of parameters is done via descriptors
- make() step is delayed until after constructor, as we don't want that
  to happen until build time

- We initially construct a dummy class that can be fleshed out with
  the actual neuron model.  The dummy class would be made by the
  initial call to nengo.Ensemble() and it wouldn't get fleshed out with
  an actual backend's neural implementation until build time

- We don't want the actual backend's class for running neurons to be
  a subclass of NeuronPool, since that's putting a lot of constraints
  on it.  Instead, we just decorate the classes to indicate what they
  support.

"""

import numpy as np
import weakref
import inspect

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
        if instance is None: return self.default
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
This is the class that should be created by an Ensemble during model
constructon.  A backend's builder can call build() on this, pass in a
list of models it knows about, and get a constructed object.
"""
class NeuronPoolSpecification(object):
    def __init__(self, n_neurons, neuron_types):
        self._allow_new_attributes = True
        self.n_neurons = n_neurons
        # make sure it's a list
        try:
            len(neuron_types)
        except TypeError:
            neuron_types = [neuron_types]
        # make sure elements in the list are instances, not classes
        for i, type in enumerate(neuron_types):
            if inspect.isclass(type):
                neuron_types[i] = type()

        self.neuron_types = neuron_types
        for n in neuron_types:
            for key in dir(n):
                if not key.startswith('_'):
                    setattr(self, key, getattr(n, key))
        self._allow_new_attributes = False

    def __setattr__(self, key, value):
        if (not key.startswith('_') and not self._allow_new_attributes
                and key not in dir(self)):
            raise AttributeError('Unknown parameter "%s"' % key)
        super(NeuronPoolSpecification, self).__setattr__(key, value)

    def build(self, pool_classes):
        # look through the list of neuron models to see if we can
        # find a match
        for model in pool_classes:
            params = {}
            for type in self.neuron_types:
                if not type.__class__ in model.neuron_types:
                    break
            else:
                for cls in model.neuron_types:
                    for key in dir(cls):
                        if not key.startswith('_'):
                            params[key] = getattr(self, key, getattr(cls, key))
                n = model()
                for key, value in params.items():
                    setattr(n, key, value)
                return n

        raise Exception('Could not find suitable neuron model')






"""
Backend-specific neuron models
"""


class NeuronPool(object):
    def make(self, n_neurons):
        raise NotImplementedError('NeuronPools must provide "make"')

    def step(self, dt, J):
        raise NotImplementedError('NeuronPools must provide "step"')


def implements(*neuron_types):
    def wrapper(klass):
        klass.neuron_types = neuron_types
        return klass
    return wrapper




@implements(LIF, Rate)
class LIFRatePool(NeuronPool):
    def make(self, n_neurons):
        pass

    def step(self, dt, J):
        old = np.seterr(divide='ignore', invalid='ignore')
        try:
            r = 1.0 / (self.tau_ref + self.tau_rc * np.log1p(1.0 / (J-1)))
            r[J <= 1] = 0
        finally:
            np.seterr(**old)
        return r * dt   # multiply by dt to do rate per timestep


@implements(LIF, Spiking)
class LIFSpikingPool(NeuronPool):
    def make(self, n_neurons):
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


@implements(LIF, Spiking, Fixed)
class LIFFixedPool(NeuronPool):
    def make(self, n_neurons):
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




@implements(Izhikevich, Spiking)
class IzhikevichPool(NeuronPool):
    def make(self, n_neurons):
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


if __name__ == '__main__':


    specs = {
            'default': [],
            'LIF spiking': [LIF, Spiking],
            'LIF rate': [LIF, Rate],
            'LIF fixed': [LIF, Fixed],
            'Iz': [Izhikevich],
            'Iz burst': [Izhikevich(a=0.02, b=0.2, c=-50, d=2)],
            }

    J = np.linspace(-2, 10, 100)
    dt = 0.001
    T = 1
    import pylab

    for name, spec in specs.items():
        pool_spec = NeuronPoolSpecification(100, spec)

        # you can change a parameter before build time
        if name=='LIF rate':
            pool_spec.tau_rc = 0.05

        spec_model = pool_spec.build(neuron_models)
        spec_model.make(pool_spec.n_neurons)

        data = []

        for i in range(int(T/dt)):
            data.append(spec_model.step(dt, J))

        tuning = np.sum(data, axis=0)/T

        pylab.plot(J, tuning, label=name)
    pylab.legend(loc='best')
    pylab.show()

