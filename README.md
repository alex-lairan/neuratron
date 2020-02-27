# Neuratron

Neuratron is a machine learning framework inspired by the Keras API.

With Neuratron, you can build, fit and use a deep neural network.

## Looking for contributors

A project of this scale cannot be done alone.
I'm looking to anyone who loves Crystal and Machine learning to provide some implementations.

The more implementation we have, the more this framework and Crystal will be attractive to data scientists.

## Installation

Neuratron is a fully crystal implementation.
There is no special requirement here.

## Usage

For this usage, we will take a full dense neural network that works for MNIST

You first need to create a model.

Currently, only the sequential model works.

```crystal
model = Neuratron::Models::Sequential.new
```

Then, you must add some layers to your model

```crystal
model.add(
  Neuratron::Layers::Dense.new(16, input_shape: [784],
    initializer: Neuratron::Initializers::Random.new(range: (-1.0..1.0)),
    activation: Neuratron::Activations::Tanh.new
  )
)
model.add(
  Neuratron::Layers::Dense.new(10,
    initializer: Neuratron::Initializers::Random.new(range: (-1.0..1.0)),
    activation: Neuratron::Activations::Tanh.new
  )
)
```

We use a `Tanh` activation, so the random initializer range should be `(-1.0..1.0)`.

We have to compile our model to build all weight and biases for our model.

```crystal
model.compile(
  loss: Neuratron::Losses::MAE.new,
  optimizer: Neuratron::Optimizers::SGD.new(momentum: 0.80),model.compile(
  loss: Neuratron::Losses::MAE.new,
  optimizer: Neuratron::Optimizers::SGD.new(momentum: 0.80),
  metrics: [Neuratron::Metrics::CategoricalAccuracy.new] of Neuratron::Metric
)

  metrics: [Neuratron::Metrics::CategoricalAccuracy.new] of Neuratron::Metric
)
```

We use the `sgd` optimizer.

All set, let's train our model!

```crystal
model.fit(
  images, # Array(LA::GMat(1, 764))
  labels, # Array(LA::GMat(1, 10))
  epochs: 10,
  batch_size: 2048,
)
```

After that, our model is ready to predict some numbers

```crystal
model.predict(images[0])
# => LA::GMat(1, 10)
```

## Development

If you have any knowledge of machine learning, don't hesitate to propose a new feature via issues, or contribute to current issues.

## Contributing

1. Fork it (<https://github.com/alex-lairan/neuratron/fork>)
2. Create your feature branch (`git checkout -b branch_name`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin branch_name`)
5. Create a new Pull Request

## Contributors

- [Alexandre Lairan](https://github.com/alex-lairan) - creator and maintainer
