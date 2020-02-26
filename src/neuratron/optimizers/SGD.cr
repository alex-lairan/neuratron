module Neuratron
  module Optimizers
    class SGD < Optimizer
      @previous_changes : Array(LA::GMat)?

      def initialize(@learning_rate = 0.1, @momentum = 0.0, @decay = 0.0)
        @previous_changes = nil
      end

      # SGD have too many responsabilities
      # and may work only for Dense layers
      def call(x_input, layers, expected, loss) : Array(LA::GMat)
        x_s = x_input.dup
        deltas = Array(LA::GMat).new

        # Φ : Activation function
        # Δinit = (d/dx Φ) ⊗ X - Y
        last_x = x_s.pop
        x_derivative = last_x.map { |x| layers.last.activation.derivative(x) }
        expectations = last_x - expected
        deltas << x_derivative.map_with_index { |x, i, j| x * expectations[i, j] }.transpose

        layers.reverse[0..-2].each do |current_layer|
          # Φ : Activation function
          # Δn-1 = (d/dx Φ) ⊗ Wn×Δn
          last_x = x_s.pop
          x_derivative = last_x.map { |x| current_layer.activation.derivative(x) }
          weight_activation = (current_layer.weight * deltas.last).transpose
          deltas << last_x.map { |x| current_layer.activation.derivative(x) }
                          .map_with_index { |x, i, j| x * weight_activation[i, j]  }
                          .transpose
        end

        deltas.reverse!

        if previous_changes = @previous_changes
          layers.zip(deltas, x_input, previous_changes).map do |layer, delta, x, momentum_weight|
            # α : Learning rate
            # β : Momentum rate
            # Wn - α×Xn+1×Δn + βΔw
            (delta * (x * @learning_rate)).transpose + @momentum * momentum_weight
          end
        else
          layers.zip(deltas, x_input).map do |layer, delta, x|
            # α : Learning rate
            # Wn - α×Xn+1×Δn
            (delta * (x * @learning_rate)).transpose
          end
        end.tap { |changes| @previous_changes = changes }
      end
    end
  end
end
