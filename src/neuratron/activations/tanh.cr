module Neuratron
  module Activations
    class Tanh < Activation
      def call(x)
        Math.tanh(x)
      end

      def derivative(x)
        1 - (Math.tanh(x) ** 2)
      end
    end
  end
end
