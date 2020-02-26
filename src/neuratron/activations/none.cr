module Neuratron
  module Activations
    class None < Activation
      def call(x)
        x
      end

      def derivative(x)
        1
      end
    end
  end
end
