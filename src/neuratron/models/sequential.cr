module Neuratron
  module Models
    class Sequential < Model
      def add(layer : Neuratron::Layer)
        layer.input_shape = @layers.last.units unless @layers.empty?
        @layers << layer
      end
    end
  end
end
