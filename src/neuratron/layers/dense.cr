module Neuratron
  module Layers
    class Dense < Layer
      def call(input : LA::GMat)
        (input * weight).map { |x| @activation.call(x) }
      end
    end
  end
end
