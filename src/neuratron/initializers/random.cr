module Neuratron
  module Initializers
    class Random < Initializer
      def initialize(@generator = ::Random.new, @range = (0.0..1.0))
      end

      def call
        @generator.rand(@range)
      end
    end
  end
end
