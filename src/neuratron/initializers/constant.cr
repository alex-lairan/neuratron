module Neuratron
  module Initializers
    class Constant < Initializer
      def initialize(@value : Float64)
      end

      def call
        @value
      end
    end

    def self.ones
      Constant.new(1.0)
    end

    def self.zeros
      Constant.new(0.0)
    end
  end
end
