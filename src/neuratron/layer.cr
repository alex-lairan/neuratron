module Neuratron
  abstract class Layer
    @weight : LA::GMat?
    @initializer : Initializer

    getter activation : Activation

    getter units : Array(Int32)
    property input_shape : Array(Int32)

    def initialize(units : Int32, @input_shape = Array(Int32).new, @initializer = Initializers.zeros, @activation = Activations::None.new)
      @units = [1, units]
      @weight = nil
    end

    abstract def call(input : LA::GMat)

    def compile
      puts "\tCompile layer, #{input_shape}"
      raise Errors::NoInputShape.new if @input_shape.empty?
      @weight = LA::GMat.new(@input_shape.product, @units.product) { |i, j| @initializer.call }
    end

    def weight
      if weight = @weight
        weight
      else
        raise Errors::NotCompiled.new
      end
    end

    def weight=(w)
      @weight = w
    end
  end
end
