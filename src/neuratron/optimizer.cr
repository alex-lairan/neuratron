module Neuratron
  abstract class Optimizer
    abstract def call(x_input, layers, expected, loss) : Array(LA::GMat)

    def batch_completed(changes)
    end
  end
end
