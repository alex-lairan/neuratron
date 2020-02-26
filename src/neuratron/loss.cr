module Neuratron
  abstract class Loss
    abstract def call(y_true, y_predict)
  end
end
