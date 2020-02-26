module Neuratron
  module Losses
    # Mean Squared Error
    class MSE < Loss
      def call(y_true, y_predict)
        (y_predict - y_true).map { |x| x ** 2 }
      end
    end
  end
end
