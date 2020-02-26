module Neuratron
  module Losses
    # Mean Absolute Error
    class MAE < Loss
      def call(y_true, y_predict)
        (y_predict - y_true).map { |x| x.abs }
      end
    end
  end
end
