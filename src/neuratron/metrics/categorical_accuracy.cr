module Neuratron
  module Metrics
    class CategoricalAccuracy < Metric
      def call(y_true, y_predict)
        argmax(y_true.to_a) == argmax(y_predict.to_a)
      end

      private def argmax(array : Array(Float64))
        max = array[0];
        index = 0;
        array.each_with_index do |e, i|
          if e > max
            max = e
            index = i
          end
        end
        index
      end
    end
  end
end
