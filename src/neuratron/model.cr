module Neuratron
  class Model
    @is_compiled = false
    @layers = Array(Layer).new

    @loss : Loss? = nil
    @optimizer : Optimizer? = nil

    def compile(loss, optimizer, metrics)
      puts "Compile model"
      @is_compiled = true
      @loss = loss
      @optimizer = optimizer
      @layers.each { |layer| layer.compile }
    end

    # TODO : Add validation data
    # TODO : Use callbacks for log
    def fit(x_data : Array(LA::GMat), y_data : Array(LA::GMat), epochs : UInt32, batch_size : UInt32, callbacks = Array(Neuratron::Callback).new)
      raise Errors::NotCompiled.new unless @is_compiled

      all_data = x_data.zip(y_data)

      epochs.times do |i|
        puts "Epochs #{i}"
        data_i = 0
        all_data.shuffle.each_slice(batch_size) do |batch|
          data_i += batch.size
          puts "\t(#{data_i} / #{all_data.size}) epoch #{i}"
          batch_changes = Array(Array(LA::GMat)).new
          channel = Channel(Array(LA::GMat)).new

          batch.each do |data, expected|
            spawn do
              all_x_s = [data] + x_from_input(data)
              changes = optimizer.call(all_x_s, @layers, expected, loss)
              channel.send changes
            end
          end

          batch.size.times do
            batch_changes << channel.receive
          end

          changes = batch_changes.reduce do |acc, changes|
            acc.zip(changes).map { |memo, change| memo + change }
          end.map do |changes|
            changes.map { |x| x / batch.size }
          end

          @layers.zip(changes).each do |layer, change_movement|
            layer.weight -= change_movement
          end
        end
      end
    end

    def evaluate
    end

    def predict(x_data : LA::GMat)
      x_from_input(x_data).last
    end

    private def x_from_input(input)
      @layers.map { |layer| input = layer.call(input)}
    end

    private def loss
      if loss = @loss
        loss
      else
        raise Errors::NotCompiled.new
      end
    end

    private def optimizer
      if optimizer = @optimizer
        optimizer
      else
        raise Errors::NotCompiled.new
      end
    end

    private def metrics
      @metrics
    end
  end
end
