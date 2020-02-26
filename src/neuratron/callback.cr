module Neuratron
  abstract class Callback
    def on_epoch_begin(epoch)
    end

    def on_epoch_end(epoch)
    end

    def on_batch_begin(batch)
    end

    def on_batch_end(batch)
    end

    def on_train_batch_begin(batch)
    end

    def on_train_batch_end(batch)
    end

    def on_train_begin()
    end

    def on_train_end()
    end

  end
end
