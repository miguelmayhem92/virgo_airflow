import pandas as pd
import numpy as np
import tensorflow as tf

from dateutil.relativedelta import relativedelta

class split_data():
    def __init__(self, data, configs):
        self.column_indices = {name: i for i, name in enumerate(data.columns)}
        self.ndata =len(data)
        self.train_df = data[0:int(self.ndata*configs['train'])]
        self.val_df = data[int(self.ndata*configs['train']):int(self.ndata*configs['val'])]
        self.test_df = data[int(self.ndata*configs['val']):]
        self.num_features = data.shape[1]

    def scaling(self):
        self.train_mean = self.train_df.mean()
        self.train_std = self.train_df.std()

        self.train_df = (self.train_df - self.train_mean) / self.train_std
        self.val_df = (self.val_df - self.train_mean) / self.train_std
        self.test_df = (self.test_df - self.train_mean) / self.train_std

### object for training
class WindowGenerator():

    def __init__(self, input_width, label_width, shift, total_data, raw_stock ,
                   train_df, val_df, test_df,
                   label_columns=None):
        # Store the raw data.
        self.raw_stock = raw_stock
        self.total_data = total_data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)  ### for each label column one dict with index
            }
            
        self.column_indices = {
            name: i for i, name in enumerate(train_df.columns) ### for each column observation one dict with index
        }

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width  # number of time steps to predict
        self.shift = shift  ## or offset

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]  ## indexes of the input data

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice] ## indexes of the label data

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]  ## select the train slice
        labels = features[:, self.labels_slice, :]  ## select the label slice
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)  ## selecting the column 

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
    
    def expected_return(
        self,model, train_mean , train_std , plot_col='stock_logdif'):

        data_ = self.total_data[-self.input_width:]
        data_ = (data_  - train_mean) / train_std
        data_ = data_.values
        data_ = data_.reshape((1,data_.shape[0],data_.shape[1]))
        plot_col_index = self.column_indices[plot_col]
        
        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        predictions = model.predict(data_)
        return predictions, data_

    def get_futur_prices(
        self,
        predictions,
        steps_futur, 
        stock_code,
        OUT_STEPS,
        train_std,
        train_mean,
        lag_days,
        n_days = 14,
        plot_col = 'stock_logdif'
        ):
    
        index = self.label_columns_indices.get(plot_col, None)

        prep_predictions = {
            'Date': [ pd.to_datetime(self.raw_stock['Date']).max() + relativedelta(days = i+1) for i in  range(OUT_STEPS)],
            'stock_price': [0]*steps_futur,
            'stock_stv': [0]*steps_futur,
            'stock_logdif': list(predictions[0,:,index]) 
        }
        prep_predictions = pd.DataFrame(prep_predictions)
        prep_predictions['stock_logdif'] = prep_predictions['stock_logdif'] * train_std[plot_col] + train_mean[plot_col]
        
        past_prices = self.raw_stock[-lag_days:]['stock_price'].values
        exp_returns = prep_predictions['stock_logdif'].values
        
        expt_price = list()

        for i in range(steps_futur):
            if i < len(past_prices):
                pred = np.exp(np.log(past_prices[i]) + exp_returns[i])
                expt_price.append(pred)
            else:
                j = i - len(past_prices)
                pred = np.exp(np.log(expt_price[j]) + exp_returns[i])
                expt_price.append(pred)
                
        prep_predictions['stock_price'] = expt_price
        prep_predictions['Type'] = 'Forecast'
        prep_predictions['StockCode'] = stock_code
        
        some_history = self.raw_stock[-(lag_days+n_days):].copy()
        some_history['Type'] = 'History'
        some_history['StockCode'] = stock_code
        
        final_ = pd.concat([some_history, prep_predictions]).reset_index(drop = True)
        final_ = final_[['Date','stock_price','Type','StockCode']]
        final_['ExecutionDate'] = pd.Timestamp.today().strftime('%Y-%m-%d')
        final_ = final_[-(steps_futur+4):]
        
        return final_, some_history, prep_predictions
    
    def get_metrics(self, model, source):

        if source == 'test':
            sample = self.test
        if source == 'validation':
            sample = self.val

        error = model.evaluate(sample)[1]     
        return error